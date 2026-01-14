import copy
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F

from components.episode_buffer import EpisodeBatch
from modules.mixers.dvd_wm_fac_mixer import DVDWMFacMixer
from modules.world_models.vae_rnn_fac import FactorizedVAERNN
from utils.rl_utils import build_td_lambda_targets

class DVDWMFacLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')

        # ----------------------------------------------------------------------
        # 1. 初始化 Mixer (混合网络)
        # ----------------------------------------------------------------------
        self.mixer = None
        if args.mixer == "dvd_wm_fac":
            self.mixer = DVDWMFacMixer(args)
        else:
            raise ValueError(f"Mixer {args.mixer} not recognised for DVDWMFacLearner.")
        
        self.target_mixer = copy.deepcopy(self.mixer)

        # ----------------------------------------------------------------------
        # 2. 初始化 World Model (环境模型/VAE)
        # ----------------------------------------------------------------------
        self.obs_dim = scheme["obs"]["vshape"]
        self.act_dim = scheme["actions_onehot"]["vshape"][0]
        
        input_shape = self.obs_dim + self.act_dim
        output_shape = self.obs_dim
        
        self.world_model = FactorizedVAERNN(args, input_shape, output_shape)
        self.world_model.to(self.device)

        # ----------------------------------------------------------------------
        # 3. 定义优化器 (分离 RL 和 WM 的参数)
        # ----------------------------------------------------------------------
        self.rl_params = list(mac.parameters()) + list(self.mixer.parameters())
        self.wm_params = list(self.world_model.parameters())

        # RL Optimizer
        if args.optimizer == 'adam':
            self.optimiser = Adam(params=self.rl_params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.rl_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # WM Optimizer (通常使用 Adam 以获得更好的收敛性)
        self.wm_optimiser = Adam(params=self.wm_params, lr=args.wm_lr if hasattr(args, 'wm_lr') else args.lr)

        # ----------------------------------------------------------------------
        # 4. 其他配置
        # ----------------------------------------------------------------------
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        
        # [核心优化] 获取 RL 专用的 Batch Size
        # 默认情况下，如果未指定 rl_batch_size，则使用 args.batch_size (即不切片)
        self.rl_batch_size = getattr(args, "rl_batch_size", args.batch_size)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        """
        训练函数执行两阶段更新：
        Phase 1: 使用完整的 batch (大 Batch) 更新 World Model。
        Phase 2: 对 batch 进行切片 (小 Batch)，使用更新后的 WM 重新计算 Latent，更新 RL Agent。
        """
        
        # 将整个 Batch 数据移到 GPU (WM 训练需要)
        # batch 维度: [Full_Batch, Seq, N, Dim]
        full_obs = batch["obs"].to(self.device)
        full_actions = batch["actions_onehot"].to(self.device)

        # ======================================================================
        # Phase 1: World Model Update (Dynamics Learning) - 大 Batch
        # ======================================================================
        
        # 1.1 WM 前向传播 (Full Batch)
        # z_all: [Full_Batch, Seq, N, Z_dim]
        # recon_all: [Full_Batch, Seq, N, Obs_dim]
        z_all_full, recon_all_full, mu_full, logvar_full, _ = self.world_model(full_obs, full_actions)
        
        # 1.2 准备 WM Loss 计算数据 (预测未来: Input[:,-1] -> Target[:,1:])
        obs_input = full_obs[:, :-1]
        act_input = full_actions[:, :-1]
        target_obs = full_obs[:, 1:]
        
        # 重新跑一次切片后的前向，保证 Loss 计算的时间步对齐
        # recon_out: [Full_Batch, Seq-1, N, Obs_Dim]
        _, recon_out, mu_train, logvar_train, _ = self.world_model(obs_input, act_input)
        
        # 1.3 计算 Losses
        # Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(recon_out, target_obs, reduction='none')
        recon_loss = recon_loss.sum(dim=-1).mean() # Sum over features, Mean over rest
        
        # KL Divergence
        kld_loss = -0.5 * th.sum(1 + logvar_train - mu_train.pow(2) - logvar_train.exp(), dim=-1)
        kld_loss = kld_loss.mean()
        
        # Total WM Loss
        beta = getattr(self.args, "wm_kl_beta", getattr(self.args, "beta", 0.1))
        wm_loss = recon_loss + beta * kld_loss

        # 1.4 更新 WM 参数
        self.wm_optimiser.zero_grad()
        wm_loss.backward()
        th.nn.utils.clip_grad_norm_(self.wm_params, self.args.grad_norm_clip)
        self.wm_optimiser.step()

        # ======================================================================
        # Phase 2: RL Agent Update (Q-Learning) - 小 Batch & 新 Latent
        # ======================================================================

        # 2.1 Batch 切片 (Slicing)
        # 我们只取前 self.rl_batch_size 个样本用于 RL 训练
        # 这样做既利用了大 Batch 训练 WM，又节省了 RL 反向传播的显存和计算
        rl_batch = batch[:self.rl_batch_size]
        
        # 2.2 [关键步骤] Re-Encoding (二次前向)
        # 使用刚刚更新过的 self.world_model 对 rl_batch 进行再次编码
        # 目的：确保 RL 获得的 z 是基于最新 WM 参数生成的 (减少非平稳性)
        if self.rl_batch_size < batch.batch_size:
            # 如果切片了，需要从原始数据中也切片出来再送入 WM
            rl_obs = full_obs[:self.rl_batch_size]
            rl_actions = full_actions[:self.rl_batch_size]
            
            # 关闭梯度计算，因为 RL 训练通常不反传给 WM (通过 detach 实现，这里 no_grad 为了省显存)
            with th.no_grad():
                z_all_rl, _, _, _, _ = self.world_model(rl_obs, rl_actions)
        else:
            # 如果没有切片 (rl_batch_size == batch_size)，直接复用上面的结果
            # 注意：理论上应该重新计算，因为 WM 参数变了。但为了效率，
            # 如果不严格要求 update 后的 z，可以复用 z_all_full 并 detach。
            # 鉴于我们追求"严谨"，这里建议重新计算，或者接受 Phase 1 的 z (稍微陈旧一步)。
            # 为了完全符合"使用更新后参数"的逻辑，我们在不切片时也应该重新计算，或者就在上面 detach。
            # 这里我们为了逻辑统一，选择复用 detach 后的旧值 (如果想要极致性能)，
            # 或者重新计算 (如果想要极致准确)。鉴于代码逻辑是"切片优化"，我们假定重新计算开销可控。
            with th.no_grad():
                z_all_rl, _, _, _, _ = self.world_model(rl_obs, rl_actions) if 'rl_obs' in locals() else self.world_model(full_obs, full_actions)

        # 2.3 准备 RL 训练数据 (使用切片后的 rl_batch)
        rewards = rl_batch["reward"][:, :-1]
        actions = rl_batch["actions"][:, :-1]
        terminated = rl_batch["terminated"][:, :-1].float()
        mask = rl_batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = rl_batch["avail_actions"]
        
        # 2.4 准备 Latent Z (Deconfounding 输入)
        # z_all_rl: [RL_Batch, Seq, N, Z]
        # 必须 detach，阻断 RL 梯度回传给 WM
        z_curr = z_all_rl[:, :-1]   # t=0 ~ T-1
        z_next = z_all_rl[:, 1:]    # t=1 ~ T
        
        # Z-Warmup (可选)
        warmup_coef = 1.0
        if getattr(self.args, "use_z_warmup", False):
            if t_env < self.args.z_warmup_steps:
                warmup_coef = float(t_env) / float(self.args.z_warmup_steps)
        
        z_curr = z_curr * warmup_coef
        z_next = z_next * warmup_coef

        # 2.5 Calculate Estimated Q-Values (Mac Forward)
        mac_out = []
        self.mac.init_hidden(rl_batch.batch_size) # 注意使用 rl_batch.batch_size
        for t in range(rl_batch.max_seq_length):
            agent_outs = self.mac.forward(rl_batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1) # [RL_Batch, T, N, A]
        
        # Pick Q-values for chosen actions
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # 2.6 Calculate Target Q-Values (Double Q-Learning)
        with th.no_grad():
            self.target_mac.init_hidden(rl_batch.batch_size)
            target_mac_out = []
            for t in range(rl_batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(rl_batch, t=t)
                target_mac_out.append(target_agent_outs)
            target_mac_out = th.stack(target_mac_out, dim=1)
            
            # Mask out unavailable actions
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            
            # Greedy action selection
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out[:, 1:], dim=3, index=cur_max_actions).squeeze(3)
            
            # Target Mixer: 传入 State[:, 1:] 和 z_next
            # 注意：rl_batch["state"] 
            z_next_detached = z_next.detach()
            target_max_qvals = self.target_mixer(target_max_qvals, rl_batch["state"][:, 1:], z_next_detached)

        # 2.7 Mixer Forward
        # 传入 State[:, :-1] 和 z_curr
        chosen_action_qvals = self.mixer(chosen_action_qvals, rl_batch["state"][:, :-1], z_curr)

        # 2.8 TD Error Calculation
        targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                        self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # 维度对齐 (防止 seq_len 不一致)
        min_t = min(chosen_action_qvals.shape[1], targets.shape[1])
        chosen_action_qvals = chosen_action_qvals[:, :min_t]
        targets = targets[:, :min_t]
        mask = mask[:, :min_t]

        td_error = (chosen_action_qvals - targets.detach())
        masked_td_error = td_error * mask.expand_as(td_error)
        loss_td = (masked_td_error ** 2).sum() / mask.sum()

        # 2.9 更新 RL 参数
        self.optimiser.zero_grad()
        loss_td.backward()
        th.nn.utils.clip_grad_norm_(self.rl_params, self.args.grad_norm_clip)
        self.optimiser.step()

        # ----------------------------------------------------------------------
        # Phase 3: Logging & Updates
        # ----------------------------------------------------------------------
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss_td.item(), t_env)
            self.logger.log_stat("loss_wm", wm_loss.item(), t_env)
            self.logger.log_stat("wm_recon_loss", recon_loss.item(), t_env)
            self.logger.log_stat("wm_kld_loss", kld_loss.item(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def evaluate_world_model(self, batch, t_env):
        """
        评估 WM 性能: 局部观测重构误差 MSE
        """
        obs = batch["obs"].to(self.device)
        actions = batch["actions_onehot"].to(self.device)
        mask = batch["filled"][:, :-1].float().squeeze(-1)
        
        obs_input = obs[:, :-1]
        act_input = actions[:, :-1]
        target_obs = obs[:, 1:]

        self.world_model.eval()
        
        with th.no_grad():
            _, recon_out, _, _, _ = self.world_model(obs_input, act_input)
            
            # 计算 MSE
            errors = (recon_out - target_obs) ** 2
            mse_per_step = errors.mean(dim=-1).mean(dim=-1) # Mean over Obs & Agents
            
            masked_mse = (mse_per_step * mask).sum() / mask.sum()

            # 简单的 R2 计算
            target_flat = target_obs.reshape(-1, self.obs_dim)
            pred_flat = recon_out.reshape(-1, self.obs_dim)
            ss_res = ((target_flat - pred_flat) ** 2).sum()
            ss_tot = ((target_flat - target_flat.mean(0)) ** 2).sum()
            r2 = 1 - (ss_res / (ss_tot + 1e-8))

            log_prefix = "test_wm_"
            self.logger.log_stat(log_prefix + "mse", masked_mse.item(), t_env)
            self.logger.log_stat(log_prefix + "r2", r2.item(), t_env)
            
            print(f"\n[Test WM] MSE: {masked_mse.item():.6f} | R2: {r2.item():.6f}")

        self.world_model.train()

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        self.world_model.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.wm_optimiser.state_dict(), "{}/wm_opt.th".format(path))
        th.save(self.world_model.state_dict(), "{}/world_model.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.wm_optimiser.load_state_dict(th.load("{}/wm_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.world_model.load_state_dict(th.load("{}/world_model.th".format(path), map_location=lambda storage, loc: storage))