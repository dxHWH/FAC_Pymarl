import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class RSSMWorldModel(nn.Module):
    def __init__(self, input_dim, args):
        super(RSSMWorldModel, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if args.use_cuda else 'cpu')

        # === 1. 维度解析 ===
        # input_dim = State_Dim + (N_Agents * N_Actions)
        if isinstance(args.state_shape, int):
            self.state_dim = args.state_shape
        else:
            self.state_dim = int(args.state_shape[0])
            
        self.action_dim = args.n_agents * args.n_actions
        
        # 验证维度
        assert input_dim == self.state_dim + self.action_dim, "RSSM Input dim mismatch!"

        # RSSM 专属维度配置
        self.hidden_dim = args.rssm_hidden_dim  # Deterministic h_t
        self.latent_dim = args.rssm_latent_dim  # Stochastic z_t
        self.embed_dim = getattr(args, "rssm_embed_dim", 256)
        self.min_std = 0.1

        # === 2. 网络组件 ===
        
        # (a) Encoder: 压缩 State 和 Action
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ELU()
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.embed_dim),
            nn.ELU()
        )

        # (b) Recurrent Model (Deterministic Path): h_t -> h_{t+1}
        # Input: h_t, z_t, a_t (Input is concat of z and a)
        self.rnn_cell = nn.GRUCell(self.latent_dim + self.embed_dim, self.hidden_dim)

        # (c) Transition Model (Prior): p(z_t | h_t)
        # 仅基于历史记忆 h_t 预测当前的 z_t 分布
        self.prior_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.embed_dim),
            nn.ELU(),
            nn.Linear(self.embed_dim, 2 * self.latent_dim) # Mean + LogVar
        )

        # (d) Representation Model (Posterior): q(z_t | h_t, s_t)
        # 结合 历史记忆 h_t 和 当前观测 s_t 推断 z_t
        self.posterior_net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.embed_dim, self.embed_dim),
            nn.ELU(),
            nn.Linear(self.embed_dim, 2 * self.latent_dim) # Mean + LogVar
        )

        # (e) State Predictor (Decoder): Predict S_{t+1} from h_{t+1} and z_t
        # h_{t+1} 包含了 a_t，是预测下一帧的关键
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, self.embed_dim),
            nn.ELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ELU(),
            nn.Linear(self.embed_dim, self.state_dim)
        )

        # (f) Reward Predictor
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, self.embed_dim),
            nn.ELU(),
            nn.Linear(self.embed_dim, 1)
        )

    def init_hidden(self, batch_size, device):
        # 返回 RNN 的初始状态 h_0
        return torch.zeros(batch_size, self.hidden_dim).to(device)

    def forward_step(self, obs_input, prev_rnn_state):
        """
        单步前向传播 (Step t)
        obs_input: [B, State + Action] -> 当前时刻 t 的观测和动作
        prev_rnn_state: [B, Hidden] -> 上一时刻的确定性状态 h_t
        """
        # 1. 拆分输入
        state_t = obs_input[:, :self.state_dim]
        action_t = obs_input[:, self.state_dim:]

        # 2. Embedding
        s_embed = self.state_encoder(state_t)
        a_embed = self.action_encoder(action_t)

        # 3. Posterior Inference (q(z_t | h_t, s_t))
        # 利用 h_t (来自 t-1 的记忆) 和 s_t (当前观测)
        post_in = torch.cat([prev_rnn_state, s_embed], dim=-1)
        post_stats = self.posterior_net(post_in)
        post_mu, post_logvar = torch.chunk(post_stats, 2, dim=-1)
        
        # 4. Prior Inference (p(z_t | h_t))
        # 仅利用 h_t (用于计算 KL Loss)
        prior_stats = self.prior_net(prev_rnn_state)
        prior_mu, prior_logvar = torch.chunk(prior_stats, 2, dim=-1)

        # 5. Sample z_t (Reparameterization)
        # 训练时使用 Posterior 采样
        z_t = self._sample_z(post_mu, post_logvar)

        # 6. Update RNN State (Deterministic Path)
        # h_{t+1} = GRU(h_t, z_t, a_t)
        rnn_in = torch.cat([z_t, a_embed], dim=-1)
        next_rnn_state = self.rnn_cell(rnn_in, prev_rnn_state)

        # 7. 打包返回
        # [Critical Change]: 必须返回 prev_rnn_state (h_t) 供 sample_latents 使用
        # 避免因果泄露 (Causality Leak)
        return next_rnn_state, (prior_mu, prior_logvar, post_mu, post_logvar, z_t, next_rnn_state, prev_rnn_state)

    def sample_latents(self, dist_params, num_samples=1):
        """
        返回给 Mixer 使用的特征。
        为了严格遵守因果性 (Causality) 和 CTDE 假设，
        必须使用 [h_t, z_t] 而非 [h_{t+1}, z_t]。
        """
        # 解包 (忽略不需要的部分)
        _, _, _, _, z_t, next_rnn_state, prev_rnn_state = dist_params
        
        # Concat [h_t, z_t]
        # prev_rnn_state (h_t) 是动作发生前的状态，z_t 是当前的观测特征
        representation = torch.cat([prev_rnn_state, z_t], dim=-1)
        
        # 增加维度适配 Learner [D, B, Latent] (D=1 for RSSM)
        return representation.unsqueeze(0)

    def infer_posterior(self, hidden_state, state):
        """
        辅助函数：仅推断后验 z_t，不更新 RNN。
        用于 Learner 在处理序列最后一步 (T) 时，补全 z_T。
        此时只有 s_T，没有 a_T，无法计算 h_{T+1}。
        """
        s_embed = self.state_encoder(state)
        
        # q(z_T | h_T, s_T)
        post_in = torch.cat([hidden_state, s_embed], dim=-1)
        post_stats = self.posterior_net(post_in)
        post_mu, post_logvar = torch.chunk(post_stats, 2, dim=-1)
        
        z_t = self._sample_z(post_mu, post_logvar)
        
        return {
            'z': z_t,
            'mu': post_mu,
            'logvar': post_logvar
        }

    def compute_loss(self, forward_outputs_list, target_states, target_rewards=None):
        """
        批量计算整个序列的 Loss
        """
        prior_means, prior_logvars = [], []
        post_means, post_logvars = [], []
        z_samples = []
        h_nexts = [] # h_{t+1} 用于 Decoder 重建 s_{t+1}

        for out in forward_outputs_list:
            _, params = out
            # 解包所有参数
            p_mu, p_lv, q_mu, q_lv, z, h_next, h_prev = params
            
            prior_means.append(p_mu)
            prior_logvars.append(p_lv)
            post_means.append(q_mu)
            post_logvars.append(q_lv)
            z_samples.append(z)
            h_nexts.append(h_next)

        # Stack Time dimension [B, T, ...]
        h_next = torch.stack(h_nexts, dim=1)
        z = torch.stack(z_samples, dim=1)
        p_mu = torch.stack(prior_means, dim=1)
        p_lv = torch.stack(prior_logvars, dim=1)
        q_mu = torch.stack(post_means, dim=1)
        q_lv = torch.stack(post_logvars, dim=1)

        # 1. Reconstruction Loss (Predict S_{t+1})
        # 使用 h_{t+1} (包含 a_t) 和 z_t 来预测
        dec_in = torch.cat([h_next, z], dim=-1)
        pred_next_state = self.decoder(dec_in)
        
        recon_loss = F.mse_loss(pred_next_state, target_states, reduction='none').mean(dim=-1)

        # 2. KL Divergence Loss
        kl_loss = self._kl_divergence(q_mu, q_lv, p_mu, p_lv)

        # 3. Reward Prediction Loss
        reward_loss = torch.zeros_like(recon_loss)
        if target_rewards is not None:
            pred_rew = self.reward_head(dec_in)
            reward_loss = F.mse_loss(pred_rew, target_rewards, reduction='none').squeeze(-1)

        return recon_loss, kl_loss, reward_loss

    def predict(self, obs_input, hidden_state, use_mean=True):
        """
        用于评估时的单步预测
        """
        next_hidden, params = self.forward_step(obs_input, hidden_state)
        # 解包 (注意结构已变)
        _, _, post_mu, post_logvar, z_t, h_next, h_prev = params
        
        # 评估时通常使用均值
        z = post_mu if use_mean else z_t
            
        dec_in = torch.cat([h_next, z], dim=-1)
        pred_next_state = self.decoder(dec_in)
        
        return pred_next_state, next_hidden

    def _sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _kl_divergence(self, mu1, lv1, mu2, lv2):
        # KL(N(mu1, lv1) || N(mu2, lv2))
        var1 = torch.exp(lv1)
        var2 = torch.exp(lv2)
        kl = 0.5 * (var1 / var2 + (mu2 - mu1)**2 / var2 - 1 + lv2 - lv1)
        return kl.sum(dim=-1)