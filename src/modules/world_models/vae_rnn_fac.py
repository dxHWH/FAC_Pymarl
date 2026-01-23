# src/modules/world_models/vae_rnn_fac.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight) # 正交初始化
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)


class FactorizedVAERNN(nn.Module):
    """
    Factorized VAE-RNN World Model (因式分解 VAE-RNN 世界模型)
    
    修改日志:
    - [Fix] 加入 Agent ID (One-Hot) 到 Encoder 和 Decoder 输入，解决多智能体共享参数导致的重构震荡。
    """

    def __init__(self, args, input_shape, output_shape):
        """
        Args:
            args: 全局参数配置对象。
            input_shape (int): Encoder 输入维度 = obs_dim + n_actions
            output_shape (int): Decoder 输出维度 = obs_dim (预测下一帧局部观测)
        """
        super(FactorizedVAERNN, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_agents = args.n_agents  # 获取智能体数量用于 One-Hot
        
        # [!!! 核心修复 !!!] 
        # 如果 args 里有 wm_hidden_dim (128)，就用它作为 WM 的隐藏层。
        self.hidden_dim = getattr(args, "wm_hidden_dim", args.rnn_hidden_dim)

        # === 修复：兼容不同的参数命名 ===
        if hasattr(args, "wm_latent_dim"):
            self.latent_dim = args.wm_latent_dim
        elif hasattr(args, "latent_dim"):
            self.latent_dim = args.latent_dim
        
        self.att_embed_dim = self.latent_dim

        # ===================================================================
        # 1. Factorized Encoder (Shared Weights across Agents)
        # ===================================================================
        # [修改] 输入维度增加 n_agents (Obs + Action + Agent_ID)
        self.fc1 = nn.Linear(input_shape + self.n_agents, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        
        # VAE 的均值和方差头
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        # ===================================================================
        # 2. Factorized Decoder (Reconstruction)
        # ===================================================================
        # [修改] 解码器输入维度增加 n_agents (Z + Agent_ID)
        # 显式告诉解码器当前重构的是哪个 Agent，降低拟合难度
        self.decoder_fc = nn.Linear(self.latent_dim + self.n_agents, self.hidden_dim)
        self.decoder_out = nn.Linear(self.hidden_dim, output_shape)

        # self.apply(init_weights)
        # ===================================================================
        # 3. Attention Aggregator (Proxy Confounder Generator)
        # ===================================================================
        self.att_query = nn.Linear(self.latent_dim, self.att_embed_dim)
        self.att_key = nn.Linear(self.latent_dim, self.att_embed_dim)
        self.att_val = nn.Linear(self.latent_dim, self.att_embed_dim)
        print("########## Using Agent-ID Conditioned WM #########")

        

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, inputs, actions, hidden_state=None):
        """
        Args:
            inputs: [Batch, Seq, N_Agents, Obs_Dim]
            actions: [Batch, Seq, N_Agents, Act_Dim]
        """
        # 获取动态维度
        b, t, n, _ = inputs.shape
        
        # ===================================================================
        # [New] 生成 One-Hot Agent ID
        # ===================================================================
        # 1. 生成基础 Eye 矩阵: [N, N]
        # 2. 扩展维度以匹配 Batch 和 Time: [1, 1, N, N] -> [B, T, N, N]
        agent_ids = torch.eye(n, device=inputs.device).unsqueeze(0).unsqueeze(0).expand(b, t, -1, -1)
        
        # -------------------------------------------------------------------
        # Step A: 准备输入 (Encoder Input)
        # -------------------------------------------------------------------
        # 拼接: Obs + Action + AgentID
        x = torch.cat([inputs, actions, agent_ids], dim=-1)  
        
        # -------------------------------------------------------------------
        # Step B: 独立编码
        # -------------------------------------------------------------------
        # 展平 Batch 和 Agent 维度
        x_flat = x.reshape(b * n, t, -1) 
        
        # 特征提取 MLP
        x_emb = F.relu(self.fc1(x_flat))
        
        # --- RNN 处理 ---
        if hidden_state is None:
            h_in = x_emb.new_zeros(1, b * n, self.hidden_dim)
        else:
            h_in = hidden_state.reshape(1, b * n, -1)
            
        rnn_out, h_out = self.rnn(x_emb, h_in)
        
        # 计算潜在分布参数
        mu = self.fc_mu(rnn_out)       # [B*N, T, Latent]
        logvar = self.fc_logvar(rnn_out)
        
        # 采样得到局部隐变量 z_local
        z_local = self.reparameterize(mu, logvar)
        
        # -------------------------------------------------------------------
        # Step C: 局部重构 (Decoder Input)
        # -------------------------------------------------------------------
        # [关键修改] 在解码时也拼接 Agent ID
        # z_local: [B*N, T, Latent]
        # agent_ids: [B, T, N, N] -> [B*N, T, N]
        agent_ids_flat = agent_ids.reshape(b * n, t, -1)
        
        # Decoder Input = Z + Agent_ID
        z_decode_input = torch.cat([z_local, agent_ids_flat], dim=-1)
        
        recon_x = F.relu(self.decoder_fc(z_decode_input))
        recon_out = self.decoder_out(recon_x) # [B*N, T, Obs_Dim]
        
        # -------------------------------------------------------------------
        # Step D: 恢复维度
        # -------------------------------------------------------------------
        z_local = z_local.reshape(b, t, n, -1)
        recon_out = recon_out.reshape(b, t, n, -1)
        mu = mu.reshape(b, t, n, -1)
        logvar = logvar.reshape(b, t, n, -1)
        
        # Detach 用于 Mixer (避免 Mixer 梯度回传影响 VAE 隐空间构建)
        z_local_detached = z_local.detach()
        
        # 输出给 Mixer 的 Z (这里你目前使用的是纯 Z_local，未开启 Attention)
        z_for_mixer = z_local_detached
        
        return z_for_mixer, recon_out, mu, logvar, h_out