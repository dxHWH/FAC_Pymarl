import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class StateAttentionZ(nn.Module):
    def __init__(self, state_dim, latent_dim, attention_dim=None):
        """
        Cross Attention Module: State queries Agents' Z
        
        Args:
            state_dim: 环境全局 State 的维度 (作为 Query)
            latent_dim: Agent Z 的维度 (作为 Key 和 Value)
            attention_dim: 内部计算注意力的维度，默认等于 latent_dim
        """
        super(StateAttentionZ, self).__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        # 如果不指定 embed_dim，默认保持和 Z 的维度一致
        self.embed_dim = attention_dim if attention_dim is not None else latent_dim
        
        # Q, K, V Projections
        # Query: 来自 State
        self.w_q = nn.Linear(state_dim, self.embed_dim)
        # Key & Value: 来自 Agent Z
        self.w_k = nn.Linear(latent_dim, self.embed_dim)
        self.w_v = nn.Linear(latent_dim, self.embed_dim)
        
        self.scale = self.embed_dim ** -0.5

    def forward(self, state, z):
        """
        Args:
            state: [B*T, State_Dim]
            z:     [B*T, N_Agents, Latent_Dim]
            
        Returns:
            z_global: [B*T, Embed_Dim] (聚合后的全局 Z)
        """
        # 1. 计算 Query (from State)
        # State: [B*T, S] -> [B*T, 1, Embed] (增加一个维度代表只有 1 个 Query)
        q = self.w_q(state).unsqueeze(1)
        
        # 2. 计算 Key & Value (from Z)
        # Z: [B*T, N, Z_dim] -> [B*T, N, Embed]
        k = self.w_k(z)
        v = self.w_v(z)
        
        # 3. Scaled Dot-Product Attention
        # Score: [B*T, 1, Embed] @ [B*T, Embed, N] -> [B*T, 1, N]
        # 物理含义: 当前 State 对每个 Agent 的关注程度
        score = th.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(score, dim=-1)
        
        # 4. Weighted Sum
        # [B*T, 1, N] @ [B*T, N, Embed] -> [B*T, 1, Embed]
        z_aggregated = th.matmul(attn_weights, v)
        
        # 5. Remove extra dim -> [B*T, Embed]
        return z_aggregated.squeeze(1)


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



class DVDWMFacMixer(nn.Module):
    def __init__(self, args):
        """
        DVDWMFacMixer: 专为 Factorized World Model 设计的 Mixer。
        核心升级: 
        1. 实现 Agent-wise Z 与 乘法调制的完全解耦。
        2. [New] 支持动态聚合 (Dynamic Aggregation) 以平衡前期稳定性和后期精度。
        
        Args Config:
            use_multiple (bool): 
                True  -> 使用乘法调制 (W = W_base * W_mod)
                False -> 使用拼接机制 (W = Hyper(Cat(State, Z)))
            use_Zmean (bool):
                True  -> 对 Z 进行 Mean Pooling (Global Z)
                False -> 保留 Agent 维度 (Agent-Wise Z)
            use_dynamic_alpha (bool): [New]
                True  -> 开启 Z 的动态混合 (Alpha * Agent_Z + (1-Alpha) * Mean_Z)
                False -> 保持静态配置 (由 use_Zmean 决定)
        """
        super(DVDWMFacMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        
        print("####### Using Decoupled DVDWMFacMixer #######")
        



        # === 参数配置 ===
        # 1. Z 维度
        if hasattr(args, "wm_latent_dim"):
            self.latent_dim = args.wm_latent_dim
        else:
            self.latent_dim = getattr(args, "latent_dim", 64)

        # 2. 核心开关
        self.use_multiple = getattr(args, "use_multiple", False)
        self.use_Zmean = getattr(args, "use_Zmean", False) # 默认为 False
        self.use_z_bias = getattr(args, "use_z_bias", True)
        self.use_wm_res = getattr(args, "use_wm_res", True)
        self.use_Zatten = getattr(args, "use_Zatten", False)
        
        #  动态聚合开关
        self.use_dynamic_alpha = getattr(args, "use_dynamic_alpha", False)
        # cross attention 开关
        self.use_cross_atten_mix = getattr(args, "use_cross_atten_mix", False)

        # 动态 Alpha 的超参数 (如果没有在 args 定义，使用默认值)
        # 建议 alpha 从 0 (全 Mean) 线性增加到 1 (全 Agent-wise)
        # 持续时间建议覆盖 WM 收敛期，例如前 2M 步
        self.alpha_start = 0.0
        self.alpha_end = 1.0
        self.alpha_duration = getattr(args, "alpha_anneal_time", 2000000) # 2M 步
        self.current_alpha = self.alpha_start

        print(f"Config: use_multiple={self.use_multiple}, use_Zmean={self.use_Zmean}")
        print(f"Config: use_dynamic_alpha={self.use_dynamic_alpha}, duration={self.alpha_duration}")
        print(f"Config: use_cross_atten_mix={ self.use_cross_atten_mix}")
        print(f"Config: use_Zatten={ self.use_Zatten}")
        if self.use_cross_atten_mix:
            # 直接实例化我们定义的模块
            self.cross_attn = StateAttentionZ(
                state_dim=self.state_dim, 
                latent_dim=self.latent_dim,
                attention_dim=self.latent_dim # 保持维度一致以便后续拼接
            )
        

        # =======================================================================
        # Hypernet 0: z_local之前的自注意力
        # =======================================================================
        self.att_embed_dim = self.latent_dim
        self.att_query = nn.Linear(self.latent_dim, self.att_embed_dim)
        self.att_key = nn.Linear(self.latent_dim, self.att_embed_dim)
        self.att_val = nn.Linear(self.latent_dim, self.att_embed_dim)

        # =======================================================================
        # Hypernet 1: 生成第一层权重 W1
        # =======================================================================
        if self.use_multiple:
            # === 模式 A: 乘法调制 ===
            # self.hyper_w_1 = nn.Sequential(
            #     nn.Linear(self.state_dim, args.hypernet_hidden_dim),
            #     nn.ReLU(),
            #     nn.Linear(args.hypernet_hidden_dim, self.embed_dim * self.n_agents)
            # )
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, args.hypernet_hidden_dim),
                nn.LayerNorm(args.hypernet_hidden_dim),  # <--- [新增] 稳定隐层分布
                nn.ReLU(),
                nn.Linear(args.hypernet_hidden_dim, self.embed_dim * self.n_agents)
            )
            
            # Modulator
            # 动态模式下，我们需要能够处理 Agent-wise 输入的能力，所以结构按 Agent-wise 初始化
            # 如果 use_dynamic_alpha=True，我们强制构建 Agent-wise 的 Modulator
            if self.use_Zmean and not self.use_dynamic_alpha:
                self.z_modulator = nn.Sequential(
                    nn.Linear(self.latent_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, self.embed_dim)
                )
            else:
                # 只要是 Agent-wise 或者 Dynamic，都用这个 Parameter Shared MLP
                self.z_modulator = nn.Sequential(
                    nn.Linear(self.latent_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, self.embed_dim)
                )
                
        else:
            # === 模式 B: 拼接机制 ===
            if self.use_Zmean and not self.use_dynamic_alpha:
                # 纯 Global Z 模式
                self.hyper_input_dim = self.state_dim + self.latent_dim
                self.hyper_w_1 = nn.Sequential(
                    nn.Linear(self.hyper_input_dim, args.hypernet_hidden_dim),
                    # nn.LayerNorm(args.hypernet_hidden_dim),  # <--- [新增] 稳定隐层分布
                    nn.ReLU(),
                    nn.Linear(args.hypernet_hidden_dim, self.embed_dim * self.n_agents)
                )
            else:
                # Agent-Wise Z 模式 或 动态混合模式
                # 动态混合本质上也是输入 Agent-wise 的数据结构，只是值被混合了
                self.hyper_input_dim = self.state_dim + self.latent_dim
                self.hyper_w_1 = nn.Sequential(
                    nn.Linear(self.hyper_input_dim, args.hypernet_hidden_dim),
                    # nn.LayerNorm(args.hypernet_hidden_dim),  # <--- [新增] 稳定隐层分布
                    nn.ReLU(),
                    nn.Linear(args.hypernet_hidden_dim, self.embed_dim)
                )

        # =======================================================================
        # Hypernet Bias 1
        # =======================================================================
        self.bias_input_dim = self.latent_dim + self.state_dim if self.use_z_bias else self.state_dim
        self.hyper_b_1 = nn.Linear(self.bias_input_dim, self.embed_dim)

        # =======================================================================
        # Hypernet 2
        # =======================================================================
        if self.use_multiple:
            self.hyper_w_2 = nn.Sequential(
                nn.Linear(self.state_dim, args.hypernet_hidden_dim),
                nn.LayerNorm(args.hypernet_hidden_dim),  # <--- [新增] 稳定隐层分布
                nn.ReLU(),
                nn.Linear(args.hypernet_hidden_dim, self.embed_dim)
            )
        else:
            w2_input_dim = self.state_dim if getattr(self.args, "use_single", False) else self.state_dim + self.latent_dim
            self.hyper_w_2 = nn.Sequential(
                nn.Linear(w2_input_dim, args.hypernet_hidden_dim),
                # nn.LayerNorm(args.hypernet_hidden_dim),  # <--- [新增] 稳定隐层分布
                nn.ReLU(),
                nn.Linear(args.hypernet_hidden_dim, self.embed_dim)
            )

        # =======================================================================
        # Hypernet Bias 2
        # =======================================================================
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(self.bias_input_dim, self.embed_dim),
            # nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )
        # 在 __init__ 最后调用
        # self.apply(init_weights)

    def update_alpha(self, t_total):
        """
        更新混合系数 alpha
        t_total: 当前总训练步数
        """
        if not self.use_dynamic_alpha:
            return

        # 线性增长: 0 -> 1
        progress = min(1.0, max(0.0, t_total / self.alpha_duration))
        self.current_alpha = self.alpha_start + progress * (self.alpha_end - self.alpha_start)
        
        # 只有在特定的 log 间隔才打印，避免刷屏 (假设调用方控制，或者这里简单处理)
        # if t_total % 10000 == 0:
        #     print(f"Mixer Dynamic Alpha Updated: {self.current_alpha:.4f}")

    def forward(self, agent_qs, states, z, t_total=0):
        """
        Args:
            t_total: 必须传入当前的 timestep 用于更新 alpha
        """
        # 尝试更新 alpha (需要 Runner 传入 t_total)
        # 如果 runner 没有传 t_total，默认 alpha 保持初始值 (或由外部控制器更新)
        if self.use_dynamic_alpha and t_total > 0:
            self.update_alpha(t_total)
            
        bs, ts, _ = agent_qs.shape
        
        states = states.reshape(-1, self.state_dim)           # [B*T, S_dim]
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)     # [B*T, 1, N]
        z = z.reshape(-1, self.n_agents, self.latent_dim)     # [B*T, N, Z_dim]
        z_for_mixer = z
        
        if(self.use_Zatten):
            # 计算 Agent 间的动力学相关性，生成全局 Proxy Confounder
            q = self.att_query(z) # [B, T, N, Emb]
            k = self.att_key(z)   # [B, T, N, Emb]
            v = self.att_val(z)   # [B, T, N, Emb]
            
            # Scaled Dot-Product Attention
            # attention_score(i, j) 表示 Agent i 和 Agent j 在动力学上的关联程度
            scaling = self.att_embed_dim ** 0.5
            scores = th.matmul(q, k.transpose(-2, -1)) / scaling # [B, T, N, N]
            attn_weights = F.softmax(scores, dim=-1)
            
            # 加权聚合: 此时 z_weighted 中的每个 Agent 特征都融合了与之相关的其他 Agent 信息
            z_attended = th.matmul(attn_weights, v) # [B, T, N, Emb]

             # 我们直接把每个 Agent 的 Z 给 Mixer，让 Mixer 自己决定怎么用
            z_for_mixer = z_attended  # 保持 [B, T, N, 64]

        # === [核心逻辑分支: 残差连接控制] ===
        if self.use_wm_res:
            # 方案：残差连接 (Local + Attention)
            # - z_local_detached: 提供稳定的原始观测信息 (保底，防止前期崩盘/后期信息瓶颈)
            # - z_attended: 提供交互上下文和显著性信息 (加速前期收敛)
            
            z_for_mixer = z + z_attended
        else:
            if self.use_Zatten:
                # 兼容旧逻辑：只使用 Attention 后的特征
                # 对应之前的黄色线(配合Mean) 或 红色线(配合Agent-wise直接输出)
                z_for_mixer = z_attended
            else:
                z_for_mixer = z



        z = z_for_mixer
        # ===================================================================
        # [Refactored] Global Z 生成策略
        # ===================================================================
        if self.use_cross_atten_mix:
            # 调用单独封装的模块
            z_global = self.cross_attn(states, z) # [B*T, Latent]
        else:
            # Standard Mean Pooling
            z_global = z.mean(dim=1) # [B*T, Latent]

        # ===================================================================
        # Layer 1 权重计算 (W1)
        # ===================================================================
        if self.use_multiple:
            # === 模式 A: 乘法调制 ===
            w1_base = th.abs(self.hyper_w_1(states)).view(-1, self.n_agents, self.embed_dim)
            
            if self.use_dynamic_alpha:
                # [Dynamic] 混合 Agent-wise 和 Global
                # 1. 计算 Agent-wise Mod
                mod_local = self.z_modulator(z) # [B*T, N, Embed]
                # 2. 计算 Global Mod (广播)
                mod_global = self.z_modulator(z_global).unsqueeze(1).expand(-1, self.n_agents, -1)
                # 3. 混合
                z_mod = self.current_alpha * mod_local + (1 - self.current_alpha) * mod_global
            
            elif self.use_Zmean:
                # [Static Global]
                z_mod = self.z_modulator(z_global).unsqueeze(1)
            else:
                # [Static Local]
                z_mod = self.z_modulator(z)
                
            w1 = th.abs(w1_base * z_mod)
            
        else:
            # === 模式 B: 拼接机制 ===
            
            if self.use_dynamic_alpha:
                # [Dynamic] 动态混合输入
                # 1. 扩展 State
                states_expanded = states.unsqueeze(1).expand(-1, self.n_agents, -1) # [B*T, N, S]
                
                # 2. 准备 Global Z 并扩展 (用于混合)
                z_global_expanded = z_global.unsqueeze(1).expand(-1, self.n_agents, -1) # [B*T, N, Z]
                
                # 3. 计算混合后的 Z
                # Z_mix = alpha * Z_local + (1 - alpha) * Z_global
                z_mixed = self.current_alpha * z + (1 - self.current_alpha) * z_global_expanded
                
                # 4. 拼接
                hyper_input = th.cat([states_expanded, z_mixed], dim=-1) # [B*T, N, S+Z]
                
                # 5. 生成权重
                w1 = th.abs(self.hyper_w_1(hyper_input))
                
            elif self.use_Zmean:
                # [Static Global]
                inputs = th.cat([states, z_global], dim=1)
                w1 = th.abs(self.hyper_w_1(inputs)).view(-1, self.n_agents, self.embed_dim)
            else:
                # [Static Local]
                states_expanded = states.unsqueeze(1).expand(-1, self.n_agents, -1)
                #print(states)
                #print(z)
                hyper_input = th.cat([states_expanded, z], dim=-1)
                w1 = th.abs(self.hyper_w_1(hyper_input))

        # ===================================================================
        # Layer 1 计算
        # ===================================================================
        bias_in = th.cat([states, z_global], dim=1) if self.use_z_bias else states
        b1 = self.hyper_b_1(bias_in).view(-1, 1, self.embed_dim)
        
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        # ===================================================================
        # Layer 2 权重计算 (W2)
        # ===================================================================
        if self.use_multiple:
            w2 = th.abs(self.hyper_w_2(states)).view(-1, self.embed_dim, 1)
        else:
            w2_in = states if getattr(self.args, "use_single", False) else th.cat([states, z_global], dim=1)
            w2 = th.abs(self.hyper_w_2(w2_in)).view(-1, self.embed_dim, 1)

        # ===================================================================
        # Layer 2 计算
        # ===================================================================
        b2 = self.hyper_b_2(bias_in).view(-1, 1, 1)
        
        y = th.bmm(hidden, w2) + b2
        
        return y.view(bs, ts, 1)