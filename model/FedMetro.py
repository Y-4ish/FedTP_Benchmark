import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LocalHistoryHyperNet(nn.Module):
    """Client-local historical context, following FedMetro's DLinear/attention path."""

    def __init__(self, num_nodes, input_dim, hidden_dim, source_steps=21):
        super().__init__()
        self.num_nodes = num_nodes
        self.source_steps = source_steps
        self.seasonal = nn.Parameter(torch.empty(num_nodes, hidden_dim, source_steps))
        self.trend = nn.Parameter(torch.empty(num_nodes, hidden_dim, source_steps))
        nn.init.constant_(self.seasonal, 1.0 / source_steps)
        nn.init.constant_(self.trend, 1.0 / source_steps)
        self.source_projection = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.Transformer(
            d_model=hidden_dim, nhead=4, num_encoder_layers=2,
            num_decoder_layers=2, dim_feedforward=4 * hidden_dim,
            dropout=0.1, activation='gelu', batch_first=True,
            norm_first=True,
        )

    def forward(self, history, x):
        # history: [S, 252, N, C]; x: [B, T, N, C]
        windows, length, nodes, features = history.shape
        if nodes != self.num_nodes or length != self.source_steps * 12:
            raise ValueError('FedMetro historical context shape mismatch')
        h = history.mean(-1).permute(0, 2, 1).reshape(windows * nodes, 1, length)
        h = F.avg_pool1d(h, kernel_size=12, stride=12).reshape(windows, nodes, self.source_steps)
        smooth = F.avg_pool1d(F.pad(h.reshape(windows * nodes, 1, -1), (12, 12), mode='replicate'),
                              kernel_size=25, stride=1).reshape_as(h)
        source = torch.einsum('snd,nhd->snh', h - smooth, self.seasonal)
        source = source + torch.einsum('snd,nhd->snh', smooth, self.trend)
        target = self.source_projection(x)
        batch, steps, _, dim = target.shape
        source = source.permute(1, 0, 2).unsqueeze(0).expand(batch, -1, -1, -1)
        source = source.reshape(batch * nodes, windows, dim)
        target = target.permute(0, 2, 1, 3).reshape(batch * nodes, steps, dim)
        def positional(length, device, dtype):
            positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
            frequencies = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype)
                                    * (-math.log(10000.0) / dim))
            encoding = torch.zeros(length, dim, device=device, dtype=dtype)
            encoding[:, 0::2] = torch.sin(positions * frequencies)
            encoding[:, 1::2] = torch.cos(positions * frequencies)
            return encoding
        source = source + positional(windows, source.device, source.dtype)
        target = target + positional(steps, target.device, target.dtype)
        attended = self.attention(source, target)
        return attended.reshape(batch, nodes, steps, dim).permute(0, 2, 1, 3)

# ==========================================
# 1. 辅助模块：Hard Concrete 分布 (用于动态掩码)
# ==========================================
class HardConcreteMask(nn.Module):
    """对应论文 Eq. (9): Dynamic Mask Generation"""
    def __init__(self, beta=0.66, gamma=-0.1, zeta=1.1):
        super(HardConcreteMask, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta

    def forward(self, log_alpha):
        # log_alpha: [Batch, N, F], as in the author's concrete gate.
        if self.training:
            # z ~ U(0, 1)
            u = torch.rand_like(log_alpha)
            # 加上极小值防止 log(0)
            u = torch.clamp(u, 1e-6, 1.0 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + log_alpha) / self.beta)
        else:
            s = torch.sigmoid(log_alpha / self.beta)
        
        s_tilde = s * (self.zeta - self.gamma) + self.gamma
        m = torch.clamp(s_tilde, 0.0, 1.0)
        return m

# ==========================================
# 2. 动态嵌入与掩码生成模块
# ==========================================
class DynamicEmbeddingAndMask(nn.Module):
    """对应论文 Algorithm 1 与 Eq. (4)-(9)"""
    def __init__(self, in_dim, hist_dim, d_D, num_nodes, d_E, mask_dim):
        super(DynamicEmbeddingAndMask, self).__init__()
        self.d_D = d_D
        self.d_E = d_E
        
        # Cross Attention 参数
        self.W_Q = nn.Linear(in_dim, d_D)
        self.W_K = nn.Linear(hist_dim, d_D)
        self.W_V = nn.Linear(hist_dim, d_D)
        
        # 动态嵌入生成参数
        self.W_E = nn.Linear(d_D, d_E)
        
        # 动态掩码生成参数
        self.W_O = nn.Linear(d_D, mask_dim)
        self.mask_generator = HardConcreteMask()
        
        # 个性化静态节点嵌入 E_i
        self.static_E = nn.Parameter(torch.randn(num_nodes, d_E))
        nn.init.xavier_normal_(self.static_E)

    def forward(self, X_t, X_p):
        # X_t: 当前时刻观测 [Batch, N, in_dim]
        # X_p: 历史 Patch 数据 [Batch, N, P, hist_dim]
        B, N, _ = X_t.shape
        
        # --- 1. Historical Pattern Mining (Cross Attention) ---
        Q = self.W_Q(X_t).unsqueeze(2)  # [B, N, 1, d_D]
        K = self.W_K(X_p)               # [B, N, P, d_D]
        V = self.W_V(X_p)               # [B, N, P, d_D]
        
        # Attention score: Softmax(QK^T / sqrt(d_k))
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_D) # [B, N, 1, P]
        attn = F.softmax(scores, dim=-1)
        
        # D_i^t: [B, N, d_D]
        D_t = torch.matmul(attn, V).squeeze(2) 
        
        # --- 2. Dynamic Embedding Generation ---
        E_hat_t = self.W_E(D_t) # 时间增量 [B, N, d_E]
        # E_i^t = E_i + \hat{E}_i^t
        # 注意广播机制: [N, d_E] + [B, N, d_E] -> [B, N, d_E]
        E_t = self.static_E.unsqueeze(0) + E_hat_t 
        
        # --- 3. Dynamic Mask Generation ---
        log_alpha = self.W_O(D_t)
        M_t = self.mask_generator(log_alpha) # [B, N, mask_dim]
        
        return E_t, M_t

# ==========================================
# 3. 多项式近似 ReLU (用于通信压缩)
# ==========================================
def poly_approx_relu(x, K=2):
    """对应论文 Eq. (14) 及 F_K(x) 分解"""
    # 简单使用泰勒展开或预设的多项式系数 (p_0, p_1, ..., p_K)
    # 此处假设 p_k 均为可学习参数或预定义，简单起见用 x, x^2, ..., x^K 拼接表示 F_K
    # 实际应用中，可以通过多项式展开来近似 ReLU
    poly_terms = [torch.ones_like(x)] + [x ** k for k in range(1, K + 1)]
    return torch.cat(poly_terms, dim=-1) # [..., d_E * K]

# ==========================================
# 4. FedMetro 专属 GRU 单元 (极致提速 & 数学等效版)
# ==========================================
class FedMetroGRUCell(nn.Module):
    """对应论文 Eq. (17), (18): Spatial-Temporal Fusion"""
    def __init__(self, input_dim, hidden_dim, d_E, K):
        super(FedMetroGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.d_E = d_E
        self.K = K
        
        # Eq(17): 多项式近似展开的可学习系数 p_k
        self.p_k = nn.Parameter(torch.ones(K + 1))
        
        # 【加速优化】：将 Z 门和 R 门的权重与偏置合并为一，减少算子调用
        # 输出维度为 hidden_dim * 2 (前一半给 Z，后一半给 R)
        self.W_zr_dyn = nn.Parameter(torch.Tensor(d_E, input_dim + hidden_dim, hidden_dim * 2))
        self.b_zr_dyn = nn.Parameter(torch.Tensor(d_E, hidden_dim * 2))
        
        # C 门依然独立 (因为它依赖于 R 门的结果)
        self.W_h_dyn = nn.Parameter(torch.Tensor(d_E, input_dim + hidden_dim, hidden_dim))
        self.b_h_dyn = nn.Parameter(torch.Tensor(d_E, hidden_dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        for param in [self.W_zr_dyn, self.W_h_dyn]:
            nn.init.xavier_normal_(param)
        for param in [self.b_zr_dyn, self.b_h_dyn]:
            nn.init.zeros_(param)

    def forward(self, x, hidden_state, E_t, F_E_t, AGG_t):
        B, N, _ = x.shape
        
        # --- 1. 计算全局空间相关性 Z_i^t(H) ---
        F_E_t_reshaped = F_E_t.view(B, N, self.K + 1, self.d_E)
        AGG_t_reshaped = AGG_t.view(B, self.K + 1, self.d_E, self.hidden_dim)
        
        # 【加速优化】：用一个爱因斯坦求和消灭 Python for 循环
        # k: poly_k, b: batch, n: node, d: d_E, h: hidden
        local_spatial_corr = torch.einsum('k, bnkd, bkdh -> bnh', self.p_k, F_E_t_reshaped, AGG_t_reshaped)
            
        Z_H = hidden_state + local_spatial_corr
        concatenation = torch.cat((x, Z_H), dim=-1) # [B, N, in + hid]
        
        # --- 2. 动态自适应权重生成与 GRU 门控 ---
        
        # 【加速优化】：合并计算 Z 门和 R 门 (减少一次大型 einsum 和一次加法)
        zr_weight_out = torch.einsum('bnd, dio, bni -> bno', E_t, self.W_zr_dyn, concatenation)
        zr_bias_out   = torch.einsum('bnd, do -> bno', E_t, self.b_zr_dyn)
        zr_out = zr_weight_out + zr_bias_out
        
        # 将 [B, N, hidden_dim * 2] 切分为两个 [B, N, hidden_dim]
        z_out, r_out = torch.chunk(zr_out, chunks=2, dim=-1)
        z = torch.sigmoid(z_out)
        r = torch.sigmoid(r_out)
        
        # C 门正常计算
        candidate_concat = torch.cat((x, r * Z_H), dim=-1)
        c_weight_out = torch.einsum('bnd, dio, bni -> bno', E_t, self.W_h_dyn, candidate_concat)
        c_bias_out   = torch.einsum('bnd, do -> bno', E_t, self.b_h_dyn)
        c = torch.tanh(c_weight_out + c_bias_out)
        
        new_hidden_state = z * hidden_state + (1 - z) * c
        return new_hidden_state

# ==========================================
# 5. 客户端主模型 (支持序列维度打包与形状自适应)
# ==========================================
class FedMetro_Client_Model(nn.Module):
    def __init__(self, num_nodes, t_in, t_out, input_dim, hidden_dim, d_E=6, K=2, d_D=32):
        super(FedMetro_Client_Model, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.t_out = t_out
        self.K = K
        self.history_hypernet = LocalHistoryHyperNet(num_nodes, input_dim, hidden_dim)
        self.register_buffer('hyper_source', None, persistent=False)
        
        self.hist_proj = nn.Linear(t_in * input_dim, hidden_dim)
        # 新增: 空间投影层，解耦 Phase1 对上一时刻 GRU 隐状态的依赖，实现时间并行
        self.spatial_proj = nn.Linear(input_dim, hidden_dim)
        
        self.dyn_emb_mask = DynamicEmbeddingAndMask(
            in_dim=input_dim, 
            hist_dim=hidden_dim, 
            d_D=d_D, 
            num_nodes=num_nodes, 
            d_E=d_E, 
            mask_dim=hidden_dim
        )
        # [修改] 传入 d_E 和 K，而不是粗暴的 d_E * K
        self.gru_cell = FedMetroGRUCell(input_dim, hidden_dim, d_E, K)
        self.out_layer = nn.Linear(hidden_dim, t_out)

    def set_hyper_source(self, train_windows, history_length=252, stride=96, max_windows=16):
        """Build context from training windows only, never validation/test windows."""
        if train_windows.ndim != 4:
            raise ValueError('FedMetro training windows must have four dimensions')
        if train_windows.shape[1] != self.num_nodes:
            raise ValueError('FedMetro training node count mismatch')
        series = torch.cat((train_windows[0].permute(1, 0, 2), train_windows[1:, :, -1, :]), dim=0)
        if series.shape[0] < history_length:
            series = torch.cat((series[:1].expand(history_length - series.shape[0], -1, -1), series), dim=0)
        starts = list(range(0, series.shape[0] - history_length + 1, stride))
        last = series.shape[0] - history_length
        if not starts or starts[-1] != last:
            starts.append(last)
        starts = starts[-max_windows:]
        self.hyper_source = torch.stack([series[start:start + history_length] for start in starts]).to(
            next(self.parameters()).device
        )

    def _normalize_input_shape(self, x):
        """自动适配数据形状，转为 [Batch, Seq_len, Nodes, Features]"""
        if x.dim() == 4 and x.shape[1] == self.num_nodes and x.shape[2] != self.num_nodes:
            x = x.permute(0, 2, 1, 3)
        elif x.dim() == 3: # 处理 [B, T, N] 或 [B, N, T]
            if x.shape[1] == self.num_nodes:
                x = x.permute(0, 2, 1).unsqueeze(-1)
            else:
                x = x.unsqueeze(-1)
        return x

    def forward_phase1(self, x):
        """
        Phase 1: 接收整个时间序列，计算每一时间步的 AGG_i
        返回: 包含所有时间步的打包张量
        """
        x = self._normalize_input_shape(x)
        B, T, N, C = x.shape
        
        # 提取历史 Patch
        X_hist = x.permute(0, 2, 1, 3).reshape(B, N, -1)
        if self.hyper_source is None:
            raise RuntimeError('FedMetro historical context must be set from training data')
        history_features = self.history_hypernet(self.hyper_source.to(x.device), x)
        X_p = history_features.permute(0, 2, 1, 3)  # [B, N, T, hidden_dim]
        
        # 将原始特征投影为构建空间相关性矩阵的基础
        H_spatial_input = history_features
        
        AGG_i_seq = []
        F_E_seq = []
        M_seq = [] # [新增] 用于收集所有时间步的掩码
        
        # 内部循环处理每一个时间步，独立计算 Embedding 和 Mask
        for t in range(T):
            X_t = x[:, t, :, :] # [B, N, C]
            E_t, M_t = self.dyn_emb_mask(X_t, X_p) 
            F_E_t = poly_approx_relu(E_t, self.K) 
            
            H_masked = H_spatial_input[:, t, :, :] * M_t 
            AGG_i_t = torch.matmul(F_E_t.transpose(1, 2), H_masked) 
            
            AGG_i_seq.append(AGG_i_t)
            F_E_seq.append(F_E_t)
            M_seq.append(M_t) 

           
            if not hasattr(self, 'E_seq_list'): self.E_seq_list = []
            self.E_seq_list.append(E_t)
            
        self.dyn_emb_mask.mask_generator.last_m = torch.stack(M_seq).sum() / 1e8
        E_seq_stacked = torch.stack(self.E_seq_list, dim=1)
        self.E_seq_list = [] 
        
        return torch.stack(AGG_i_seq, dim=1), torch.stack(F_E_seq, dim=1), E_seq_stacked

    # [修改] 新增 E_seq 参数
    def forward_phase2(self, x, F_E_seq, E_seq, AGG_global_seq):
        x = self._normalize_input_shape(x)
        B, T, N, C = x.shape
        H_prev = torch.zeros(B, N, self.hidden_dim, device=x.device)
        
        for t in range(T):
            X_t = x[:, t, :, :]
            F_E_t = F_E_seq[:, t, :, :]
            E_t = E_seq[:, t, :, :]  # [新增]
            AGG_global_t = AGG_global_seq[:, t, :, :]
            
            # [修改] 传入 E_t 供动态权重生成使用
            H_prev = self.gru_cell(X_t, H_prev, E_t, F_E_t, AGG_global_t)
            
        return self.out_layer(H_prev)


class _FederatedSum(torch.autograd.Function):
    """Exchange a graph-convolution sufficient statistic and its adjoint."""

    @staticmethod
    def forward(ctx, local_value, exchange, tag):
        ctx.exchange = exchange
        ctx.tag = tag
        return exchange('forward', tag, local_value.detach()).to(local_value)

    @staticmethod
    def backward(ctx, output_gradient):
        local_gradient = ctx.exchange('backward', ctx.tag, output_gradient.detach())
        return local_gradient.to(output_gradient), None, None


class _PaperGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, order):
        super().__init__()
        self.order = order
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, input_dim, output_dim))
        self.bias_pool = nn.Parameter(torch.zeros(embed_dim, output_dim))
        nn.init.xavier_uniform_(self.weights_pool)

    def forward(self, features, embeddings, mask, coefficients, exchange, tag):
        masked = features * mask
        basis = [torch.ones(embeddings.shape[0], 1, device=embeddings.device,
                            dtype=embeddings.dtype)]
        for degree in range(1, self.order + 1):
            basis.append(torch.einsum('ni,nj->nij', basis[-1], embeddings).flatten(1))
        local_stats = [torch.einsum('nd,bnc->bdc', item, masked) for item in basis]
        packed = torch.cat(local_stats, dim=1)
        global_stats = _FederatedSum.apply(packed, exchange, tag)
        sizes = [item.shape[1] for item in basis]
        global_parts = global_stats.split(sizes, dim=1)
        spatial = sum(
            coefficients[degree] * torch.einsum('nd,bdc->bnc', basis[degree], global_parts[degree])
            for degree in range(self.order + 1)
        )
        projected = masked + spatial
        weights = torch.einsum('nd,dio->nio', embeddings, self.weights_pool)
        bias = torch.einsum('nd,do->no', embeddings, self.bias_pool)
        return torch.einsum('bni,nio->bno', projected, weights) + bias


class FedMetroPaperAdapted(nn.Module):
    """FedMetro's one-layer gated graph recurrence on a local PeMS partition."""

    def __init__(self, num_nodes, t_in, t_out, input_dim=1, hidden_dim=64,
                 embed_dim=6, order=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.t_in = t_in
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.history_hypernet = LocalHistoryHyperNet(num_nodes, input_dim, hidden_dim)
        self.register_buffer('hyper_source', None, persistent=False)
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        # The author's FED path projects local HyperNet features and appends
        # them to the observed input before the recurrent graph cell.
        self.main_weights_pool = nn.Parameter(torch.empty(embed_dim, hidden_dim, input_dim))
        nn.init.xavier_uniform_(self.main_weights_pool)
        self.mask_logits = nn.Linear(hidden_dim, 2 * input_dim + hidden_dim)
        self.mask_generator = HardConcreteMask(beta=2.0 / 3.0)
        self.gate = _PaperGraphConv(2 * input_dim + hidden_dim, 2 * hidden_dim, embed_dim, order)
        self.update = _PaperGraphConv(2 * input_dim + hidden_dim, hidden_dim, embed_dim, order)
        self.poly_coefficients = nn.Parameter(torch.randn(order + 1))
        self.prediction_head = nn.Linear(hidden_dim, t_out)

    set_hyper_source = FedMetro_Client_Model.set_hyper_source

    def forward(self, x, exchange):
        if self.hyper_source is None:
            raise RuntimeError('FedMetro historical context must come from training windows')
        if x.shape[1] == self.num_nodes:
            x = x.permute(0, 2, 1, 3)
        if x.shape[1] != self.t_in or x.shape[2] != self.num_nodes:
            raise ValueError(f'Unexpected FedMetro input shape: {tuple(x.shape)}')
        history = self.history_hypernet(self.hyper_source.to(x.device), x)
        state = x.new_zeros(x.shape[0], self.num_nodes, self.hidden_dim)
        mask_sum = x.new_zeros(())
        history_weights = torch.einsum('nd,dhi->nhi', self.node_embeddings,
                                       self.main_weights_pool)
        for step in range(self.t_in):
            mask = self.mask_generator(self.mask_logits(history[:, step]))
            mask_sum = mask_sum + mask.sum()
            history_input = torch.einsum('bnh,nhi->bni', history[:, step], history_weights)
            recurrent_input = torch.cat((history_input, x[:, step]), dim=-1)
            gates = self.gate(torch.cat((recurrent_input, state), dim=-1),
                              self.node_embeddings, mask, self.poly_coefficients,
                              exchange, f't{step}_gate')
            z, r = torch.sigmoid(gates).chunk(2, dim=-1)
            candidate = self.update(torch.cat((recurrent_input, z * state), dim=-1),
                                    self.node_embeddings, mask, self.poly_coefficients,
                                    exchange, f't{step}_update')
            proposal = torch.tanh(candidate)
            state = r * state + (1.0 - r) * proposal
        return self.prediction_head(state), mask_sum / 1e8
