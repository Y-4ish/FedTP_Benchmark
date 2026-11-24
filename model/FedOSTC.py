import torch
import torch.nn as nn
from torch_geometric.utils import softmax


class Encoder(nn.Module):
    """GRU时间编码器（处理多节点批次输入）"""

    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        # input_dim 改为 1，适应你的数据特征维度
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # 输入x形状: (B, N, T, F)
        B, N, T, F = x.size()
        # 合并批次和节点维度，视为独立的序列
        x = x.reshape(B * N, T, F)  # (B*N, T, F)
        
        _, h = self.gru(x)  # h: (1, B*N, D_h)
        
        h = h.squeeze(0).view(B, N, -1)  # (B, N, D_h)
        return h


class SpatialAttention(nn.Module):
    """批次化图注意力层"""

    def __init__(self, in_features=64, out_features=64, heads=1):
        super().__init__()
        if in_features != out_features or heads != 1:
            raise ValueError("FedOSTC paper attention preserves hidden width and uses one head")
        self.score = nn.Linear(2 * in_features, 1, bias=False)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, batch_edge_index):
        """
        x形状: (B, N, D_in)
        batch_edge_index: (2, B*E)
        """
        B, N, D_in = x.size()
        x_flat = x.reshape(B * N, D_in)
        source, target = batch_edge_index
        non_self = source != target
        source, target = source[non_self], target[non_self]
        nodes = torch.arange(B * N, device=x.device)
        source = torch.cat((source, nodes))
        target = torch.cat((target, nodes))
        score = self.activation(self.score(torch.cat((x_flat[target], x_flat[source]), dim=-1))).squeeze(-1)
        attention = softmax(score, target, num_nodes=B * N)
        aggregated = torch.zeros_like(x_flat)
        aggregated.index_add_(0, target, attention[:, None] * x_flat[source])
        return torch.sigmoid(aggregated).reshape(B, N, D_in)


class Decoder(nn.Module):
    """多节点GRU解码器"""

    def __init__(self, input_dim=64, hidden_dim=128, pred_steps=12):
        super().__init__()
        self.initial_state = nn.Linear(input_dim, hidden_dim)
        # input_size=1 (假设解码器输入是上一帧的预测值，维度为1)
        self.gru_cell = nn.GRUCell(input_size=1, hidden_size=hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.pred_steps = pred_steps

    def forward(self, h):
        # h形状: (B, N, D_h) - 编码器的最后状态作为解码器的初始隐藏状态
        B, N, D_h = h.size()
        h = self.initial_state(h.reshape(B * N, D_h))

        # 初始化解码器的第一个输入 (全0或者Go Symbol)
        decoder_input = torch.zeros(B * N, 1).to(h.device)
        predictions = []

        # 逐步解码
        for _ in range(self.pred_steps):
            h = self.gru_cell(decoder_input, h)  # (B*N, D_h)
            pred = self.fc(h)  # (B*N, 1)
            predictions.append(pred)
            decoder_input = pred.detach()  # Autoregressive: 使用预测值作为下一步输入

        # 拼接所有时间步
        predictions = torch.cat(predictions, dim=1)  # (B*N, S)
        return predictions.view(B, N, -1)  # (B, N, S)


class FedOSTC(nn.Module):
    def __init__(self, enc_dim=64, gat_dim=64, pred_steps=3, dec_dim=128):
        super().__init__()
        self.encoder = Encoder(input_dim=1, hidden_dim=enc_dim)
        self.gat = SpatialAttention(in_features=enc_dim, out_features=gat_dim)
        self.decoder = Decoder(input_dim=gat_dim, hidden_dim=dec_dim, pred_steps=pred_steps)
        self.register_buffer('edge_index', None)

    # 1. 供 Client 调用：提取本地时间特征
    def forward_encoder(self, x):
        if x.dim() == 5: x = x.squeeze(-1)
        if x.dim() == 3: x = x.unsqueeze(0)
        h_time = self.encoder(x) # 输出形状: (B, N_local, D)
        return h_time

    # 2. 供 Server 调用：跨节点空间聚合
    def forward_server_gat(self, h_global, global_edge_index):
        # h_global 形状: (B, N_total, D)
        # global_edge_index 形状: (2, E_total)
        # 这里的 GAT 逻辑需要适配处理全局大图
        B, N_total, D_in = h_global.size()
        
        # 扩展 edge_index 以适应 Batch 维度 (参考你原版代码的逻辑)
        edge_indices = global_edge_index.unsqueeze(1).repeat(1, B, 1) 
        offsets = torch.arange(B, device=h_global.device) * N_total
        offsets = offsets.view(1, B, 1)
        batch_edge_index = (edge_indices + offsets).view(2, -1)
        
        h_spatio_global = self.gat(h_global, batch_edge_index) 
        return h_spatio_global # 输出形状: (B, N_total, D_out)

    # 3. 供 Client 调用：本地解码预测
    def forward_decoder(self, h_spatio_local):
        predictions = self.decoder(h_spatio_local)
        return predictions.unsqueeze(-1) # 恢复形状匹配标签
