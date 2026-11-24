import torch
from torch import nn
from torch.nn import MultiheadAttention
from torch_geometric.nn import GATConv, GCNConv


class TrafficPatternModule(nn.Module):
    """
    Shared TP module.
    返回:
        h_gru   : 给 ST 分支用（更接近官方实现）
        tp_pred : TP 分支自己的预测输出
    """

    def __init__(self, hidden_dim, message_dim, his_num, pred_num):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.his_num = his_num

        self.tp_gru = nn.GRU(message_dim, 1, batch_first=True)
        self.gru_out = nn.Linear(his_num, hidden_dim)
        self.tp_attention = MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        self.tp_head1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.tp_head2 = nn.Linear(hidden_dim, pred_num)

    def forward(self, x):
        # FATE 常见格式: [B, T_in, N, C] -> [B, N, T_in, C]
        if x.shape[1] == self.his_num and x.shape[2] != self.his_num:
            x = x.transpose(1, 2)

        batch_size, num_nodes, his_len, message_dim = x.shape
        x_bn = x.reshape(batch_size * num_nodes, his_len, message_dim)

        gru_out, _ = self.tp_gru(x_bn)
        h_gru = self.gru_out(gru_out.squeeze(-1).reshape(batch_size, num_nodes, his_len))
        # Attend over nodes within each example. The released implementation
        # omits batch_first and unintentionally mixes different batch examples.
        h_att, _ = self.tp_attention(h_gru, h_gru, h_gru)
        tp_pred = self.tp_head2(self.tp_head1(torch.cat([h_gru, h_att], dim=-1)))
        return h_gru, tp_pred


class SharedModel(nn.Module):
    def __init__(self, hidden_dim, message_dim, his_num, pred_num):
        super().__init__()
        self.tp_module = TrafficPatternModule(hidden_dim, message_dim, his_num, pred_num)

    def forward(self, x):
        return self.tp_module(x)


class SpatialStructureModule(nn.Module):
    """
    更接近官方 SpatialModel:
    - SS: str_init + random walk -> GCN -> h_ss
    - ST: GAT 直接吃 temporal GRU output (h_gru) -> h_gat
    """

    def __init__(self, num_nodes, hidden_dim, pred_num, edge_index, gcn_layers=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.pred_num = pred_num
        self.register_buffer("str_init", self._random_walk_features(edge_index, num_nodes, hidden_dim))
        self.embedding_s = nn.Linear(hidden_dim, 2 * hidden_dim)

        self.graph_convs = nn.ModuleList(
            [GCNConv(2 * hidden_dim, 2 * hidden_dim) for _ in range(gcn_layers)]
        )
        self.out_head = nn.Linear(2 * hidden_dim, pred_num)

        # 官方里是 GAT 吃 temporal embedding
        self.sp_learner = GATConv(hidden_dim, pred_num, heads=3, concat=False, dropout=0.1)

    @staticmethod
    def _random_walk_features(edge_index, num_nodes, n_rw):
        # Author Structure_init computes return probabilities from A * D^{-1}.
        edge_index = edge_index.cpu()
        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
        adj[edge_index[0], edge_index[1]] = 1.0
        degree = torch.bincount(edge_index[0], minlength=num_nodes).float().clamp_min(1)
        transition = adj / degree.unsqueeze(0)
        walk_power = transition
        features = []
        for _ in range(n_rw):
            features.append(torch.diagonal(walk_power))
            walk_power = walk_power @ transition
        return torch.stack(features, dim=-1)

    def _batched_edge_index(self, edge_index, batch_size, num_nodes, device):
        edge_index = edge_index.to(device)
        edge_num = edge_index.shape[1]

        offsets = torch.arange(batch_size, device=device).repeat_interleave(edge_num) * num_nodes
        src = edge_index[0].repeat(batch_size) + offsets
        dst = edge_index[1].repeat(batch_size) + offsets
        return torch.stack([src, dst], dim=0)

    def forward(self, edge_index, h_gru):
        batch_size, num_nodes, _ = h_gru.shape

        # ---- SS branch ----
        node_embed = self.embedding_s(self.str_init)
        for conv in self.graph_convs:
            node_embed = torch.tanh(conv(node_embed, edge_index))

        h_ss = self.out_head(node_embed)                           # [N, pred_num]
        h_ss = h_ss.unsqueeze(0).expand(batch_size, -1, -1)       # [B, N, pred_num]

        # ---- ST branch ----
        h_gru_flat = h_gru.reshape(batch_size * num_nodes, self.hidden_dim)
        edge_index_batch = self._batched_edge_index(edge_index, batch_size, num_nodes, h_gru.device)

        h_gat_flat = self.sp_learner(h_gru_flat, edge_index_batch)
        h_gat = h_gat_flat.reshape(batch_size, num_nodes, self.pred_num)

        return h_ss, h_gat


class GenerateLinear(nn.Module):
    """
    更接近官方仓库里的 Generate_Linear:
    output = W(meta) * meta + b(meta)
    """

    def __init__(self, pred_num):
        super().__init__()
        self.w_linear = nn.Linear(pred_num, pred_num)
        self.b_linear = nn.Linear(pred_num, 1)

    def forward(self, meta_knowledge):
        w_meta = self.w_linear(meta_knowledge)
        b_meta = self.b_linear(meta_knowledge)
        return w_meta * meta_knowledge + b_meta


class STNET_pFedCTP(nn.Module):
    def __init__(self, num_nodes, hidden_dim, his_num, pred_num, message_dim=1, gcn_layers=1, edge_index=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_num = pred_num
        self.edge_index = edge_index

        # trainer 里联邦共享就是 shareModel
        self.shareModel = SharedModel(hidden_dim, message_dim, his_num, pred_num)

        self.spatialModel = SpatialStructureModule(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            pred_num=pred_num,
            edge_index=edge_index,
            gcn_layers=gcn_layers,
        )

        # 命名也尽量贴近官方
        self.stPare = GenerateLinear(pred_num=pred_num)
        self.stPredictor = nn.Linear(pred_num * 3, pred_num)

    def forward(self, x):
        if self.edge_index is None:
            raise ValueError("edge_index must be provided for STNET_pFedCTP")

        # TP
        h_gru, tp_pred = self.shareModel(x)

        # SS + ST (GAT 吃 h_gru)
        h_ss, h_gat = self.spatialModel(self.edge_index, h_gru)

        # HyperNetwork
        h_st = self.stPare(h_gat)

        # Final predictor
        st_output = self.stPredictor(torch.cat([h_ss, h_st, tp_pred], dim=-1))
        return st_output
