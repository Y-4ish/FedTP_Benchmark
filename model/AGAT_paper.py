"""PeMS-adapted ASTGAT backbone for FedAGAT (Information Sciences 667, 2024)."""

import torch
from torch import nn
from torch.nn import functional as F


class NetworkGenerator(nn.Module):
    def __init__(self, max_nodes, repeats=6):
        super().__init__()
        self.logits = nn.Parameter(torch.ones(repeats, max_nodes, max_nodes, 2))
        self.edge_weight = nn.Parameter(torch.zeros(repeats, max_nodes, max_nodes))

    def forward(self, ids, initial_graph):
        logits = self.logits.index_select(1, ids).index_select(2, ids)
        weights = self.edge_weight.index_select(1, ids).index_select(2, ids)
        sampled = F.gumbel_softmax(logits, tau=10.0, dim=-1)[..., 0]
        generated = (sampled * F.softplus(weights)).mean(dim=0)
        return 0.5 * generated + 0.5 * initial_graph


class ResidualEndConv(nn.Module):
    def __init__(self, prediction_steps):
        super().__init__()
        self.first = nn.Linear(48, 512)
        self.second = nn.Linear(512, 512)
        self.last = nn.Linear(512, prediction_steps)

    def forward(self, features):
        hidden = F.relu(self.first(features))
        hidden = F.relu(self.second(hidden)) + hidden
        return self.last(hidden)


class GatedTemporalConv(nn.Module):
    def __init__(self, input_width, output_width=128):
        super().__init__()
        self.first = nn.Conv2d(input_width, output_width, (1, 3), padding=(0, 1))
        self.second = nn.Conv2d(input_width, output_width, (1, 3), padding=(0, 1))
        self.gate = nn.Conv2d(input_width, output_width, (1, 3), padding=(0, 1))

    def forward(self, value):
        value = value.permute(0, 3, 1, 2)
        gate = torch.sigmoid(self.gate(value))
        result = gate * self.first(value) + (1 - gate) * self.second(value)
        return result.permute(0, 2, 3, 1)


class TalkingHeadsGraphAttention(nn.Module):
    def __init__(self, input_width, heads=6, concat=True):
        super().__init__()
        self.heads, self.concat = heads, concat
        self.values = nn.Linear(input_width, heads * 48, bias=False)
        self.source = nn.Parameter(torch.empty(heads, 48))
        self.target = nn.Parameter(torch.empty(heads, 48))
        self.pre_softmax = nn.Linear(heads, heads, bias=False)
        self.post_softmax = nn.Linear(heads, heads, bias=False)
        nn.init.xavier_uniform_(self.source)
        nn.init.xavier_uniform_(self.target)

    def forward(self, features, graph):
        batch, nodes, _ = features.shape
        values = self.values(features).reshape(batch, nodes, self.heads, 48)
        source = (values * self.source).sum(-1)
        target = (values * self.target).sum(-1)
        scores = F.leaky_relu(source[:, :, None, :] + target[:, None, :, :], 0.2)
        attention = self.post_softmax(torch.softmax(self.pre_softmax(scores), dim=2))
        result = torch.einsum('bijh,bjhd->bihd', attention * graph[None, :, :, None], values)
        if self.concat:
            return F.elu(result.reshape(batch, nodes, self.heads * 48))
        return F.elu(result.mean(dim=2))


class SpatialTemporalLayer(nn.Module):
    def __init__(self, input_width, steps, embedding_width=64, concat=True):
        super().__init__()
        self.steps = steps
        self.temporal = nn.Sequential(
            GatedTemporalConv(input_width),
            GatedTemporalConv(128),
            GatedTemporalConv(128),
        )
        self.spatial = TalkingHeadsGraphAttention(steps * 128 + embedding_width, concat=concat)
        output_width = 288 if concat else 48
        self.residual = nn.Linear(steps * input_width, output_width)
        self.norm = nn.LayerNorm(output_width)

    def forward(self, values, graph, embedding):
        batch, nodes, _, _ = values.shape
        temporal = self.temporal(values).reshape(batch, nodes, self.steps * 128)
        features = torch.cat((temporal, embedding[None].expand(batch, -1, -1)), dim=-1)
        result = self.spatial(features, graph)
        return F.relu(self.norm(result + self.residual(values.reshape(batch, nodes, -1))))


class ASTGAT(nn.Module):
    def __init__(self, num_nodes, in_dim, pred_len, adj, emb_dim=64, max_nodes=None, node_ids=None):
        super().__init__()
        self.nodes, self.steps = int(num_nodes), int(in_dim)
        max_nodes = int(max_nodes or num_nodes)
        self.register_buffer('adj', adj.float())
        ids = torch.arange(num_nodes) if node_ids is None else torch.as_tensor(node_ids, dtype=torch.long)
        if ids.numel() != num_nodes:
            raise ValueError('FedAGAT local node IDs do not match the graph')
        self.register_buffer('node_ids', ids)
        self.generator = NetworkGenerator(max_nodes)
        self.spatial_emb = nn.Embedding(max_nodes, emb_dim)
        self.layers = nn.ModuleList([
            SpatialTemporalLayer(1, self.steps, emb_dim),
            SpatialTemporalLayer(24, self.steps, emb_dim),
            SpatialTemporalLayer(24, self.steps, emb_dim, concat=False),
        ])
        self.end_conv = ResidualEndConv(pred_len)

    @staticmethod
    def graph_loss(features, graph):
        degree = graph.sum(1).clamp_min(1e-8)
        squared = features.square().sum(-1)
        propagated = torch.einsum('ij,bjd->bid', graph, features)
        smoothness = (degree[None] * squared).sum(-1) - (features * propagated).sum((1, 2))
        nodes = graph.shape[0]
        return smoothness.mean() / (nodes * nodes) - torch.log(degree).mean() + torch.linalg.vector_norm(graph) / (nodes * nodes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        if x.ndim != 4:
            raise ValueError(f'FedAGAT expected 4D input, got {tuple(x.shape)}')
        if x.shape[1] == self.steps and x.shape[2] == self.nodes:
            x = x.transpose(1, 2)
        if x.shape[1] != self.nodes or x.shape[2] != self.steps or x.shape[-1] != 1:
            raise ValueError(f'FedAGAT expected [B,{self.nodes},{self.steps},1], got {tuple(x.shape)}')
        graph = self.generator(self.node_ids, self.adj)
        embedding = self.spatial_emb(self.node_ids)
        hidden = x
        for index, layer in enumerate(self.layers):
            output = layer(hidden, graph, embedding)
            if index < 2:
                hidden = output.reshape(x.shape[0], self.nodes, self.steps, 24)
        prediction = self.end_conv(output).unsqueeze(-1)
        if self.training:
            return prediction, self.graph_loss(output, graph)
        return prediction
