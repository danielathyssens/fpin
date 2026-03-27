
import torch
import torch.nn as nn

class GraphEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=3, norm=True, dropout=0.1):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, norm, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, node_feats, edge_feats, mask=None):
        h = self.node_proj(node_feats)  # [B, N, H]
        e = self.edge_proj(edge_feats) if edge_feats.size(-1) == 1 else edge_feats  # [B, N, N, H]

        for layer in self.layers:
            h = layer(h, e, mask)

        return h


class GraphAttentionLayer(nn.Module):
    """ Self-attention over nodes, with:

    Q = W_q * h_i, K = W_k * h_j, V = W_v * h_j
    Scores from QKᵀ define attention weights.
    Edge features e_{ij} (distances) are projected
    and added to the values V_j during aggregation."""

    def __init__(self, hidden_dim, norm=True, dropout=0.1):
        super().__init__()
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.edge_update = nn.Linear(hidden_dim, hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_dim) if norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, e, mask=None):
        # h: [B, N, H], e: [B, N, N, H]
        Q, K, V = self.linear_q(h), self.linear_k(h), self.linear_v(h)  # [B, N, H]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (h.size(-1) ** 0.5)  # [B, N, N]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)  # [B, N, N]
        edge_term = self.edge_update(e)  # [B, N, N, H]
        #     attn: shape [B, N, N] → attention scores between all node pairs.
        #     V: shape [B, N, H] → value vectors.
        #     edge_term: shape [B, N, N, H] → projected edge embeddings (e.g. from distance matrix).
        #     V.unsqueeze(1) → [B, 1, N, H] gets broadcast across destination nodes i as V_j.
        #     V + edge_term → [B, N, N, H], where for each (i,j), you sum V_j + e_ij.
        #     Then matmul with attn.unsqueeze(-1) → [B, N, N, 1] gives the weighted sum over j.
        #     --> explicit form of the soft-attention message passing. ✅
        agg = torch.einsum("bij,bijh->bih", attn, V.unsqueeze(1) + edge_term)

        h = h + self.dropout(agg)
        h = self.norm1(h)

        h = h + self.dropout(self.ff(h))
        h = self.norm2(h)

        return h