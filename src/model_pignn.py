import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool

def laplacian_quadratic(node_emb, edge_index, edge_weight):
    """
    Σ_{(i,j)∈E} w_ij ||h_i - h_j||^2  (on the batched graph; PyG edge_index already offsets nodes)
    """
    if edge_index.numel() == 0:
        return node_emb.new_tensor(0.0)
    i, j = edge_index[0], edge_index[1]
    diff = node_emb[i] - node_emb[j]
    return (edge_weight.view(-1) * (diff.pow(2).sum(dim=1))).sum()

class PIGNN(nn.Module):
    """
    Graph encoder (SAGE/GCN) + global pooling + MLP head -> predicts [nadir, RoCoF, t_settle]
    Physics-informed penalty: Laplacian smoothness on node embeddings.
    """
    def __init__(self, in_dim, meta_dim=5, hidden=64, out_dim=3, conv='sage'):
        super().__init__()
        if conv == 'sage':
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
        else:
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, hidden)

        self.head = nn.Sequential(
            nn.Linear(hidden + meta_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, data):
        x = data.x
        ei = data.edge_index
        x = F.relu(self.conv1(x, ei))
        x = F.relu(self.conv2(x, ei))
        node_emb = x
        g = global_mean_pool(node_emb, data.batch)
        if hasattr(data, 'meta'):
            meta = data.meta
            if meta.dim() == 1:
                meta = meta.view(1, -1).repeat(g.size(0), 1)
            elif meta.size(0) != g.size(0):
                # If someone packed meta per-node by mistake, squeeze to per-graph
                meta = meta.view(g.size(0), -1)
            g = torch.cat([g, meta], dim=1)
        yhat = self.head(g)
        return yhat, node_emb

def loss_with_physics(pred, target, node_emb, edge_index, edge_attr, lambda_lap=1e-3):
    """
    pred: [B, 3], target: [B, 3] (or [1,3] per-graph pre-batch).
    If target arrives 1-D from an old dataset, coerce it to [B, 3].
    """
    # --- Fix target shape if needed ---
    if target.dim() == 1:
        # assume contiguous triples per graph: [nadir, rocof, t_settle, nadir, rocof, ...]
        target = target.view(-1, 3)
    elif target.dim() == 2 and target.size(0) != pred.size(0):
        # Some collates might produce [1,3] per-graph repeated weirdly.
        # Coerce to match batch size if total elements line up.
        if target.numel() == pred.size(0) * 3:
            target = target.view(pred.size(0), 3)

    # --- MSE main terms ---
    mse = F.mse_loss(pred[:, :2], target[:, :2])  # nadir, RoCoF
    mask = target[:, 2] >= 0
    if mask.any():
        mse = mse + F.mse_loss(pred[mask, 2], target[mask, 2])

    # --- Physics regularizer ---
    phys = laplacian_quadratic(node_emb, edge_index, edge_attr.view(-1) if edge_attr is not None else node_emb.new_zeros(0))
    return mse + lambda_lap * phys

