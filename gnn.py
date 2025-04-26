import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# — 1. Load edge list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
df_edges = pd.read_csv("ingredient_cooccur_graph.csv")
edges     = list(zip(df_edges["source"], df_edges["target"]))
weights   = {(u, v): w for u, v, w in zip(df_edges["source"], df_edges["target"], df_edges["weight"]) }

# — 2. Load features
df_feat    = pd.read_csv("feature_matrix.csv", index_col="node")  

# —— 保持和训练图完全一致的节点顺序
nodes_full = sorted(set(df_edges["source"]) | set(df_edges["target"]))
df_feat    = df_feat.loc[nodes_full]      

# —— 构造张量
X          = torch.tensor(df_feat.values.astype(np.float32), device=device)

# —— 建立索引映射
idx_map    = {n: i for i, n in enumerate(nodes_full)}


# — 3. Split edges
e_train, e_tmp  = train_test_split(edges, test_size=0.30, random_state=42)
e_val,   e_test = train_test_split(e_tmp,   test_size=0.50, random_state=42)

# — 4. Build train graph
G_train = nx.Graph()
G_train.add_nodes_from(nodes_full)
G_train.add_weighted_edges_from([
    (u, v, weights.get((u, v), weights.get((v, u), 1.0)))
    for u, v in e_train
])

# compute degree for rare-split (you can use weighted degree if preferred)
deg = dict(G_train.degree())

# — 5. Negative sampling helper
def sample_neg(n, forbidden, G):
    negs = set()
    while len(negs) < n:
        u, v = np.random.choice(nodes_full, 2, replace=False)
        if ((u, v) not in forbidden and (v, u) not in forbidden and not G.has_edge(u, v)):
            negs.add((u, v))
    return list(negs)

forbidden   = set(edges)
e_train_neg = sample_neg(len(e_train), forbidden, G_train)
e_val_neg   = sample_neg(len(e_val),   forbidden, G_train)
e_test_neg  = sample_neg(len(e_test),  forbidden, G_train)

# — 6. Build normalized adjacency
N = len(nodes_full)
A = np.zeros((N, N), dtype=np.float32)
for u, v in e_train:
    i, j = idx_map[u], idx_map[v]
    w    = weights.get((u, v), weights.get((v, u), 1.0))
    A[i, j] = A[j, i] = w

deg_array = A.sum(axis=1)
D_inv_s   = np.diag(1.0 / np.sqrt(deg_array + 1e-9))
A_norm    = torch.tensor(D_inv_s @ A @ D_inv_s, device=device)

# — 7. Define GCN
class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hid_dim=64):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
    def encode(self, x, A):
        h = A @ x
        h = F.relu(self.lin1(h))
        h = A @ h
        h = F.relu(self.lin2(h))
        return h

# Instantiate
model     = SimpleGCN(in_dim=X.shape[1], hid_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

# Prepare data
train_pairs = e_train + e_train_neg
train_lbls  = torch.cat([torch.ones(len(e_train)), torch.zeros(len(e_train_neg))], dim=0).to(device)
val_pairs   = e_val   + e_val_neg
val_lbls    = torch.cat([torch.ones(len(e_val)),   torch.zeros(len(e_val_neg))],   dim=0).to(device)

# — 8. Train with Early Stopping
best_val_loss = float('inf')
patience      = 10
patience_cnt  = 0
best_state    = None

for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    Z = model.encode(X, A_norm)
    logits = torch.stack([
        (Z[idx_map[u]] * Z[idx_map[v]]).sum()
        for u, v in train_pairs
    ])
    loss = criterion(logits, train_lbls)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        Z_val      = model.encode(X, A_norm)
        logits_val = torch.stack([
            (Z_val[idx_map[u]] * Z_val[idx_map[v]]).sum()
            for u, v in val_pairs
        ])
        val_loss = criterion(logits_val, val_lbls)

    print(f"Epoch {epoch:2d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_state    = model.state_dict()
        patience_cnt  = 0
    else:
        patience_cnt += 1
        if patience_cnt >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# — 9. Test evaluation on two splits
model.eval()
with torch.no_grad():
    Z_np = model.encode(X, A_norm).cpu().numpy()

# -- Original test set
orig_pairs = e_test + e_test_neg
orig_lbls  = np.array([1]*len(e_test) + [0]*len(e_test_neg))
orig_scores = [
    float(
        np.dot(Z_np[idx_map[u]], Z_np[idx_map[v]]) /
        (np.linalg.norm(Z_np[idx_map[u]]) * np.linalg.norm(Z_np[idx_map[v]]) + 1e-9)
    )
    for u, v in orig_pairs
]
orig_auc = roc_auc_score(orig_lbls, orig_scores)
orig_ap  = average_precision_score(orig_lbls, orig_scores)
print(f"\nGCN + Cosine on Original test → AUC: {orig_auc:.4f}, AP: {orig_ap:.4f}")

# -- Rare test set (degree ≤ 25th percentile)
threshold = np.percentile(list(deg.values()), 25)
rare_pos  = [e for e in e_test     if deg[e[0]] <= threshold and deg[e[1]] <= threshold]
rare_neg  = [e for e in e_test_neg if deg[e[0]] <= threshold and deg[e[1]] <= threshold]
rare_pairs = rare_pos + rare_neg
rare_lbls  = np.array([1]*len(rare_pos) + [0]*len(rare_neg))
rare_scores = [
    float(
        np.dot(Z_np[idx_map[u]], Z_np[idx_map[v]]) /
        (np.linalg.norm(Z_np[idx_map[u]]) * np.linalg.norm(Z_np[idx_map[v]]) + 1e-9)
    )
    for u, v in rare_pairs
]
rare_auc = roc_auc_score(rare_lbls, rare_scores)
rare_ap  = average_precision_score(rare_lbls, rare_scores)
print(f"GCN + Cosine on Rare   test → AUC: {rare_auc:.4f}, AP: {rare_ap:.4f}")
