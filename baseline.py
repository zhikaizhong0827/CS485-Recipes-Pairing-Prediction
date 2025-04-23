import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, average_precision_score

# — 1. 读边列表
df_edges = pd.read_csv("ingredient_cooccur_graph.csv")
edges = list(zip(df_edges["source"], df_edges["target"]))
weights = {
    (u, v): w for u, v, w in zip(df_edges["source"], df_edges["target"], df_edges["weight"])
}

# — 2. 拆分正例
e_train, e_tmp = train_test_split(edges, test_size=0.30, random_state=42)
e_val,   e_test = train_test_split(e_tmp,   test_size=0.50, random_state=42)

# — 3. 构建训练图
G_train = nx.Graph()
G_train.add_nodes_from(set(df_edges["source"]) | set(df_edges["target"]))
G_train.add_weighted_edges_from([
    (u, v, weights.get((u, v), weights.get((v, u), 1)))
    for u, v in e_train
])
nodes = list(G_train.nodes())

# — 4. 负采样函数
def sample_neg_edges(G, n, forbidden):
    negs = set()
    while len(negs) < n:
        u, v = np.random.choice(nodes, 2, replace=False)
        if (u, v) not in forbidden and (v, u) not in forbidden and not G.has_edge(u, v):
            negs.add((u, v))
    return list(negs)

forbidden = set(edges)
e_train_neg = sample_neg_edges(G_train, len(e_train), forbidden)
e_val_neg   = sample_neg_edges(G_train, len(e_val),   forbidden)
e_test_neg  = sample_neg_edges(G_train, len(e_test),  forbidden)

print("Dataset sizes:")
print(f"  Train → positive: {len(e_train):>4d}, negative: {len(e_train_neg):>4d}, total: {len(e_train)+len(e_train_neg):>4d}")
print(f"  Val   → positive: {len(e_val):>4d}, negative: {len(e_val_neg):>4d}, total: {len(e_val)+len(e_val_neg):>4d}")
print(f"  Test  → positive: {len(e_test):>4d}, negative: {len(e_test_neg):>4d}, total: {len(e_test)+len(e_test_neg):>4d}")

# — 5. 计算 Baseline 分数
# 5.1 共现计数
train_cooc = nx.get_edge_attributes(G_train, 'weight')

# 5.2 PMI
total_pairs = sum(train_cooc.values())
marginal = {}
for (u, v), w in train_cooc.items():
    marginal[u] = marginal.get(u, 0) + w
    marginal[v] = marginal.get(v, 0) + w

pmi = {}
for (u, v), w in train_cooc.items():
    pi, pj = marginal[u] / total_pairs, marginal[v] / total_pairs
    pij = w / total_pairs
    pmi[(u, v)] = np.log(pij / (pi * pj) + 1e-9)

# 5.3 图启发式
def cn(u, v): return len(list(nx.common_neighbors(G_train, u, v)))
aa_index = { (u, v): score for u, v, score in nx.adamic_adar_index(G_train) }

# 5.4 SVD + Cosine
idx_map = { node: i for i, node in enumerate(nodes) }
A = np.zeros((len(nodes), len(nodes)))
for (u, v), w in train_cooc.items():
    i, j = idx_map[u], idx_map[v]
    A[i, j] = w; A[j, i] = w

svd = TruncatedSVD(n_components=20, random_state=42)
emb = normalize(svd.fit_transform(A))

def svd_cos(u, v):
    return float(np.dot(emb[idx_map[u]], emb[idx_map[v]]))

# — 6. 在测试集上预测并评估
test_pairs = e_test + e_test_neg
test_labels = [1]*len(e_test) + [0]*len(e_test_neg)

scores = {
    'Cooc': [train_cooc.get(p, train_cooc.get((p[1],p[0]), 0)) for p in test_pairs],
    'PMI':  [pmi.get(p, pmi.get((p[1],p[0]), 0))       for p in test_pairs],
    'CN':   [cn(u, v)                                  for u, v in test_pairs],
    'AA':   [aa_index.get(p, aa_index.get((p[1],p[0]),0)) for p in test_pairs],
    'SVD':  [svd_cos(u, v)                             for u, v in test_pairs]
}

results = {}
for name, sc in scores.items():
    auc = roc_auc_score(test_labels, sc)
    ap  = average_precision_score(test_labels, sc)
    results[name] = (auc, ap)
    print(f"{name:5s} → AUC: {auc:.4f}, AP: {ap:.4f}")
