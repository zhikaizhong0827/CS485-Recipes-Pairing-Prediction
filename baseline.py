import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, average_precision_score

# 1. 读取边列表，并保留权重
df_edges = pd.read_csv("ingredient_cooccur_graph.csv")
edges = list(zip(df_edges["source"], df_edges["target"]))
weights = {
    (u, v): w
    for u, v, w in zip(df_edges["source"], df_edges["target"], df_edges["weight"])
}

# 2. 拆分正例
e_train, e_tmp = train_test_split(edges, test_size=0.30, random_state=42)
e_val, e_test = train_test_split(e_tmp, test_size=0.50, random_state=42)

# 3. 构建训练图（加权）
G_train = nx.Graph()
G_train.add_nodes_from(set(df_edges["source"]) | set(df_edges["target"]))
G_train.add_weighted_edges_from([
    (u, v, weights.get((u, v), weights.get((v, u), 1)))
    for u, v in e_train
])
nodes = list(G_train.nodes())

# 4. 计算 cooc 和 PMI 所需的统计量
train_cooc = nx.get_edge_attributes(G_train, 'weight')
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

# 5. 构建矩阵 A 并做 Truncated SVD 嵌入
idx_map = { node: i for i, node in enumerate(nodes) }
A = np.zeros((len(nodes), len(nodes)))
for (u, v), w in train_cooc.items():
    i, j = idx_map[u], idx_map[v]
    A[i, j] = w
    A[j, i] = w
svd = TruncatedSVD(n_components=20, random_state=42)
emb = normalize(svd.fit_transform(A))

def svd_cos(u, v):
    return float(np.dot(emb[idx_map[u]], emb[idx_map[v]]))

# 6. 定义 CN 和 AA
cn  = lambda u, v: len(list(nx.common_neighbors(G_train, u, v)))
aa_index = { (u, v): s for u, v, s in nx.adamic_adar_index(G_train) }

# 7. 负采样（测试集）
def sample_neg_edges(G, n, forbidden):
    negs = set()
    while len(negs) < n:
        u, v = np.random.choice(nodes, 2, replace=False)
        if (u, v) not in forbidden and (v, u) not in forbidden and not G.has_edge(u, v):
            negs.add((u, v))
    return list(negs)

forbidden = set(edges)
e_test_neg = sample_neg_edges(G_train, len(e_test), forbidden)

# 8. 在原始测试集上评估五种方法
test_pairs_orig  = e_test + e_test_neg
test_labels_orig = [1]*len(e_test) + [0]*len(e_test_neg)

scores_orig = {
    'Cooc': [train_cooc.get(p, train_cooc.get((p[1],p[0]), 0)) for p in test_pairs_orig],
    'PMI':  [pmi.get(p,  pmi.get((p[1],p[0]), 0))               for p in test_pairs_orig],
    'CN':   [cn(u, v)                                          for u, v in test_pairs_orig],
    'AA':   [aa_index.get(p, aa_index.get((p[1],p[0]), 0))    for p in test_pairs_orig],
    'SVD':  [svd_cos(u, v)                                     for u, v in test_pairs_orig],
}

print("Original test metrics:")
for name, sc in scores_orig.items():
    auc = roc_auc_score(test_labels_orig, sc)
    ap  = average_precision_score(test_labels_orig, sc)
    print(f"  {name:4s} → AUC: {auc:.4f}, AP: {ap:.4f}")

# 9. Rare 子集测试（度在前 25% 的节点）
deg = dict(G_train.degree())
threshold = np.percentile(list(deg.values()), 25)
rare_pos = [e for e in e_test     if deg[e[0]] <= threshold and deg[e[1]] <= threshold]
rare_neg = [e for e in e_test_neg if deg[e[0]] <= threshold and deg[e[1]] <= threshold]

test_pairs_rare  = rare_pos + rare_neg
test_labels_rare = [1]*len(rare_pos) + [0]*len(rare_neg)

scores_rare = {
    'Cooc': [train_cooc.get(p, train_cooc.get((p[1],p[0]), 0)) for p in test_pairs_rare],
    'PMI':  [pmi.get(p,  pmi.get((p[1],p[0]), 0))               for p in test_pairs_rare],
    'CN':   [cn(u, v)                                          for u, v in test_pairs_rare],
    'AA':   [aa_index.get(p, aa_index.get((p[1],p[0]), 0))    for p in test_pairs_rare],
    'SVD':  [svd_cos(u, v)                                     for u, v in test_pairs_rare],
}

print("\nRare test metrics:")
for name, sc in scores_rare.items():
    auc = roc_auc_score(test_labels_rare, sc)
    ap  = average_precision_score(test_labels_rare, sc)
    print(f"  {name:4s} → AUC: {auc:.4f}, AP: {ap:.4f}")
