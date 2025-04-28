import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import k_hop_subgraph, to_undirected, negative_sampling
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load data
def load_data():
    # Load nodes
    unique_nodes = pd.read_csv("unique_nodes.csv")
    nodes = list(unique_nodes["node"])

    # Load edges with weights
    df_edges = pd.read_csv("ingredient_cooccur_graph.csv")
    edges = list(zip(df_edges["source"], df_edges["target"]))
    weights = {
        (u, v): w
        for u, v, w in zip(df_edges["source"], df_edges["target"], df_edges["weight"])
    }

    # Load node features
    feature_df = pd.read_csv("feature_matrix.csv")
    feature_dict = {}
    for _, row in feature_df.iterrows():
        node = row['node']
        features = row.iloc[1:].values.astype(np.float32)  # Skip 'node' column
        feature_dict[node] = features

    return nodes, edges, weights, feature_dict


# Create a scaled feature matrix for future use
def create_feature_matrix(nodes, feature_dict):
    feature_dim = len(next(iter(feature_dict.values())))
    feature_matrix = np.zeros((len(nodes), feature_dim), dtype=np.float32)

    for i, node in enumerate(nodes):
        if node in feature_dict:
            feature_matrix[i] = feature_dict[node]

    # Scale features to [0, 1] range for better training
    for j in range(feature_dim):
        col = feature_matrix[:, j]
        min_val, max_val = np.min(col), np.max(col)
        if max_val > min_val:
            feature_matrix[:, j] = (col - min_val) / (max_val - min_val)

    return feature_matrix


# Split data
def split_data(edges):
    e_train, e_tmp = train_test_split(edges, test_size=0.30, random_state=42)
    e_val, e_test = train_test_split(e_tmp, test_size=0.50, random_state=42)
    return e_train, e_val, e_test


# Build graph for training
def build_graph(nodes, edges, weights):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from([
        (u, v, weights.get((u, v), weights.get((v, u), 1)))
        for u, v in edges
    ])
    return G


# Sample negative edges
def sample_neg_edges(G, n, forbidden):
    nodes = list(G.nodes())
    negs = set()
    while len(negs) < n:
        u, v = np.random.choice(nodes, 2, replace=False)
        if (u, v) not in forbidden and (v, u) not in forbidden and not G.has_edge(u, v):
            negs.add((u, v))
    return list(negs)


# Convert to PyTorch Geometric format
def prepare_pyg_data(G, feature_matrix, nodes):
    # Create node mapping
    node_mapping = {node: i for i, node in enumerate(nodes)}

    # Prepare edge index
    edge_index = []
    for u, v in G.edges():
        edge_index.append([node_mapping[u], node_mapping[v]])
        edge_index.append([node_mapping[v], node_mapping[u]])  # Add reverse edge
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Prepare edge weights
    edge_weights = []
    for u, v in G.edges():
        w = G[u][v].get('weight', 1.0)
        edge_weights.append(w)
        edge_weights.append(w)  # For reverse edge
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    # Prepare node features
    x = torch.tensor(feature_matrix, dtype=torch.float)

    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights)

    return data, node_mapping


# Extract enclosing subgraph for node pairs
def extract_enclosing_subgraphs(data, node_pairs, node_mapping, k=2):
    subgraphs = []

    for u, v in node_pairs:
        u_idx, v_idx = node_mapping[u], node_mapping[v]

        # Get k-hop neighbors for both nodes
        u_nodes, u_edge_index, _, _ = k_hop_subgraph(
            u_idx, k, data.edge_index, relabel_nodes=False,
            num_nodes=data.num_nodes)

        v_nodes, v_edge_index, _, _ = k_hop_subgraph(
            v_idx, k, data.edge_index, relabel_nodes=False,
            num_nodes=data.num_nodes)

        # Combine the nodes and get the induced subgraph
        nodes = torch.cat([u_nodes, v_nodes]).unique()

        # Get the induced subgraph
        subgraph_edge_index = []
        subgraph_edge_attr = []

        for i in range(data.edge_index.size(1)):
            src, dst = data.edge_index[0, i], data.edge_index[1, i]
            if src in nodes and dst in nodes:
                subgraph_edge_index.append([src.item(), dst.item()])
                if data.edge_attr is not None:
                    subgraph_edge_attr.append(data.edge_attr[i].item())

        # If no edges in subgraph, create a dummy edge between u and v (will be masked later)
        if len(subgraph_edge_index) == 0:
            subgraph_edge_index = [[u_idx.item(), v_idx.item()]]
            subgraph_edge_attr = [0.0]  # Weight 0 indicates dummy edge

        # Convert to tensor
        subgraph_edge_index = torch.tensor(subgraph_edge_index, dtype=torch.long).t().contiguous()

        if len(subgraph_edge_attr) > 0:
            subgraph_edge_attr = torch.tensor(subgraph_edge_attr, dtype=torch.float)
        else:
            subgraph_edge_attr = None

        # Create node mapping for the subgraph
        subgraph_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes.tolist())}

        # Relabel nodes
        new_edge_index = torch.zeros_like(subgraph_edge_index)
        for i in range(subgraph_edge_index.size(1)):
            src = subgraph_edge_index[0, i].item()
            dst = subgraph_edge_index[1, i].item()
            new_edge_index[0, i] = subgraph_mapping[src]
            new_edge_index[1, i] = subgraph_mapping[dst]

        # Create subgraph features
        subgraph_x = data.x[nodes]

        # Identify the new indices of u and v
        new_u_idx = subgraph_mapping.get(u_idx, -1)
        new_v_idx = subgraph_mapping.get(v_idx, -1)

        # Create positional encoding for u and v (one-hot)
        pos_encoding = torch.zeros(subgraph_x.size(0), 2)
        if new_u_idx >= 0:
            pos_encoding[new_u_idx, 0] = 1.0
        if new_v_idx >= 0:
            pos_encoding[new_v_idx, 1] = 1.0

        # Add structural features: distance to u and v
        struct_features = torch.zeros(subgraph_x.size(0), 2)

        # Create a small nx graph to compute distances
        sg = nx.Graph()
        for i in range(new_edge_index.size(1)):
            src = new_edge_index[0, i].item()
            dst = new_edge_index[1, i].item()
            sg.add_edge(src, dst)

        # Compute shortest paths using networkx
        if new_u_idx >= 0 and new_v_idx >= 0:
            # Distance to u
            for node_idx in range(subgraph_x.size(0)):
                try:
                    struct_features[node_idx, 0] = nx.shortest_path_length(sg, node_idx, new_u_idx)
                except nx.NetworkXNoPath:
                    struct_features[node_idx, 0] = subgraph_x.size(0)  # Max possible distance

                try:
                    struct_features[node_idx, 1] = nx.shortest_path_length(sg, node_idx, new_v_idx)
                except nx.NetworkXNoPath:
                    struct_features[node_idx, 1] = subgraph_x.size(0)  # Max possible distance

        # Normalize structural features
        max_dist = subgraph_x.size(0)
        struct_features = struct_features / max_dist

        # Combine node features, positional encoding, and structural features
        augmented_x = torch.cat([subgraph_x, pos_encoding, struct_features], dim=1)

        # Create subgraph data object
        subgraph = Data(
            x=augmented_x,
            edge_index=new_edge_index,
            edge_attr=subgraph_edge_attr,
            u_idx=torch.tensor([new_u_idx]),
            v_idx=torch.tensor([new_v_idx]),
            num_nodes=subgraph_x.size(0)
        )

        subgraphs.append(subgraph)

    return subgraphs


# Attention-based subgraph GNN model
class SubgraphAttentionGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, heads=4, dropout=0.2):
        super(SubgraphAttentionGNN, self).__init__()

        # GNN layers
        self.convs = nn.ModuleList()

        # First layer: GAT with multiple heads
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))

        # Last layer
        self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))

        # Node scoring layer
        self.node_score = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

        # Edge prediction MLP
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        u_idx, v_idx = data.u_idx.item(), data.v_idx.item()

        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)

        # Extract central node embeddings
        if u_idx >= 0 and v_idx >= 0:
            u_emb = x[u_idx]
            v_emb = x[v_idx]

            # Combine embeddings for link prediction
            combined = torch.cat([u_emb, v_emb], dim=0)

            # Predict link
            pred = self.edge_predictor(combined)
            return torch.sigmoid(pred)
        else:
            # If either node is missing from subgraph, predict no edge
            return torch.tensor([0.0], device=x.device)


# Heterogeneous model combining subgraph and direct node information
class HeteroSubgraphGNN(nn.Module):
    def __init__(self, in_channels, node_channels, hidden_channels=64):
        super(HeteroSubgraphGNN, self).__init__()

        # Subgraph GNN with attention
        self.subgraph_gnn = SubgraphAttentionGNN(in_channels, hidden_channels)

        # Direct node feature processing
        self.node_encoder = nn.Sequential(
            nn.Linear(node_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 1, hidden_channels),  # +1 for cosine similarity
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, data, node_feats_u, node_feats_v):
        # Get subgraph-based prediction
        subgraph_pred = self.subgraph_gnn(data)

        # Process direct node features
        u_emb = self.node_encoder(node_feats_u)
        v_emb = self.node_encoder(node_feats_v)

        # Calculate cosine similarity between node embeddings
        cosine_sim = F.cosine_similarity(u_emb, v_emb, dim=0).unsqueeze(0)

        # Concatenate
        combined = torch.cat([u_emb, v_emb, cosine_sim], dim=0)

        # Final prediction
        final_pred = self.classifier(combined)
        return torch.sigmoid(final_pred)


# Training function with cosine similarity feature
def train_hetero_model(model, train_data, val_data, node_features, node_mapping, optimizer, epochs=100):
    best_val_loss = float('inf')
    patience = 10
    counter = 0

    train_subgraphs, train_pairs, train_labels = train_data
    val_subgraphs, val_pairs, val_labels = val_data

    train_dataset = list(zip(train_subgraphs, train_pairs, train_labels))
    val_dataset = list(zip(val_subgraphs, val_pairs, val_labels))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0

        for batch in train_loader:
            subgraph, (u, v), label = batch[0], batch[1][0], batch[2][0]
            subgraph = subgraph.to(device)

            # Get node features
            u_idx, v_idx = node_mapping[u], node_mapping[v]
            u_feat = node_features[u_idx].unsqueeze(0).to(device)
            v_feat = node_features[v_idx].unsqueeze(0).to(device)

            # Forward pass
            optimizer.zero_grad()
            out = model(subgraph, u_feat, v_feat)

            # Loss calculation
            target = torch.tensor([float(label)], device=device)
            loss = F.binary_cross_entropy(out.view(-1), target)

            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                subgraph, (u, v), label = batch[0], batch[1][0], batch[2][0]
                subgraph = subgraph.to(device)

                # Get node features
                u_idx, v_idx = node_mapping[u], node_mapping[v]
                u_feat = node_features[u_idx].unsqueeze(0).to(device)
                v_feat = node_features[v_idx].unsqueeze(0).to(device)

                # Forward pass
                out = model(subgraph, u_feat, v_feat)

                # Loss calculation
                target = torch.tensor([float(label)], device=device)
                val_loss += F.binary_cross_entropy(out.view(-1), target).item()

        # Print progress
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Train Loss: {total_loss / len(train_loader):.4f}, '
                  f'Val Loss: {val_loss / len(val_loader):.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_hetero_model.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    # Load best model
    model.load_state_dict(torch.load('best_hetero_model.pt'))
    return model


# Evaluation function for hetero model
def evaluate_hetero_model(model, test_data, node_features, node_mapping):
    model.eval()
    test_subgraphs, test_pairs, test_labels = test_data

    test_dataset = list(zip(test_subgraphs, test_pairs, test_labels))
    test_loader = DataLoader(test_dataset, batch_size=1)

    preds = []
    labels = []

    with torch.no_grad():
        for batch in test_loader:
            subgraph, (u, v), label = batch[0], batch[1][0], batch[2][0]
            subgraph = subgraph.to(device)

            # Get node features
            u_idx, v_idx = node_mapping[u], node_mapping[v]
            u_feat = node_features[u_idx].unsqueeze(0).to(device)
            v_feat = node_features[v_idx].unsqueeze(0).to(device)

            # Forward pass
            out = model(subgraph, u_feat, v_feat)

            preds.append(out.item())
            labels.append(label)

    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)

    return auc, ap


# Main function with both models
def main():
    print("Loading data...")
    nodes, edges, weights, feature_dict = load_data()
    e_train, e_val, e_test = split_data(edges)

    print("Building graph...")
    G_train = build_graph(nodes, e_train, weights)

    print("Creating negative samples...")
    forbidden = set(edges) | set((v, u) for u, v in edges)
    e_train_neg = sample_neg_edges(G_train, len(e_train), forbidden)
    e_val_neg = sample_neg_edges(G_train, len(e_val), forbidden)
    e_test_neg = sample_neg_edges(G_train, len(e_test), forbidden)

    print("Preparing feature matrix...")
    feature_matrix = create_feature_matrix(nodes, feature_dict)

    print("Preparing PyG data...")
    pyg_data, node_mapping = prepare_pyg_data(G_train, feature_matrix, nodes)
    node_features = torch.tensor(feature_matrix, dtype=torch.float).to(device)

    print("Extracting subgraphs for training...")
    train_pairs = e_train + e_train_neg
    train_labels = [1] * len(e_train) + [0] * len(e_train_neg)
    train_subgraphs = extract_enclosing_subgraphs(pyg_data, train_pairs, node_mapping, k=2)

    print("Extracting subgraphs for validation...")
    val_pairs = e_val + e_val_neg
    val_labels = [1] * len(e_val) + [0] * len(e_val_neg)
    val_subgraphs = extract_enclosing_subgraphs(pyg_data, val_pairs, node_mapping, k=2)

    print("Extracting subgraphs for testing...")
    test_pairs = e_test + e_test_neg
    test_labels = [1] * len(e_test) + [0] * len(e_test_neg)
    test_subgraphs = extract_enclosing_subgraphs(pyg_data, test_pairs, node_mapping, k=2)

    # Group data for easier handling
    train_data = (train_subgraphs, train_pairs, train_labels)
    val_data = (val_subgraphs, val_pairs, val_labels)
    test_data = (test_subgraphs, test_pairs, test_labels)

    # 1. Train and evaluate basic subgraph attention GNN
    print("\n=== Training Subgraph Attention GNN ===")
    in_channels = pyg_data.x.shape[1] + 4  # Original features + 2 position indicators + 2 distance features

    # Create data loaders with actual Data objects (not pairs)
    for i, subgraph in enumerate(train_subgraphs):
        subgraph.y = torch.tensor([train_labels[i]], dtype=torch.float)

    for i, subgraph in enumerate(val_subgraphs):
        subgraph.y = torch.tensor([val_labels[i]], dtype=torch.float)

    for i, subgraph in enumerate(test_subgraphs):
        subgraph.y = torch.tensor([test_labels[i]], dtype=torch.float)

    train_loader = DataLoader(train_subgraphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subgraphs, batch_size=32)
    test_loader = DataLoader(test_subgraphs, batch_size=32)

    # Initialize and train model
    model1 = SubgraphAttentionGNN(in_channels).to(device)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001, weight_decay=1e-4)

    # Train model function similar to before
    best_val_loss = float('inf')
    patience = 10
    counter = 0
    epochs = 100

    for epoch in range(epochs):
        # Training
        model1.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer1.zero_grad()

            # Get prediction
            out = model1(batch)

            # Compute loss
            target = batch.y.float().to(device)
            loss = F.binary_cross_entropy(out.view(-1), target)

            # Backward and optimize
            loss.backward()
            optimizer1.step()
            total_loss += loss.item()

        # Validation
        model1.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model1(batch)
                target = batch.y.float().to(device)
                val_loss += F.binary_cross_entropy(out.view(-1), target).item()

        # Print progress
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Train Loss: {total_loss / len(train_loader):.4f}, '
                  f'Val Loss: {val_loss / len(val_loader):.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model1.state_dict(), 'best_model1.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    # Load best model
    model1.load_state_dict(torch.load('best_model1.pt'))

    # Evaluate model
    model1.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model1(batch)
            preds.append(out.view(-1).cpu())
            labels.append(batch.y.cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    auc1 = roc_auc_score(labels, preds)
    ap1 = average_precision_score(labels, preds)

    print(f"Test metrics for Subgraph Attention GNN: AUC: {auc1:.4f}, AP: {ap1:.4f}")

    # 2. Train and evaluate heterogeneous model
    print("\n=== Training Heterogeneous Subgraph GNN ===")
    node_channels = pyg_data.x.shape[1]
    model2 = HeteroSubgraphGNN(in_channels, node_channels).to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001, weight_decay=1e-4)

    # Train model
    model2 = train_hetero_model(model2, train_data, val_data, node_features, node_mapping, optimizer2)

    # Evaluate on test set
    auc2, ap2 = evaluate_hetero_model(model2, test_data, node_features, node_mapping)
    print(f"Test metrics for Heterogeneous Subgraph GNN: AUC: {auc2:.4f}, AP: {ap2:.4f}")

    # 3. Evaluate on rare node pairs
    deg = dict(G_train.degree())
    threshold = np.percentile(list(deg.values()), 25)
    rare_pos = [e for e in e_test if deg[e[0]] <= threshold and deg[e[1]] <= threshold]
    rare_neg = [e for e in e_test_neg if deg[e[0]] <= threshold and deg[e[1]] <= threshold]

    print(f"\nTesting on rare node pairs (degree <= {threshold:.1f}):")
    print(f"Number of positive rare pairs: {len(rare_pos)}")
    print(f"Number of negative rare pairs: {len(rare_neg)}")

    rare_pairs = rare_pos + rare_neg
    rare_labels = [1] * len(rare_pos) + [0] * len(rare_neg)
    rare_subgraphs = extract_enclosing_subgraphs(pyg_data, rare_pairs, node_mapping, k=2)

    # Add labels to subgraphs
    for i, subgraph in enumerate(rare_subgraphs):
        subgraph.y = torch.tensor([rare_labels[i]], dtype=torch.float)

    rare_loader = DataLoader(rare_subgraphs, batch_size=32)

    # Evaluate model1 on rare pairs
    model1.eval()
    rare_preds = []
    rare_labels_tensor = []

    with torch.no_grad():
        for batch in rare_loader:
            batch = batch.to(device)
            out = model1(batch)
            rare_preds.append(out.view(-1).cpu())
            rare_labels_tensor.append(batch.y.cpu())

    rare_preds = torch.cat(rare_preds).numpy()
    rare_labels_tensor = torch.cat(rare_labels_tensor).numpy()

    rare_auc1 = roc_auc_score(rare_labels_tensor, rare_preds)
    rare_ap1 = average_precision_score(rare_labels_tensor, rare_preds)

    print(f"Rare test metrics for Subgraph Attention GNN: AUC: {rare_auc1:.4f}, AP: {rare_ap1:.4f}")

    # Evaluate model2 on rare pairs
    rare_data = (rare_subgraphs, rare_pairs, rare_labels)
    rare_auc2, rare_ap2 = evaluate_hetero_model(model2, rare_data, node_features, node_mapping)
    print(f"Rare test metrics for Heterogeneous Subgraph GNN: AUC: {rare_auc2:.4f}, AP: {rare_ap2:.4f}")

    # 4. Run and compare with baseline methods
    print("\n=== Comparing with Baseline Methods ===")

    # Run baseline.py and collect results
    import subprocess
    try:
        baseline_results = subprocess.check_output(['python', 'baseline.py'], universal_newlines=True)
        print("Baseline results:")
        print(baseline_results)
    except Exception as e:
        print(f"Failed to run baseline.py: {e}")
        print("Manually recording our GNN results for comparison:")

    # Summary of our GNN results
    print("\nSummary of GNN Model Results:")
    print("Original Test Set:")
    print(f"  Subgraph Attention GNN:     AUC: {auc1:.4f}, AP: {ap1:.4f}")
    print(f"  Heterogeneous Subgraph GNN: AUC: {auc2:.4f}, AP: {ap2:.4f}")
    print("Rare Test Set:")
    print(f"  Subgraph Attention GNN:     AUC: {rare_auc1:.4f}, AP: {rare_ap1:.4f}")
    print(f"  Heterogeneous Subgraph GNN: AUC: {rare_auc2:.4f}, AP: {rare_ap2:.4f}")


if __name__ == "__main__":
    main()
