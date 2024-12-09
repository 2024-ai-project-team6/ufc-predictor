import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로드
file_path = '/content/large_dataset.csv'
df = pd.read_csv(file_path)

# 데이터 전처리
# 1. Remove rows where gender is "Women"
df = df[df['gender'] != 'Women']

# 2. Filter out weight_class categories with fewer than 50 rows
weight_class_counts = df['weight_class'].value_counts()
valid_weight_classes = weight_class_counts[weight_class_counts >= 50].index
df = df[df['weight_class'].isin(valid_weight_classes)]

# 3. One-hot encode r_stance and b_stance
df = pd.get_dummies(df, columns=['r_stance', 'b_stance'], prefix=['r_stance', 'b_stance'], drop_first=True)

# 4. Ensure numeric data
numeric_columns = [col for col in df.columns if col.startswith("r_") or col.startswith("b_")]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

# 5. 언더샘플링
def undersample(df, target_column='winner'):
    class_counts = df[target_column].value_counts()
    min_class_count = class_counts.min()

    # 각 클래스에서 동일한 수의 샘플을 무작위로 추출
    sampled_dfs = []
    for cls in class_counts.index:
        sampled_dfs.append(df[df[target_column] == cls].sample(n=min_class_count, random_state=42))

    # 데이터프레임 재구성
    return pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

df = undersample(df, target_column='winner')

# 동적 그래프 생성 함수
def create_dynamic_graph(df, sequence_length=10):
    dynamic_graphs = []
    weight_classes = df['weight_class'].unique()

    for weight_class in weight_classes:
        weight_class_data = df[df['weight_class'] == weight_class]
        graph = nx.DiGraph()
        sequence = []

        for idx, row in weight_class_data.iterrows():
            r_fighter = row['r_fighter']
            b_fighter = row['b_fighter']

            # Add nodes if not exist
            if r_fighter not in graph:
                graph.add_node(r_fighter, features={})
            if b_fighter not in graph:
                graph.add_node(b_fighter, features={})

            # Update node features
            for col in row.index:
                if col.startswith("r_"):
                    graph.nodes[r_fighter]['features'][col] = float(row[col])
                elif col.startswith("b_"):
                    graph.nodes[b_fighter]['features'][col] = float(row[col])

            # Add or update edge features
            edge = (r_fighter, b_fighter)
            winner = row['winner']
            if winner == "Red":
                graph.add_edge(*edge, winner=1)
            elif winner == "Blue":
                graph.add_edge(*edge, winner=0)

            # Append graph to sequence
            sequence.append(graph.copy())
            if len(sequence) == sequence_length:
                dynamic_graphs.append(sequence)
                sequence = []

    return dynamic_graphs

# 동적 그래프 시퀀스 생성
dynamic_graph_sequences = create_dynamic_graph(df)

# 데이터 분할 (6:2:2 비율로)
train_data, test_data = train_test_split(dynamic_graph_sequences, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# 모델 정의
class EvolveGCN_H(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EvolveGCN_H, self).__init__()
        self.hidden_dim = hidden_dim

        # GCN Layers
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # LSTM for dynamic feature update
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Fully connected layer for final prediction
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequences):
        """
        Args:
            sequences (list of graphs): A sequence of dynamic graphs.
        Returns:
            Output logits for the last graph in the sequence.
        """
        all_node_features = []  # Collect node features from the sequence

        for graph in sequences:
            # Extract node features and edge index
            x = torch.tensor(
                [list(graph.nodes[n]['features'].values()) for n in graph.nodes],
                dtype=torch.float,
            )
            edge_index = torch.tensor(
                list(graph.edges), dtype=torch.long
            ).t().contiguous()

            # Apply GCN layers
            x = self.gcn1(x, edge_index)
            x = torch.relu(x)
            x = self.gcn2(x, edge_index)
            x = torch.relu(x)

            # Collect node features
            all_node_features.append(x)

        # Stack all node features and pass through LSTM
        all_node_features = torch.stack(all_node_features, dim=1)
        lstm_out, _ = self.lstm(all_node_features)

        # Use the last output of LSTM for prediction
        output = self.fc(lstm_out[:, -1, :])
        return output

# 모델 학습 함수
def train_model(train_data, val_data, model, epochs=50, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sequence in train_data:
            optimizer.zero_grad()
            # Forward pass
            output = model(sequence)

            # Calculate loss using the last graph in the sequence
            last_graph = sequence[-1]
            y = torch.tensor(
                [last_graph.edges[e]['winner'] for e in last_graph.edges],
                dtype=torch.long,
            )
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for sequence in val_data:
                output = model(sequence)
                last_graph = sequence[-1]
                y = torch.tensor(
                    [last_graph.edges[e]['winner'] for e in last_graph.edges],
                    dtype=torch.long,
                )
                val_loss += loss_fn(output, y).item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss}, Val Loss: {val_loss}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

# 모델 초기화 및 학습
input_dim = len(numeric_columns)
hidden_dim = 64
output_dim = 2  # Red or Blue

model = EvolveGCN_H(input_dim, hidden_dim, output_dim)
train_model(train_data, val_data, model)

# 모델 테스트
def evaluate_model(test_data, model):
    model.eval()
    all_true = []
    all_preds = []

    with torch.no_grad():
        for sequence in test_data:
            output = model(sequence)
            last_graph = sequence[-1]
            y = torch.tensor(
                [last_graph.edges[e]['winner'] for e in last_graph.edges],
                dtype=torch.long,
            )

            # Get the predictions (Red or Blue)
            _, preds = torch.max(output, dim=1)

            all_true.extend(y.numpy())
            all_preds.extend(preds.numpy())

    # Accuracy 계산
    accuracy = accuracy_score(all_true, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 모델 평가
evaluate_model(test_data, model)
