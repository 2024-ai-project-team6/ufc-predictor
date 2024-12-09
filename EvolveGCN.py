!pip install torch torch-geometric joblib

import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from itertools import product
from joblib import Parallel, delayed

# 데이터 로드
file_path = '/content/large_dataset.csv'
df = pd.read_csv(file_path)

# 데이터 전처리
df = df[df['gender'] != 'Women']  # 1. Remove rows where gender is "Women"
weight_class_counts = df['weight_class'].value_counts()  # 2. Filter out weight_class categories
valid_weight_classes = weight_class_counts[weight_class_counts >= 50].index
df = df[df['weight_class'].isin(valid_weight_classes)]
df = pd.get_dummies(df, columns=['r_stance', 'b_stance'], prefix=['r_stance', 'b_stance'], drop_first=True)  # 3. One-hot encoding
numeric_columns = [col for col in df.columns if col.startswith("r_") or col.startswith("b_")]  # 4. Ensure numeric data
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

# 언더샘플링
def undersample(df, target_column='winner'):
    class_counts = df[target_column].value_counts()
    min_class_count = class_counts.min()
    sampled_dfs = [df[df[target_column] == cls].sample(n=min_class_count, random_state=42) for cls in class_counts.index]
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
            r_fighter, b_fighter = row['r_fighter'], row['b_fighter']
            if r_fighter not in graph:
                graph.add_node(r_fighter, features={})
            if b_fighter not in graph:
                graph.add_node(b_fighter, features={})
            for col in row.index:
                if col.startswith("r_"):
                    graph.nodes[r_fighter]['features'][col] = float(row[col])
                elif col.startswith("b_"):
                    graph.nodes[b_fighter]['features'][col] = float(row[col])
            edge = (r_fighter, b_fighter)
            graph.add_edge(*edge, winner=1 if row['winner'] == "Red" else 0)
            sequence.append(graph.copy())
            if len(sequence) == sequence_length:
                dynamic_graphs.append(sequence)
                sequence = []
    return dynamic_graphs

# 동적 그래프 시퀀스 생성
dynamic_graph_sequences = create_dynamic_graph(df)

# 모델 정의
class EvolveGCN_H(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EvolveGCN_H, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequences):
        all_node_features = []
        for graph in sequences:
            x = torch.tensor([list(graph.nodes[n]['features'].values()) for n in graph.nodes], dtype=torch.float)
            edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
            x = torch.relu(self.gcn1(x, edge_index))
            x = torch.relu(self.gcn2(x, edge_index))
            all_node_features.append(x)
        all_node_features = torch.stack(all_node_features, dim=1)
        lstm_out, _ = self.lstm(all_node_features)
        output = self.fc(lstm_out[:, -1, :])
        return output

# Parallelized cross-validation function
def hyperparameter_search_parallel(train_data, model_class, param_grid, k=5, n_jobs=-1):
    best_params, best_score = None, 0
    param_combinations = list(product(*param_grid.values()))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    def train_and_evaluate(param_dict, train_data, kf):
        scores = []
        hidden_dim = param_dict['hidden_dim']
        lr = param_dict['lr']

        for train_idx, val_idx in kf.split(train_data):
            train_fold = [train_data[i] for i in train_idx]
            val_fold = [train_data[i] for i in val_idx]

            # 모델 생성
            model = model_class(input_dim=len(numeric_columns), hidden_dim=hidden_dim, output_dim=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()

            # 간단한 학습 루프
            for _ in range(10):  # 10 에포크
                model.train()
                for sequence in train_fold:
                    optimizer.zero_grad()
                    output = model(sequence)
                    y = torch.tensor([sequence[-1].edges[e]['winner'] for e in sequence[-1].edges], dtype=torch.long)
                    loss = loss_fn(output, y)
                    loss.backward()
                    optimizer.step()

            # Validation
            model.eval()
            val_acc = []
            with torch.no_grad():
                for sequence in val_fold:
                    output = model(sequence)
                    y = torch.tensor([sequence[-1].edges[e]['winner'] for e in sequence[-1].edges], dtype=torch.long)
                    preds = torch.argmax(output, dim=1)
                    val_acc.append(accuracy_score(y.numpy(), preds.numpy()))
            scores.append(np.mean(val_acc))

        return np.mean(scores)

    # Parallel computation
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_and_evaluate)(dict(zip(param_grid.keys(), params)), train_data, kf)
        for params in param_combinations
    )

    # Find the best hyperparameters
    for param_idx, score in enumerate(results):
        if score > best_score:
            best_score = score
            best_params = dict(zip(param_grid.keys(), param_combinations[param_idx]))

    return best_params

# 하이퍼파라미터 탐색
param_grid = {
    'hidden_dim': [32, 64, 128],
    'lr': [0.001, 0.01, 0.1]
}
best_params = hyperparameter_search_parallel(dynamic_graph_sequences, EvolveGCN_H, param_grid, n_jobs=-1)

# 최적화된 하이퍼파라미터 출력
print("Best Hyperparameters:", best_params)

# 최적 하이퍼파라미터로 학습 및 평가
train_data, test_data = train_test_split(dynamic_graph_sequences, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

model = EvolveGCN_H(
    input_dim=len(numeric_columns),
    hidden_dim=best_params['hidden_dim'],
    output_dim=2
)

# 모델 학습 함수
def train_model(train_data, val_data, model, epochs=50, patience=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sequence in train_data:
            optimizer.zero_grad()
            output = model(sequence)
            y = torch.tensor([sequence[-1].edges[e]['winner'] for e in sequence[-1].edges], dtype=torch.long)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequence in val_data:
                output = model(sequence)
                y = torch.tensor([sequence[-1].edges[e]['winner'] for e in sequence[-1].edges], dtype=torch.long)
                val_loss += loss_fn(output, y).item()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

train_model(train_data, val_data, model, lr=best_params['lr'])

# 모델 테스트
def evaluate_model(test_data, model):
    model.eval()
    all_true, all_preds = [], []
    with torch.no_grad():
        for sequence in test_data:
            output = model(sequence)
            y = torch.tensor([sequence[-1].edges[e]['winner'] for e in sequence[-1].edges], dtype=torch.long)
            preds = torch.argmax(output, dim=1)
            all_true.extend(y.numpy())
            all_preds.extend(preds.numpy())
    acc = accuracy_score(all_true, all_preds)
    print(f"Test Accuracy: {acc:.4f}")

evaluate_model(test_data, model)
