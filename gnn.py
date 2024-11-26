import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from common import plot_training_history
from sklearn.metrics import f1_score, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
import numpy as np

# Step 1: Load the datasets
fighter_stats_path = './preprocessed_fighter_stats.csv'
event_dataset_path = './event_dataset_2.csv'

fighter_stats = pd.read_csv(fighter_stats_path)
event_dataset = pd.read_csv(event_dataset_path)

event_dataset['date'] = pd.to_datetime(event_dataset['date'])
event_dataset = event_dataset[event_dataset['date'] >= '1994-01-01']

# 데이터셋을 시간순으로 정렬
event_dataset = event_dataset.sort_values(by='date')

# 전체 데이터 크기 계산
total_size = len(event_dataset)

# train:validation:test = 8:1:1 비율로 분할
train_size = int(total_size * 0.8)
val_size = int(total_size * 0.1)

# 시간 순서대로 분할
train_dataset = event_dataset.iloc[:train_size]
val_dataset = event_dataset.iloc[train_size:train_size+val_size] 
test_dataset = event_dataset.iloc[train_size+val_size:]
print(train_dataset['date'].min(), train_dataset['date'].max())
print(val_dataset['date'].min(), val_dataset['date'].max())
print(test_dataset['date'].min(), test_dataset['date'].max())

print(f'Train set size: {len(train_dataset)}')
print(f'Validation set size: {len(val_dataset)}')
print(f'Test set size: {len(test_dataset)}')

# Step 2: Split the dataset by weight class and sort by date
weight_classes = event_dataset['weight_class'].unique()
sorted_weight_class_datasets = {}

for weight_class in weight_classes:
    wc_data = event_dataset[event_dataset['weight_class'] == weight_class]
    wc_data = wc_data.sort_values(by='date')
    sorted_weight_class_datasets[weight_class] = wc_data
    
    
# Step 3: Function to create a graph for GNN training
def prepare_link_prediction_data(wc_data, fighter_stats):
    # 하나의 그래프 생성 (모든 노드 포함)
    G = nx.DiGraph()
    
    # 모든 선수(노드)를 그래프에 추가
    for index, row in wc_data.iterrows():
        for fighter in [row['r_fighter'], row['b_fighter']]:
            if fighter not in G:
                fighter_data = fighter_stats[fighter_stats['name'] == fighter]
                if not fighter_data.empty:
                    attributes = fighter_data.iloc[0].to_dict()
                    del attributes['name']
                    numeric_attrs = {k: float(v) for k, v in attributes.items() 
                                   if isinstance(v, (int, float))}
                    G.add_node(fighter, **numeric_attrs)
    
    # 엣지 데이터 준비
    edges = []
    for index, row in wc_data.iterrows():
        r_fighter = row['r_fighter']
        b_fighter = row['b_fighter']
        winner = row['winner']
        
        if winner == 'Red':
            edges.append((r_fighter, b_fighter))
        elif winner == 'Blue':
            edges.append((b_fighter, r_fighter))
    
    # 엣지 데이터 분할
    total_edges = len(edges)
    train_size = int(total_edges * 0.8)
    val_size = int(total_edges * 0.1)
    
    train_edges = edges[:train_size]
    val_edges = edges[train_size:train_size+val_size]
    test_edges = edges[train_size+val_size:]
    print(f"train_edges: {len(train_edges)}, val_edges: {len(val_edges)}, test_edges: {len(test_edges)}")
    
    return {
        'graph': G,  # 노드 정보만 포함된 그래프
        'train_edges': train_edges,
        'val_edges': val_edges,
        'test_edges': test_edges
    }

# 각 체급별로 데이터 준비
data_by_weight_class = {}

for weight_class, wc_data in sorted_weight_class_datasets.items():
    data_by_weight_class[weight_class] = prepare_link_prediction_data(
        wc_data, fighter_stats
    )

# 데이터 통계 출력
for weight_class, data in data_by_weight_class.items():
    print(f"\n{weight_class}:")
    print(f"노드 수: {data['graph'].number_of_nodes()}")
    print(f"학습 엣지 수: {len(data['train_edges'])}")
    print(f"검증 엣지 수: {len(data['val_edges'])}")
    print(f"테스트 엣지 수: {len(data['test_edges'])}")

class DirectedLinkPredictionGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 2)  # 2 classes: direction (0: no edge or reverse, 1: forward)
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # edge_label_index: 2 x E tensor of node indices
        src, dst = edge_label_index
        src_embeddings = z[src]
        dst_embeddings = z[dst]
        # Concatenate source and destination embeddings
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
        # print(edge_embeddings)
        return self.classifier(edge_embeddings)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

def convert_to_pytorch_geometric_data(graph_data, weight_class):
    G = graph_data['graph']
    train_edges = graph_data['train_edges']
    
    # 노드 특성 행렬 생성
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # 노드 특성 텐서 생성
    node_features = []
    for node in nodes:
        features = [G.nodes[node][feat] for feat in G.nodes[node]]
        node_features.append(features)
    x = torch.tensor(node_features, dtype=torch.float)
    
    # 학습용 엣지 인덱스 텐서 생성
    edge_index = []
    for src, dst in train_edges:
        edge_index.append([node_to_idx[src], node_to_idx[dst]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    return Data(x=x, edge_index=edge_index), node_to_idx

def train_model(model, data, optimizer, edge_label_index, labels):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index, edge_label_index)
    loss = F.cross_entropy(out, labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    return loss.item()

def test_model(model, data, edge_label_index, labels):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, edge_label_index)
        pred = out.argmax(dim=-1)
        correct = (pred == labels).sum()
        acc = int(correct) / len(labels)
    return acc

def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f1

# 모델 학습
def train_weight_class_model(weight_class, graph_data):
    # 데이터 준비
    data, node_to_idx = convert_to_pytorch_geometric_data(graph_data, weight_class)
    model = DirectedLinkPredictionGNN(in_channels=data.x.size(1), hidden_channels=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    # 학습용 엣지와 레이블 준비
    train_edges = graph_data['train_edges']
    val_edges = graph_data['val_edges']
    
    # 학습 데이터 준비
    train_edge_pairs = []
    train_labels = []
    for src, dst in train_edges:
        train_edge_pairs.append([node_to_idx[src], node_to_idx[dst]])
        train_labels.append(1)
        train_edge_pairs.append([node_to_idx[dst], node_to_idx[src]])
        train_labels.append(0)
    
    train_edge_index = torch.tensor(train_edge_pairs, dtype=torch.long).t()
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    
    # 검증 데이터 준비
    val_edge_pairs = []
    val_labels = []
    for src, dst in val_edges:
        val_edge_pairs.append([node_to_idx[src], node_to_idx[dst]])
        val_labels.append(1)
        val_edge_pairs.append([node_to_idx[dst], node_to_idx[src]])
        val_labels.append(0)
    
    val_edge_index = torch.tensor(val_edge_pairs, dtype=torch.long).t()
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    
    # 학습 시작 전 초기화
    train_losses = []
    val_losses = []
    
    # 학습
    num_epochs = 300
    print(f"\n{weight_class} 학습 시작:")
    print("Epoch\tLoss\t\tTrain F1\tVal F1")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # 학습 단계
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, train_edge_index)
        loss = F.cross_entropy(out, train_labels)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # 검증 단계
        model.eval()
        with torch.no_grad():
            # 학습 세트 평가
            train_out = model(data.x, data.edge_index, train_edge_index)
            train_pred = train_out.argmax(dim=-1)
            train_prec, train_rec, train_f1 = calculate_metrics(
                train_labels.cpu().numpy(), 
                train_pred.cpu().numpy()
            )
            
            # 검증 세트 평가
            val_out = model(data.x, data.edge_index, val_edge_index)
            val_loss = F.cross_entropy(val_out, val_labels)
            val_pred = val_out.argmax(dim=-1)
            val_prec, val_rec, val_f1 = calculate_metrics(
                val_labels.cpu().numpy(), 
                val_pred.cpu().numpy()
            )
        
        val_losses.append(val_loss.item())
        
        # 10 에포크마다 결과 출력
        if (epoch + 1) % 10 == 0:
            print(f"{epoch+1:3d}\t{loss.item():.4f}\t{train_f1:.4f}\t{val_f1:.4f}")
    
    # 손실값 그래프 그리기
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {weight_class}')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 10)  # y축 범위를 0~10으로 설정
    plt.savefig(f'loss_plot_{weight_class}.png')
    plt.close()
    
    # Loss 결과 저장
    with open(f'loss_{weight_class}.txt', 'w') as f:
        f.write("Train Loss: " + str(train_losses) + "\n")
        f.write("Val Loss: " + str(val_losses))
    
    return model, data, node_to_idx, {
        'train_losses': train_losses,
        'val_losses': val_losses
    }

def predict_winner(model, data, fighter1, fighter2, node_to_idx):
    model.eval()
    with torch.no_grad():
        # 양방향 모두 예측
        edge_index_1_2 = torch.tensor([[node_to_idx[fighter1], node_to_idx[fighter2]]], 
                                    dtype=torch.long).t()
        edge_index_2_1 = torch.tensor([[node_to_idx[fighter2], node_to_idx[fighter1]]], 
                                    dtype=torch.long).t()
        
        # fighter1 -> fighter2 방향 예측
        out_1_2 = model(data.x, data.edge_index, edge_index_1_2)
        prob_1_2 = F.softmax(out_1_2, dim=1)
        
        # fighter2 -> fighter1 방향 예측
        out_2_1 = model(data.x, data.edge_index, edge_index_2_1)
        prob_2_1 = F.softmax(out_2_1, dim=1)
        
        # 승자 결정 (더 높은 확률을 가진 방향 선택)
        if prob_1_2[0][1] > prob_2_1[0][1]:
            return fighter1, prob_1_2[0][1].item()
        else:
            return fighter2, prob_2_1[0][1].item()

# 모델 학습
models_data = {}
for weight_class, graph_data in data_by_weight_class.items():
    if weight_class == 'Lightweight':
        print(f"\n학습 시작: {weight_class}")
        model, data, node_to_idx, metrics = train_weight_class_model(weight_class, graph_data)
        models_data[weight_class] = {
            'model': model,
            'data': data,
            'node_to_idx': node_to_idx,
            'metrics': metrics
        }

# 예측 예시
def predict_fight(fighter1, fighter2, weight_class):
    model_data = models_data[weight_class]
    winner, probability = predict_winner(
        model_data['model'],
        model_data['data'],
        fighter1,
        fighter2,
        model_data['node_to_idx']
    )
    print(f"예측 승자: {winner}")
    print(f"승리 확률: {probability:.4f}")