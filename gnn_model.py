import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from common import plot_training_history
from sklearn.metrics import f1_score

# Step 1: Load the datasets
fighter_stats_path = './preprocessed_fighter_stats.csv'
event_dataset_path = './event_dataset_2.csv'

fighter_stats = pd.read_csv(fighter_stats_path)
event_dataset = pd.read_csv(event_dataset_path)

event_dataset['date'] = pd.to_datetime(event_dataset['date'])
event_dataset = event_dataset[event_dataset['date'] >= '1994-01-01']

# Step 2: Split the dataset by weight class and sort by date
weight_classes = event_dataset['weight_class'].unique()
sorted_weight_class_datasets = {}

for weight_class in weight_classes:
    wc_data = event_dataset[event_dataset['weight_class'] == weight_class]
    wc_data = wc_data.sort_values(by='date')
    sorted_weight_class_datasets[weight_class] = wc_data

# Step 3: Function to create a dynamic graph for each weight class
def create_dynamic_graph(wc_data, fighter_stats):
    dynamic_graphs = []
    G = nx.DiGraph()  # Directed graph for each weight class

    for index, row in wc_data.iterrows():
        r_fighter = row['r_fighter']
        b_fighter = row['b_fighter']

        # Add node for r_fighter if not exists
        if r_fighter not in G:
            r_data = fighter_stats[fighter_stats['name'] == r_fighter]
            if not r_data.empty:
                attributes = r_data.iloc[0].to_dict()
                del attributes['name']  # Remove identifier column
                attributes['win'] = 0
                attributes['lose'] = 0
                G.add_node(r_fighter, **attributes)
        
        # Add node for b_fighter if not exists
        if b_fighter not in G:
            b_data = fighter_stats[fighter_stats['name'] == b_fighter]
            if not b_data.empty:
                attributes = b_data.iloc[0].to_dict()
                del attributes['name']  # Remove identifier column
                attributes['win'] = 0
                attributes['lose'] = 0
                G.add_node(b_fighter, **attributes)
                
        winner = row['winner']
        
        if r_fighter in G and b_fighter in G:
            r_fighter_node = G.nodes[r_fighter]
            b_fighter_node = G.nodes[b_fighter]
        else:
            continue
        
        if r_fighter_node == {} or b_fighter_node == {}:
            continue
        
        if winner == 'Red':
            r_fighter_node['win'] += 1
            b_fighter_node['lose'] += 1
            G.add_edge(r_fighter, b_fighter, date=row['date'])
        else:
            b_fighter_node['win'] += 1
            r_fighter_node['lose'] += 1
            G.add_edge(b_fighter, r_fighter, date=row['date'])
        
        dynamic_graphs.append(nx.DiGraph(G))
        
    return dynamic_graphs

dynamic_graphs_by_weight_class = {}

for weight_class, wc_data in sorted_weight_class_datasets.items():
    dynamic_graphs_by_weight_class[weight_class] = create_dynamic_graph(wc_data, fighter_stats)

# Dynamic graphs are now prepared for each weight class.
# Further steps for LSTM-based graph network will follow.

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
    
# # Prepare datasets for all weight classes
# datasets_by_weight_class = {}
# for weight_class, dynamic_graphs in dynamic_graphs_by_weight_class.items():
#     datasets_by_weight_class[weight_class] = DynamicGraphDataset(dynamic_graphs)

# Step 4: Define the GNN + LSTM Model
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool

class GNN_LSTM_Model(nn.Module):
    def __init__(self, node_feat_dim, gnn_hidden_dim, lstm_hidden_dim, num_layers, output_dim):
        super(GNN_LSTM_Model, self).__init__()
        # GNN 부분
        self.conv1 = GCNConv(node_feat_dim, gnn_hidden_dim)
        self.conv2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        self.relu = nn.ReLU()
        
        # LSTM 부분
        self.lstm = nn.LSTM(gnn_hidden_dim, lstm_hidden_dim, num_layers, batch_first=True)
        
        # 최종 출력층
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, graph_sequence):
        graph_embeddings = []
        for seq in graph_sequence:  # 배치의 각 시퀀스에 대해
            seq_embeddings = []
            for graph_data in seq:  # 시퀀스의 각 그래프에 대해
                adj_matrix, node_features = graph_data
                # print("adj_matrix:", adj_matrix)
                # print("node_features:", node_features)

                adj_matrix = torch.as_tensor(adj_matrix, dtype=torch.float)
                x = torch.as_tensor(node_features, dtype=torch.float)
                
                # adj_matrix에서 edge_index 추출
                edge_index = torch.nonzero(adj_matrix).t().contiguous()
                
                # GNN 처리
                x = self.conv1(x, edge_index)
                x = self.relu(x)
                x = self.conv2(x, edge_index)
                x = self.relu(x)
                
                # 글로벌 풀링
                graph_emb = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
                seq_embeddings.append(graph_emb)
            
            # 시퀀스의 모든 래프 임베딩을 스택
            seq_embeddings = torch.stack(seq_embeddings).squeeze(1)
            graph_embeddings.append(seq_embeddings)
        
        # 배치의 모든 시퀀스를 스택
        graph_embeddings = torch.stack(graph_embeddings)
        
        lstm_out, _ = self.lstm(graph_embeddings)
        lstm_out = lstm_out[:, -1, :]
        
        out = self.fc(lstm_out)
        out = self.sigmoid(out)
        print("out:", out)
        return out

# Step 5: Modify DynamicGraphDataset to Work with Graph Neural Networks
class DynamicGraphDatasetForGNN(Dataset):
    def __init__(self, dynamic_graphs, event_data):
        self.graph_sequences = self.process_graphs(dynamic_graphs)
        self.event_data = event_data  # 이벤트 데이터 저장
        
    def process_graphs(self, dynamic_graphs):
        sequences = []
        for i, graph in enumerate(dynamic_graphs):
            # 노드 특성 추출
            node_features = []
            node_names = []  # 노드 이름 순서 저장
            for node in graph.nodes():
                node_names.append(node)
                features = [
                    float(graph.nodes[node].get('height', 0)),
                    float(graph.nodes[node].get('weight', 0)), 
                    float(graph.nodes[node].get('reach', 0)),
                    float(graph.nodes[node].get('stance', 0)),
                    float(graph.nodes[node].get('win', 0)),
                    float(graph.nodes[node].get('lose', 0))
                ]
                node_features.append(features)
            
            adj_matrix = nx.adjacency_matrix(graph).todense()
            node_features = np.array(node_features)
            
            # 이벤트 정보도 함께 저장
            event_info = {
                'node_names': node_names,  # 노드 순서 저장
                'adj_matrix': adj_matrix,
                'node_features': node_features,
                'event_idx': i  # 이벤트 인덱스 저장
            }
            sequences.append(event_info)
            
        return sequences
    
    def __len__(self):
        return len(self.graph_sequences) - 1
    
    def __getitem__(self, idx):
        # 현재 시퀀스와 다음 이벤트 정보 가져오기
        sequence = self.graph_sequences[:idx+1]
        next_event = self.event_data.iloc[idx+1]
        
        # 다음 경기의 파이터 정보
        r_fighter = next_event['r_fighter']
        b_fighter = next_event['b_fighter']
        winner = next_event['winner']  # 'Red' 또는 'Blue'
        
        # 레이블 생성 (승자가 Red면 1, Blue면 0)
        label = torch.tensor([1.0 if winner == 'Red' else 0.0], dtype=torch.float)
        
        # 파이터 정보도 함께 반환
        fight_info = {
            'r_fighter': r_fighter,
            'b_fighter': b_fighter,
            'winner': winner
        }
        
        return sequence, label, fight_info

# Prepare datasets for all weight classes with labels
labeled_datasets_by_weight_class = {}
for weight_class, dynamic_graphs in dynamic_graphs_by_weight_class.items():
    wc_data = sorted_weight_class_datasets[weight_class]  # 해당 체급의 이벤트 데이터
    labeled_datasets_by_weight_class[weight_class] = DynamicGraphDatasetForGNN(dynamic_graphs, wc_data)

# Step 6: Training Loop
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for sequences, labels, fight_info in dataloader:
        labels = labels.to(device)
        
        processed_sequences = []
        for seq in sequences:
            processed_seq = []
            for graph_data in seq:
                # graph_data는 딕셔너리 형태이므로 필요한 데이터를 직접 접근
                adj_matrix = graph_data['adj_matrix']
                node_features = graph_data['node_features']
                adj_matrix = torch.tensor(adj_matrix, dtype=torch.float).to(device)
                node_features = torch.tensor(node_features, dtype=torch.float).to(device)
                processed_seq.append((adj_matrix, node_features))
            processed_sequences.append(processed_seq)
        
        optimizer.zero_grad()
        outputs = model(processed_sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    fight_results = []  # 경기 결과 저장
    
    with torch.no_grad():
        for sequences, labels, fight_info in dataloader:
            # 시퀀스와 레이블을 디바이스로 이동
            labels = labels.to(device)
            
            # 시퀀스의 각 그래프 데이터를 디바이스로 이동
            processed_sequences = []
            for seq in sequences:
                processed_seq = []
                for graph_data in seq:
                    # graph_data는 딕셔너리 형태이므로 필요한 데이터를 직접 접근
                    adj_matrix = graph_data['adj_matrix']
                    node_features = graph_data['node_features']
                    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float).to(device)
                    node_features = torch.tensor(node_features, dtype=torch.float).to(device)
                    processed_seq.append((adj_matrix, node_features))
                processed_sequences.append(processed_seq)
            
            outputs = model(processed_sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            print(outputs)
            # 예측값과 실제값 저장
            predictions = (outputs > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 경기 결과 저장
            for i in range(len(predictions)):
                fight_results.append({
                    'r_fighter': fight_info[i]['r_fighter'],
                    'b_fighter': fight_info[i]['b_fighter'],
                    'predicted': 'Red' if predictions[i] > 0.5 else 'Blue',
                    'actual': fight_info[i]['winner']
                })
    
    # F1 스코어 계산
    f1 = f1_score(all_labels, all_predictions, average='binary')
    
    return total_loss / len(dataloader), f1, fight_results

def custom_collate(batch):
    # 배치 내에서 가장 긴 시퀀스의 길이를 찾습니다
    max_seq_len = max(len(sequences) for sequences, _, _ in batch)
    
    # 모든 시퀀스를 동일한 길이로 패딩합니다
    padded_sequences = []
    labels = []
    fight_infos = []
    
    for sequences, label, info in batch:
        # 현재 시퀀스가 최대 길이보다 짧으면 마지막 그래프로 패딩
        if len(sequences) < max_seq_len:
            padding = [sequences[-1]] * (max_seq_len - len(sequences))
            sequences = sequences + padding
        padded_sequences.append(sequences)
        labels.append(label)
        fight_infos.append(info)
    
    # 모이블을 스택
    labels_tensor = torch.stack(labels)
    
    return padded_sequences, labels_tensor, fight_infos

# 데이터셋 분할 함수
def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # 순차적으로 데이터셋 분할
    indices = list(range(len(dataset)))
    train_dataset = torch.utils.data.Subset(dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(dataset, indices[train_size:train_size+val_size])
    test_dataset = torch.utils.data.Subset(dataset, indices[train_size+val_size:])
    
    return train_dataset, val_dataset, test_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 손실값을 저장할 리스트 추가
train_losses = []
val_losses = []

# 각 체급별 데이터셋 분할 및 학습
for weight_class, dataset in labeled_datasets_by_weight_class.items():
    #if weight_class != 'Middleweight':
    #    continue
    print(f'\n{weight_class} 데이터 처리 시작')
    print(f'전체 데이터 크기: {len(dataset)}')
    
    # 데이터셋 분할
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    print(f'학습 데이터: {len(train_dataset)}, 검증 데이터: {len(val_dataset)}, 평가 데이터: {len(test_dataset)}')
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, 
                            batch_size=32, 
                            shuffle=False,
                            collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, 
                          batch_size=32, 
                          shuffle=False,
                          collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, 
                           batch_size=32, 
                           shuffle=False,
                           collate_fn=custom_collate)
    
    print(f'데이터 로더 생성 완료')
    
    # 모델 초기화
    node_feat_dim = dataset.graph_sequences[0]['node_features'].shape[1]
    model = GNN_LSTM_Model(node_feat_dim=node_feat_dim,
                          gnn_hidden_dim=64,
                          lstm_hidden_dim=128,
                          num_layers=2,
                          output_dim=1).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 조기 종료를 위한 변수들
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 100
    
    # 학습 루프
    epochs = 100
    for epoch in range(epochs):
        # 학습
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        # 검증
        val_loss, val_f1, val_fight_results = evaluate_model(model, val_loader, criterion, device)
        
        # 조실값 저장
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 조기 종료 로직
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 최고 성능 모델 저장
            torch.save(model.state_dict(), f'best_model_{weight_class}.pth')
        else:
            patience_counter += 1
        
        #if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
        
        # 조기 종료
        # if patience_counter >= patience:
        #     print(f'Early stopping at epoch {epoch+1}')
        #     break
    
    # 학습이 끝난 후 그래프 그리기
    plot_training_history(train_losses, val_losses, weight_class)
    
    # 테스트 세트에서 최종 평가
    model.load_state_dict(torch.load(f'best_model_{weight_class}.pth'))
    test_loss, test_f1, test_fight_results = evaluate_model(model, test_loader, criterion, device)
    print(f'\n{weight_class} 최종 테스트 손실: {test_loss:.4f}, Test F1: {test_f1:.4f}')

# Step 9: Prediction Function
def predict_outcome(model, current_sequence, device):
    model.eval()
    with torch.no_grad():
        sequence = [torch.tensor(s, dtype=torch.float).to(device) for s in current_sequence]
        output = model(sequence)
        prediction = (output > 0.5).float()
    return prediction

# 예시 사용법
# new_event_sequence = [...]  # 새로운 이벤트까지의 그래프 시퀀스
# outcome = predict_outcome(model, new_event_sequence, device)
# print(f'예측된 승부 결과: {outcome}')