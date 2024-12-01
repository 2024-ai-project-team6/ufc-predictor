import torch
from torch_geometric.nn import GCNConv
from torch.nn import functional as F
import pandas as pd
from collections import OrderedDict
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

class MyGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        source_node = x[edge_index[0]]
        target_node = x[edge_index[1]]
        
        edge_features = torch.cat([source_node, target_node], dim=1)
        edge_predictions = self.edge_mlp(edge_features)
        edge_predictions = torch.sigmoid(edge_predictions)
        return edge_predictions
    
def load_data():
    fighter_stats_path = './preprocessed_fighter_stats.csv'
    event_dataset_path = './event_dataset_2.csv'

    fighter_stats = pd.read_csv(fighter_stats_path)
    event_dataset = pd.read_csv(event_dataset_path)
    
    event_dataset['date'] = pd.to_datetime(event_dataset['date'])
    
    event_dataset = event_dataset.sort_values(by='date')
    
    weight_classes = event_dataset['weight_class'].unique()
    sorted_weight_class_datasets = OrderedDict()

    for weight_class in sorted(weight_classes):
        wc_data = event_dataset[event_dataset['weight_class'] == weight_class]
        wc_data = wc_data.sort_values(by='date')
        sorted_weight_class_datasets[weight_class] = wc_data
        
    node_features_dict = OrderedDict()
    # 각 weight class 별로 노드를 구성
    for weight_class, wc_data in sorted_weight_class_datasets.items():
        # 해당 체급의 모든 선수 목록 추출
        fighters = pd.concat([wc_data['r_fighter'], wc_data['b_fighter']]).unique()
        
        # 각 선수별 특징 저장을 위한 딕셔너리
        weight_class_features = OrderedDict()
        
        for fighter in fighters:
            # fighter_stats에서 해당 선수의 특징 찾기
            fighter_data = fighter_stats[fighter_stats['name'] == fighter]
            
            if len(fighter_data) > 0:
                # 선수 데이터가 있는 경우, 필요한 특징들을 선택
                features = fighter_data.iloc[0].drop(['name']).values  # name 컬럼을 제외한 모든 특징
                weight_class_features[fighter] = features
            else:
                # 선수 데이터가 없는 경우, 0으로 채운 특징 벡터 생성
                features = np.zeros(len(fighter_stats.columns) - 1)  # name 컬럼을 제외한 크기
                weight_class_features[fighter] = features
        
        # 체급별 선수 특징을 전체 노드 특징 딕셔너리에 저장
        node_features_dict[weight_class] = weight_class_features
    
    node_features = {}
    edge_indices = {}  # 체급별 edge_index를 저장할 딕셔너리
    edge_labels = {}
    for weight_class, wc_data in sorted_weight_class_datasets.items():
        # 해당 체급의 모든 선수 목록
        fighters = list(node_features_dict[weight_class].keys())
        
        # node_features
        features_list = list(node_features_dict[weight_class].values())
        features_array = np.array(features_list, dtype=np.float32)
        fighters_feature = torch.tensor(features_array)
        # 선수 이름을 인덱스로 매핑하는 딕셔너리 생성
        fighter_to_idx = {name: idx for idx, name in enumerate(fighters)}
        
        node_features[weight_class] = fighters_feature
        
        # edge_index를 저장할 리스트 (source_nodes, target_nodes)
        sources = [] # red
        targets = [] # blue
        
        # 경기 결과 저장 (1: red 승리, 0: blue 승리)
        labels = []
        
        # 각 경기에 대해 edge 생성
        for _, fight in wc_data.iterrows():
            r_fighter = fight['r_fighter']
            b_fighter = fight['b_fighter']
            winner = fight['winner']
            
            sources.append(fighter_to_idx[r_fighter])
            targets.append(fighter_to_idx[b_fighter])
            labels.append(1 if winner == "Red" else 0)
        
        # 리스트를 텐서로 변환
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_indices[weight_class] = edge_index
        
        labels = torch.tensor(labels, dtype=torch.float)
        edge_labels[weight_class] = labels
    
    return node_features, edge_indices, edge_labels, weight_classes

node_features, edge_indices, edge_labels, weight_classes = load_data()

print(weight_classes)

for weight_class in weight_classes:
    if weight_class != 'Light Heavyweight':
        continue
    print(f'============{weight_class}============')
    print("node_features", node_features[weight_class].shape)
    print("edge_indices", edge_indices[weight_class].shape)
    print("edge_labels", edge_labels[weight_class].shape)
    
    # 경기 수가 10 미만인 경우 패스
    if len(edge_labels[weight_class]) < 10:
        continue
    
    train_size = int(len(edge_labels[weight_class]) * 0.5)
    validation_size = int(len(edge_labels[weight_class]) * 0.25)
    test_size = int(len(edge_labels[weight_class]) * 0.25)
    
    # edge_indices와 edge_labels를 train, validation, test로 나누기
    train_edge_index = edge_indices[weight_class][:, :train_size]
    train_edge_labels = edge_labels[weight_class][:train_size]
    
    validation_edge_index = edge_indices[weight_class][:, train_size:train_size+validation_size]
    validation_edge_labels = edge_labels[weight_class][train_size:train_size+validation_size]
    
    test_edge_index = edge_indices[weight_class][:, train_size+validation_size:train_size+validation_size+test_size]
    test_edge_labels = edge_labels[weight_class][train_size+validation_size:train_size+validation_size+test_size]
    
    print("train_edge_index", train_edge_index.shape)
    print("train_edge_labels", train_edge_labels.shape)
    print("validation_edge_index", validation_edge_index.shape)
    print("validation_edge_labels", validation_edge_labels.shape)
    print("test_edge_index", test_edge_index.shape)
    print("test_edge_labels", test_edge_labels.shape)
    
    # 모델 초기화
    model = MyGNN(in_channels=node_features[weight_class].shape[1], hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    # 학습 과정을 추적하기 위한 리스트 추가
    train_losses = []
    val_losses = []
    
    # 학습
    for epoch in range(2000):
        model.train()
        
        optimizer.zero_grad()
        
        predictions = model(node_features[weight_class], train_edge_index)
        loss = F.binary_cross_entropy(predictions.squeeze(), train_edge_labels)
        
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_predictions = model(node_features[weight_class], validation_edge_index)
            val_loss = F.binary_cross_entropy(val_predictions.squeeze(), validation_edge_labels)
            
        # loss 값들을 리스트에 저장
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    # 학습이 끝난 후 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {weight_class}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'temporal_gnn/loss_plot_{weight_class}.png')
    plt.close()
    
    print(f'============{weight_class}============')
    print(f'최종 Loss: {loss.item():.4f}')
    # print(f'예측값: {predictions.squeeze().detach().numpy()}')
    # print(f'실제값: {edge_labels[weight_class][:train_size].detach().numpy()}')
    
    # 평가
    model.eval()
    with torch.no_grad():
        final_predictions = model(node_features[weight_class], test_edge_index)
        final_loss = F.binary_cross_entropy(final_predictions.squeeze(), test_edge_labels)
        
        # 예측값을 0과 1로 이진화
        pred_labels = (final_predictions.squeeze() > 0.5).float()
        
        train_predictions = model(node_features[weight_class], train_edge_index)
        train_pred_labels = (train_predictions.squeeze() > 0.5).float()
        
        # f1 스코어 계산
        f1 = f1_score(test_edge_labels.numpy(), pred_labels.numpy())
        train_f1 = f1_score(train_edge_labels.numpy(), train_pred_labels.numpy())
        print(f'테스트 Loss: {final_loss.item():.4f}')
        print(f'테스트 F1 Score: {f1:.4f}')
        print(f'훈련 F1 Score: {train_f1:.4f}')