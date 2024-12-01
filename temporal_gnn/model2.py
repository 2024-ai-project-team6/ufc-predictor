# Fighter Stats의 모든 컬럼을 사용하는 GNN 모델입니다.

import torch
from torch_geometric.nn import GCNConv
from torch.nn import functional as F
import pandas as pd
from collections import OrderedDict
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
import os
import argparse

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
    fighter_stats_path = 'dataset/preprocessed/fighter_stats.csv'
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

def evaluate_model(model, node_features, validation_edge_index, validation_edge_labels, 
                  test_edge_index, test_edge_labels):
    with torch.no_grad():
        # 검증 데이터로 최적 임계값 찾기
        val_predictions = model(node_features, validation_edge_index)
        val_loss = F.binary_cross_entropy(val_predictions.squeeze(), validation_edge_labels)
        
        # ROC 커브를 통한 최적 임계값 찾기
        fpr, tpr, thresholds = roc_curve(validation_edge_labels.numpy(), 
                                       val_predictions.squeeze().numpy())
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Precision-Recall 커브를 통한 최적 임계값 찾기
        precisions, recalls, pr_thresholds = precision_recall_curve(
            validation_edge_labels.numpy(), val_predictions.squeeze().numpy())
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_pr_idx = np.argmax(f1_scores)
        optimal_pr_threshold = pr_thresholds[optimal_pr_idx-1] if optimal_pr_idx > 0 else 0
        
        # 두 방법으로 구한 임계값의 평균 사용
        final_threshold = (optimal_threshold + optimal_pr_threshold) / 2
        print(f'Validation으로 찾은 Optimal Threshold: {final_threshold:.4f}')
        
        # 테스트 데이터에 최적 임계값 적용
        test_predictions = model(node_features, test_edge_index)
        test_loss = F.binary_cross_entropy(test_predictions.squeeze(), test_edge_labels)
        pred_labels = (test_predictions.squeeze() > final_threshold).float()
        
        print(f'예측값: {pred_labels.numpy()}')
        print(f'실제값: {test_edge_labels.numpy()}')
        
        print(f'Validation Loss: {val_loss.item():.4f}')
        print(f'Test Loss: {test_loss.item():.4f}')
        # F1 점수 계산 시 예측값을 이진값으로 변환
        val_pred_binary = (val_predictions.squeeze() > final_threshold).float().numpy()
        test_pred_binary = (test_predictions.squeeze() > final_threshold).float().numpy()
        
        # f1 스코어 계산 (이진값 사용)
        f1_val = f1_score(validation_edge_labels.numpy(), val_pred_binary)
        f1 = f1_score(test_edge_labels.numpy(), test_pred_binary)
        print(f'Validation F1 Score: {f1_val:.4f}')
        print(f'Test F1 Score: {f1:.4f}')

def parse_args():
    parser = argparse.ArgumentParser(description='GNN Model Training and Evaluation')
    parser.add_argument('--evaluate', type=str, help='Path to the model to evaluate')
    parser.add_argument('--weight_class', type=str, required=True, help='Weight class to train/evaluate',
                      choices=['Lightweight', 'Welterweight', 'Middleweight', 
                              'Light Heavyweight', 'Heavyweight', 'Women\'s Strawweight',
                              'Women\'s Flyweight', 'Women\'s Bantamweight'])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    node_features, edge_indices, edge_labels, weight_classes = load_data()
    
    if args.evaluate:
        print(f"모델 평가 모드: {args.evaluate}")
        weight_class = args.weight_class
        print(f"체급: {weight_class}")
        
        # 데이터 준비
        train_size = int(len(edge_labels[weight_class]) * 0.6)
        validation_size = int(len(edge_labels[weight_class]) * 0.2)
        test_size = int(len(edge_labels[weight_class]) * 0.2)
        
        print(f"train_size: {train_size}")
        print(f"validation_size: {validation_size}")
        print(f"test_size: {test_size}")
        
        validation_edge_index = edge_indices[weight_class][:, train_size:train_size+validation_size]
        validation_edge_labels = edge_labels[weight_class][train_size:train_size+validation_size]
        
        test_edge_index = edge_indices[weight_class][:, train_size+validation_size:train_size+validation_size+test_size]
        test_edge_labels = edge_labels[weight_class][train_size+validation_size:train_size+validation_size+test_size]
        
        # 모델 불러오기
        model = MyGNN(in_channels=node_features[weight_class].shape[1], hidden_channels=16)
        model.load_state_dict(torch.load(f'temporal_gnn/models/{weight_class}/{args.evaluate}'))
        model.eval()
        
        # 모델 평가
        print("\n=== 모델 평가 결과 ===")
        evaluate_model(model, node_features[weight_class], validation_edge_index, 
                      validation_edge_labels, test_edge_index, test_edge_labels)
    else:
        # 학습 모드
        weight_classes = [args.weight_class]
            
        for weight_class in weight_classes:
            print(f'============{weight_class}============')
            # print("node_features", node_features[weight_class].shape)
            # print("edge_indices", edge_indices[weight_class].shape)
            # print("edge_labels", edge_labels[weight_class].shape)
            
            # 경기 수가 10 미만인 경우 패스
            if len(edge_labels[weight_class]) < 10:
                print(f"{weight_class}의 경기 수가 너무 적습니다. 건너뜁니다.")
                continue
            
            train_size = int(len(edge_labels[weight_class]) * 0.6)
            validation_size = int(len(edge_labels[weight_class]) * 0.2)
            test_size = int(len(edge_labels[weight_class]) * 0.2)
            
            # edge_indices와 edge_labels를 train, validation, test로 나누기
            train_edge_index = edge_indices[weight_class][:, :train_size]
            train_edge_labels = edge_labels[weight_class][:train_size]
            
            validation_edge_index = edge_indices[weight_class][:, train_size:train_size+validation_size]
            validation_edge_labels = edge_labels[weight_class][train_size:train_size+validation_size]
            
            test_edge_index = edge_indices[weight_class][:, train_size+validation_size:train_size+validation_size+test_size]
            test_edge_labels = edge_labels[weight_class][train_size+validation_size:train_size+validation_size+test_size]
            
            # print("train_edge_index", train_edge_index.shape)
            # print("train_edge_labels", train_edge_labels.shape)
            # print("validation_edge_index", validation_edge_index.shape)
            # print("validation_edge_labels", validation_edge_labels.shape)
            # print("test_edge_index", test_edge_index.shape)
            # print("test_edge_labels", test_edge_labels.shape)
            
            # 모델 초기화
            model = MyGNN(in_channels=node_features[weight_class].shape[1], hidden_channels=16)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # 학습 시작 전 초기화
            train_losses = []
            val_losses = []
            
            # 모델 저장 디렉토리 생성
            save_dir = f'temporal_gnn/models/{weight_class}'
            os.makedirs(save_dir, exist_ok=True)
            
            best_val_f1 = 0
            best_val_epoch = 0
            best_test_f1_at_best_val = 0
            best_test_f1 = 0
            best_test_epoch = 0
            best_val_f1_at_best_test = 0
            
            # 학습 시작 전 초기화에 추가
            best_val_loss = float('inf')
            best_val_loss_epoch = 0
            test_f1_at_best_val_loss = 0
            val_f1_at_best_val_loss = 0
            
            # 학습
            for epoch in range(300):
                model.train()
                optimizer.zero_grad()
                
                predictions = model(node_features[weight_class], train_edge_index)
                loss = F.binary_cross_entropy(predictions.squeeze(), train_edge_labels)
                
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    # 모델 저장
                    model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
                    torch.save(model.state_dict(), model_path)
                    
                    # 평가 모드로 전환
                    model.eval()
                    with torch.no_grad():
                        # Validation 데이터로 최적 임계값 찾기
                        val_predictions = model(node_features[weight_class], validation_edge_index)
                        val_loss = F.binary_cross_entropy(val_predictions.squeeze(), validation_edge_labels)
                        
                        # ROC 커브와 PR 커브로 최적 임계값 찾기
                        fpr, tpr, thresholds = roc_curve(validation_edge_labels.numpy(), 
                                                       val_predictions.squeeze().numpy())
                        optimal_idx = np.argmax(tpr - fpr)
                        optimal_threshold = thresholds[optimal_idx]
                        
                        precisions, recalls, pr_thresholds = precision_recall_curve(
                            validation_edge_labels.numpy(), val_predictions.squeeze().numpy())
                        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
                        optimal_pr_idx = np.argmax(f1_scores)
                        optimal_pr_threshold = pr_thresholds[optimal_pr_idx-1] if optimal_pr_idx > 0 else 0
                        
                        final_threshold = (optimal_threshold + optimal_pr_threshold) / 2
                        
                        # Validation F1 Score 계산
                        val_pred_labels = (val_predictions.squeeze() > final_threshold).float()
                        val_f1 = f1_score(validation_edge_labels.numpy(), val_pred_labels.numpy())
                        
                        # Test F1 Score 계산
                        test_predictions = model(node_features[weight_class], test_edge_index)
                        test_pred_labels = (test_predictions.squeeze() > final_threshold).float()
                        test_f1 = f1_score(test_edge_labels.numpy(), test_pred_labels.numpy())
                        
                        # 최고 성능 업데이트
                        if val_f1 > best_val_f1:
                            best_val_f1 = val_f1
                            best_val_epoch = epoch + 1
                            best_test_f1_at_best_val = test_f1
                        
                        if test_f1 > best_test_f1:
                            best_test_f1 = test_f1
                            best_test_epoch = epoch + 1
                            best_val_f1_at_best_test = val_f1
                            
                        # validation loss 기준 최고 성능 업데이트
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_val_loss_epoch = epoch + 1
                            test_f1_at_best_val_loss = test_f1
                            val_f1_at_best_val_loss = val_f1
                        
                        print(f'\nEpoch {epoch+1}:')
                        print(f'Threshold: {final_threshold:.4f}')
                        print(f'Train Loss: {loss.item():.4f}')
                        print(f'Validation Loss: {val_loss.item():.4f}')
            
            # 학습 완료 후 최종 결과 출력
            print("\n=== 학습 최종 결과 ===")
            print(f"Best Validation Loss: {best_val_loss:.4f} (Epoch {best_val_loss_epoch})")
            print(f"- At Best Val Loss - Val F1: {val_f1_at_best_val_loss:.4f}, Test F1: {test_f1_at_best_val_loss:.4f}")
            print("")
            print(f"Best Validation F1 Score: {best_val_f1:.4f} (Epoch {best_val_epoch}, Test F1 Score: {best_test_f1_at_best_val:.4f})")
            print(f"Best Test F1 Score: {best_test_f1:.4f} (Epoch {best_test_epoch}, Validation F1 Score: {best_val_f1_at_best_test:.4f})")