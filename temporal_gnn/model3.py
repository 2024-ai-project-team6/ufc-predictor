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
import json

THRESHOLD = 0.5
DROP_RATE = 0.2
EPOCHS = 500
LEARNING_RATE = 0.01
HIDDEN_CHANNELS = 16


class MyGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, hidden_channels * 4)
        self.conv4 = GCNConv(hidden_channels * 4, hidden_channels * 8)
        self.conv5 = GCNConv(hidden_channels * 8, hidden_channels * 16)
        
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * 16 * hidden_channels, hidden_channels*16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels*16, hidden_channels*8),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels*8, hidden_channels*4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels*4, hidden_channels*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels*2, 1)
        )
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROP_RATE, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROP_RATE, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROP_RATE, training=self.training)
        
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROP_RATE, training=self.training)
        
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROP_RATE, training=self.training)
        
        source_node = x[edge_index[0]]
        target_node = x[edge_index[1]]
        
        edge_features = torch.cat([source_node, target_node], dim=1)
        edge_predictions = self.edge_mlp(edge_features)
        edge_predictions = torch.sigmoid(edge_predictions)
        return edge_predictions
    
def load_data():
    fighter_stats_path = 'dataset/preprocessed/fighter_stats.csv'
    event_dataset_path = 'dataset/preprocessed/event_list_shuffled.csv'

    fighter_stats = pd.read_csv(fighter_stats_path)
    event_dataset = pd.read_csv(event_dataset_path)
    
    # 날짜를 datetime 형식으로 변환
    event_dataset['date'] = pd.to_datetime(event_dataset['date'])
    
    # 2001년 이상의 데이터만 필터링
    event_dataset = event_dataset[event_dataset['date'].dt.year >= 2001]
    
    # 날짜 기준으로 정렬
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
    edge_counts = {}  # 각 파이터의 엣지 수를 저장할 딕셔너리 추가
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
        
        # 각 파이터의 엣지 수를 저장할 딕셔너리 초기화
        fighter_edge_count = {name: 0 for name in fighters}
        
        for _, fight in wc_data.iterrows():
            r_fighter = fight['r_fighter']
            b_fighter = fight['b_fighter']
            winner = fight['winner']
            
            sources.append(fighter_to_idx[r_fighter])
            targets.append(fighter_to_idx[b_fighter])
            labels.append(1 if winner == "Red" else 0)
        
        # train_data에 대해서만 엣지 수 증가
        train_size = int(len(wc_data) * 0.6)
        train_data = wc_data.iloc[:train_size]
        for _, fight in train_data.iterrows():
            r_fighter = fight['r_fighter']
            b_fighter = fight['b_fighter']
            
            # 각 파이터의 엣지 수 증가
            fighter_edge_count[r_fighter] += 1
            fighter_edge_count[b_fighter] += 1
        
        # 엣지 수를 저장
        edge_counts[weight_class] = fighter_edge_count
        
        # 리스트를 텐서로 변환
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_indices[weight_class] = edge_index
        
        labels = torch.tensor(labels, dtype=torch.float)
        edge_labels[weight_class] = labels
    
    return node_features, edge_indices, edge_labels, weight_classes, sorted_weight_class_datasets, edge_counts

def evaluate_model(model, node_features, validation_edge_index, validation_edge_labels, 
                  test_edge_index, test_edge_labels, raw_data, edge_counts):
    with torch.no_grad():
        # 검증 데이터로 최적 임계값 찾기
        val_predictions = model(node_features, validation_edge_index)
        val_loss = F.binary_cross_entropy(val_predictions.squeeze(), validation_edge_labels)
        
        # ROC 커브를 통한 최적 임계값 찾기
        fpr, tpr, thresholds = roc_curve(validation_edge_labels.numpy(), 
                                       val_predictions.squeeze().numpy())
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        
        # 테스트 데이터에 최적 임계값 적용
        test_predictions = model(node_features, test_edge_index)
        test_loss = F.binary_cross_entropy(test_predictions.squeeze(), test_edge_labels)
        pred_labels = (test_predictions.squeeze() > THRESHOLD).float()
        
        results = []
        for i, (pred, label) in enumerate(zip(pred_labels.numpy(), test_edge_labels.numpy())):
            result = {
                'date': str(raw_data.iloc[i]['date']),
                'r_fighter': raw_data.iloc[i]['r_fighter'],
                'b_fighter': raw_data.iloc[i]['b_fighter'],
                'r_edge_count': edge_counts[raw_data.iloc[i]['r_fighter']],
                'b_edge_count': edge_counts[raw_data.iloc[i]['b_fighter']],
                'success': int(pred == label)
            }
            results.append(result)
            
        # 결과를 파일로 저장
        with open(f'results/{weight_class}.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f'test_predictions: {test_predictions.squeeze()}')
        print(f'예측값: {pred_labels.numpy()}')
        print(f'실제값: {test_edge_labels.numpy()}')
        
        print(f'Validation Loss: {val_loss.item():.4f}')
        print(f'Test Loss: {test_loss.item():.4f}')
        # F1 점수 계산 시 예측값을 이진값으로 변환
        val_pred_binary = (val_predictions.squeeze() > THRESHOLD).float().numpy()
        test_pred_binary = (test_predictions.squeeze() > THRESHOLD).float().numpy()
        
        # f1 스코어 계산 (이진값 사용)
        f1_val_pos = f1_score(validation_edge_labels.numpy(), val_pred_binary)
        f1_val_neg = f1_score(validation_edge_labels.numpy(), val_pred_binary, pos_label=0)
        
        f1_test_pos = f1_score(test_edge_labels.numpy(), test_pred_binary)
        f1_test_neg = f1_score(test_edge_labels.numpy(), test_pred_binary, pos_label=0)
        
        print(f'Validation F1 Score (Positive): {f1_val_pos:.4f}')
        print(f'Validation F1 Score (Negative): {f1_val_neg:.4f}')
        print(f'Test F1 Score (Positive): {f1_test_pos:.4f}')
        print(f'Test F1 Score (Negative): {f1_test_neg:.4f}')
        
        # 검증 데이터 정확도 계산
        val_correct = (val_pred_binary == validation_edge_labels.numpy()).sum()
        val_total = len(validation_edge_labels)
        val_accuracy = val_correct / val_total
        
        # 테스트 데이터 정확도 계산
        test_correct = (test_pred_binary == test_edge_labels.numpy()).sum()
        test_total = len(test_edge_labels)
        test_accuracy = test_correct / test_total
        
        print("\n=== 예측 정확도 ===")
        print(f'Validation: {val_correct}/{val_total} 경기 예측 성공 (정확도: {val_accuracy:.4f})')
        print(f'Test: {test_correct}/{test_total} 경기 예측 성공 (정확도: {test_accuracy:.4f})')
        
        # Red/Blue 선수별 정확도 계산
        val_red_correct = ((val_pred_binary == 1) & (validation_edge_labels.numpy() == 1)).sum()
        val_red_total = (validation_edge_labels.numpy() == 1).sum()
        val_blue_correct = ((val_pred_binary == 0) & (validation_edge_labels.numpy() == 0)).sum()
        val_blue_total = (validation_edge_labels.numpy() == 0).sum()
        
        test_red_correct = ((test_pred_binary == 1) & (test_edge_labels.numpy() == 1)).sum()
        test_red_total = (test_edge_labels.numpy() == 1).sum()
        test_blue_correct = ((test_pred_binary == 0) & (test_edge_labels.numpy() == 0)).sum()
        test_blue_total = (test_edge_labels.numpy() == 0).sum()
        
        print("\n=== Red/Blue 선수별 예측 정확도 ===")
        print(f'Validation Red 승리 예측: {val_red_correct}/{val_red_total} (정확도: {val_red_correct/val_red_total:.4f})')
        print(f'Validation Blue 승리 예측: {val_blue_correct}/{val_blue_total} (정확도: {val_blue_correct/val_blue_total:.4f})')
        print(f'Test Red 승리 예측: {test_red_correct}/{test_red_total} (정확도: {test_red_correct/test_red_total:.4f})')
        print(f'Test Blue 승리 예측: {test_blue_correct}/{test_blue_total} (정확도: {test_blue_correct/test_blue_total:.4f})')
        
        print(f'Test 전체 예측 정확도: {test_correct}/{test_total} (정확도: {test_accuracy:.4f})')

def parse_args():
    parser = argparse.ArgumentParser(description='GNN Model Training and Evaluation')
    parser.add_argument('--evaluate', type=str, help='Path to the model to evaluate')
    parser.add_argument('--weight_class', type=str, required=True, help='Weight class to train/evaluate',
                      choices=['Lightweight', 'Welterweight', 'Middleweight', 
                              'Light Heavyweight', 'Heavyweight', 'Women\'s Strawweight',
                              'Women\'s Flyweight', 'Women\'s Bantamweight'])
    return parser.parse_args()

def plot_losses(train_losses, val_losses, weight_class, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Losses - {weight_class}')
    plt.legend()
    plt.grid(True)
    
    # 그래프 저장
    plot_path = os.path.join(save_dir, 'model3_loss_plot.png')
    plt.savefig(plot_path)
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    node_features, edge_indices, edge_labels, weight_classes, wc_data, edge_counts = load_data()
    
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
        print(f"test_size: {len(edge_labels[weight_class]) - train_size - validation_size}")
        
        validation_edge_index = edge_indices[weight_class][:, train_size:train_size+validation_size]
        validation_edge_labels = edge_labels[weight_class][train_size:train_size+validation_size]
        
        test_edge_index = edge_indices[weight_class][:, train_size+validation_size:]
        test_edge_labels = edge_labels[weight_class][train_size+validation_size:]
        
        # 모델 불러오기
        model = MyGNN(in_channels=node_features[weight_class].shape[1], hidden_channels=HIDDEN_CHANNELS)
        model.load_state_dict(torch.load(f'temporal_gnn/models/{weight_class}/{args.evaluate}'))
        model.eval()
        
        # 모델 평가
        print("\n=== 모델 평가 결과 ===")
        evaluate_model(model, node_features[weight_class], validation_edge_index, 
                      validation_edge_labels, test_edge_index, test_edge_labels, wc_data[weight_class][train_size+validation_size:], edge_counts[weight_class])
    else:
        # 학습 모드
        weight_classes = [args.weight_class]
            
        for weight_class in weight_classes:
            print(f'============{weight_class}============')

            # 경기 수가 10 미만인 경우 패스
            if len(edge_labels[weight_class]) < 10:
                print(f"{weight_class}의 경기 수가 너무 적습니다. 건너뜁니다.")
                continue
            
            train_size = int(len(edge_labels[weight_class]) * 0.6)
            validation_size = int(len(edge_labels[weight_class]) * 0.2)
            test_size = int(len(edge_labels[weight_class]) * 0.2)
            
            print(f"train_size: {train_size}")
            print(f"validation_size: {validation_size}")
            print(f"test_size: {test_size}")
            
            # edge_indices와 edge_labels를 train, validation, test로 나누기
            train_edge_index = edge_indices[weight_class][:, :train_size]
            train_edge_labels = edge_labels[weight_class][:train_size]
            
            validation_edge_index = edge_indices[weight_class][:, train_size:train_size+validation_size]
            validation_edge_labels = edge_labels[weight_class][train_size:train_size+validation_size]
            
            test_edge_index = edge_indices[weight_class][:, train_size+validation_size:]
            test_edge_labels = edge_labels[weight_class][train_size+validation_size:]
            
            # print("train_edge_index", train_edge_index.shape)
            # print("train_edge_labels", train_edge_labels.shape)
            # print("validation_edge_index", validation_edge_index.shape)
            # print("validation_edge_labels", validation_edge_labels.shape)
            # print("test_edge_index", test_edge_index.shape)
            # print("test_edge_labels", test_edge_labels.shape)
            
            # 모델 초기화
            model = MyGNN(in_channels=node_features[weight_class].shape[1], hidden_channels=HIDDEN_CHANNELS)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
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
            
            # 최고 정확도 추적 변수 초기화
            best_val_accuracy = 0
            best_val_accuracy_epoch = 0
            best_test_accuracy = 0
            best_test_accuracy_epoch = 0
            best_test_accuracy_at_best_val = 0
            best_val_accuracy_at_best_test = 0
            
            # 학습
            for epoch in range(EPOCHS):
                model.train()
                optimizer.zero_grad()
                
                predictions = model(node_features[weight_class], train_edge_index)
                loss = F.binary_cross_entropy(predictions.squeeze(), train_edge_labels)
                
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 1 == 0:
                    # 모델 저장
                    model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
                    torch.save(model.state_dict(), model_path)
                    
                    # 평가 모드로 전환
                    model.eval()
                    with torch.no_grad():
                        # Validation 데이터로 최적 임계값 찾기
                        val_predictions = model(node_features[weight_class], validation_edge_index)
                        val_loss = F.binary_cross_entropy(val_predictions.squeeze(), validation_edge_labels)
                        
                        # Validation F1 Score 계산
                        val_pred_labels = (val_predictions.squeeze() > THRESHOLD).float()
                        val_f1 = f1_score(validation_edge_labels.numpy(), val_pred_labels.numpy())
                        
                        # Validation 정확도 계산
                        val_correct = (val_pred_labels == validation_edge_labels).sum().item()
                        val_accuracy = val_correct / len(validation_edge_labels)
                        
                        # Test F1 Score 계산
                        test_predictions = model(node_features[weight_class], test_edge_index)
                        test_pred_labels = (test_predictions.squeeze() > THRESHOLD).float()
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
                            
                        # Test 정확도 계산
                        test_correct = (test_pred_labels == test_edge_labels).sum().item()
                        test_accuracy = test_correct / len(test_edge_labels)
                        
                        # 최고 Test 정확도 업데이트
                        if test_accuracy > best_test_accuracy:
                            best_test_accuracy = test_accuracy
                            best_test_accuracy_epoch = epoch + 1
                            best_val_accuracy_at_best_test = val_accuracy
                        
                        # print(f'\nEpoch {epoch+1}:')
                        # print(f'Train Loss: {loss.item():.4f}')
                        # print(f'Validation Loss: {val_loss.item():.4f}')
                
                        # 학습 루프 내부에서 손실값 저장 (epoch 루프 내)
                        train_losses.append(loss.item())
                        val_losses.append(val_loss.item())
                        
                        # 최고 Validation 정확도 업데이트
                        if val_accuracy > best_val_accuracy:
                            best_val_accuracy = val_accuracy
                            best_val_accuracy_epoch = epoch + 1
                            best_test_accuracy_at_best_val = test_accuracy
            
            # 학습 완료 후 그래프 그리기 (학습 루프 종료 후)
            plot_losses(train_losses, val_losses, weight_class, save_dir)
            
            # 학습 완료 후 최종 결과 출력
            print("\n=== 학습 최종 결과 ===")
            print(f"Best Validation Loss: {best_val_loss:.4f} (Epoch {best_val_loss_epoch})")
            print(f"- At Best Val Loss - Val F1: {val_f1_at_best_val_loss:.4f}, Test F1: {test_f1_at_best_val_loss:.4f}")
            print("")
            print(f"Best Validation F1 Score: {best_val_f1:.4f} (Epoch {best_val_epoch}, Test F1 Score: {best_test_f1_at_best_val:.4f})")
            print(f"Best Test F1 Score: {best_test_f1:.4f} (Epoch {best_test_epoch}, Validation F1 Score: {best_val_f1_at_best_test:.4f})")
            print(f"Best Validation Accuracy: {best_val_accuracy:.4f} (Epoch {best_val_accuracy_epoch}, Test Accuracy: {best_test_accuracy_at_best_val:.4f})")
            print(f"Best Test Accuracy: {best_test_accuracy:.4f} (Epoch {best_test_accuracy_epoch}, Validation Accuracy: {best_val_accuracy_at_best_test:.4f})")