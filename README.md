# ufc-predictor
* 현재 Goal
본 프로젝트의 목표는 UFC 경기 승부를 예측하는 딥러닝 기반 모델을 개발하여 배팅 유저 및 도박사들이 보다 정교하고 신뢰할 수 있는 예측을 할 수 있도록 지원하는 것입니다. 이를 위해 GNN을 사용하되, 선수 간의 상대 전적 데이터를 시계열로 반영할 수 있도록 LSTM을 결합하여 선수의 통계적 성과와 경기의 상황적 요소들을 모두 고려하는 정확도 높은 예측 시스템을 구축하는 것이 목표입니다.

* 구체화
1. 실시간 예측/라운드별 예측이 아닌 전체 예측만 실시할 것인가?
2. 몇라운드 승리/승리 방식 예측 제외한 단순 승패예측만 실시할 것인가?

* 데이터셋
기존 - https://www.kaggle.com/datasets/maksbasher/ufc-complete-dataset-all-events-1996-2024
대체 - https://www.kaggle.com/datasets/calmdownkarm/ufcdataset
