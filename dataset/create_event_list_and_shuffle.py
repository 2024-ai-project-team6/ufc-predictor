import pandas as pd
import random

# CSV 파일 읽기
large_dataset = pd.read_csv('dataset/raw/large_dataset.csv')
completed_events = pd.read_csv('dataset/raw/completed_events_small.csv')

# completed_events에서 event_name을 인덱스로 설정
completed_events.set_index('event', inplace=True)

# 날짜를 찾고 대체
large_dataset['date'] = large_dataset['event_name'].map(completed_events['date'])

# 날짜 형식 변경
large_dataset['date'] = pd.to_datetime(large_dataset['date']).dt.strftime('%m/%d/%Y')

# 필요한 열만 선택
new_dataset = large_dataset[['date', 'r_fighter', 'b_fighter', 'winner', 'weight_class']]

new_dataset.to_csv('dataset/preprocessed/event_list.csv', index=False)

# 승리한 파이터를 r_fighter로 배치하고 winner를 Red로 설정
for index, row in new_dataset.iterrows():
    if row['winner'] == 'Blue':
        # r_fighter와 b_fighter의 값을 교환
        new_dataset.at[index, 'r_fighter'], new_dataset.at[index, 'b_fighter'] = row['b_fighter'], row['r_fighter']
        # winner를 Red로 설정
        new_dataset.at[index, 'winner'] = 'Red'
        
# 업데이트된 데이터셋 저장
new_dataset.to_csv('dataset/preprocessed/event_list_ordered.csv', index=False)

# weight_class 별로 승리 횟수 균등화를 위한 스왑
for weight_class, group in new_dataset.groupby('weight_class'):
    num_rows = len(group)
    half_rows = num_rows // 2
    
    # 현재 r_fighter가 이긴 횟수
    r_wins = group['winner'].value_counts().get('Red', 0)
    
    # r_fighter의 승리 횟수가 절반보다 많으면 스왑
    if r_wins > half_rows:
        swap_count = r_wins - half_rows
        swap_indices = random.sample(group.index.tolist(), swap_count)
        
        for index in swap_indices:
            # r_fighter와 b_fighter의 값을 교환
            new_dataset.at[index, 'r_fighter'], new_dataset.at[index, 'b_fighter'] = new_dataset.at[index, 'b_fighter'], new_dataset.at[index, 'r_fighter']
            # winner를 Blue로 설정
            new_dataset.at[index, 'winner'] = 'Blue'

new_dataset.to_csv('dataset/preprocessed/event_list_shuffled.csv', index=False)