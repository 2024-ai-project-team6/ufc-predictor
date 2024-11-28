import pandas as pd

# 두 데이터셋 읽기
event_df = pd.read_csv('event_dataset.csv')
large_df = pd.read_csv('dataset/large_dataset.csv')

# 더 정확한 매칭을 위한 키 생성 (날짜, 체급, 두 파이터 이름 모두 포함)
def create_match_key(row):
    fighters = sorted([row['r_fighter'], row['b_fighter']])  # 파이터 이름을 정렬하여 순서 상관없이 매칭
    return f"{row['date']}_{row['weight_class']}_{fighters[0]}_{fighters[1]}"

event_df['match_key'] = event_df.apply(create_match_key, axis=1)

# large_dataset의 매칭 키 생성
def create_large_match_key(row):
    date = row['event_name'].split(':')[0].strip()  # 이벤트 이름에서 날짜 추출 필요
    fighters = sorted([row['r_fighter'], row['b_fighter']])
    return f"{date}_{row['weight_class']}_{fighters[0]}_{fighters[1]}"

large_df['match_key'] = large_df.apply(create_large_match_key, axis=1)

# 매칭하여 업데이트
for idx, row in event_df.iterrows():
    match_key = row['match_key']
    matching_row = large_df[large_df['match_key'] == match_key]
    
    if not matching_row.empty:
        event_df.at[idx, 'r_fighter'] = matching_row['r_fighter'].iloc[0]
        event_df.at[idx, 'b_fighter'] = matching_row['b_fighter'].iloc[0]
        event_df.at[idx, 'winner'] = matching_row['winner'].iloc[0]

# 결과 저장
event_df = event_df[['date', 'r_fighter', 'b_fighter', 'winner', 'weight_class']]
event_df.to_csv('event_dataset.csv', index=False)