import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weight_class', type=str, default='Lightweight')
args = parser.parse_args()

# CSV 파일 읽기
df_large = pd.read_csv('dataset/raw/large_dataset.csv')
df_events = pd.read_csv('dataset/raw/completed_events_small.csv')

# JSON 파일 읽기
with open(f'results/{args.weight_class}.json', 'r') as file:
    lightweight_data = json.load(file)

# JSON 데이터를 DataFrame으로 변환
df_lightweight_json = pd.DataFrame(lightweight_data)

# 이벤트 이름과 날짜 매핑 생성
df_events['date'] = pd.to_datetime(df_events['date'], format='%B %d, %Y')
event_date_mapping = df_events.set_index('event')['date'].to_dict()

# large_dataset의 event_name을 date로 변환
df_large['date'] = df_large['event_name'].map(event_date_mapping)

# 날짜 형식 통일
df_large['date'] = df_large['date'].dt.strftime('%Y-%m-%d')
df_lightweight_json['date'] = pd.to_datetime(df_lightweight_json['date']).dt.strftime('%Y-%m-%d')

# 'Lightweight' 클래스만 필터링
df_lightweight = df_large[df_large['weight_class'] == args.weight_class]

# 두 데이터프레임을 결합
# r_fighter와 b_fighter의 순서가 바뀌었을 수 있으므로 두 가지 경우를 고려
merged_df = pd.merge(df_lightweight, df_lightweight_json, 
                     left_on=['date', 'r_fighter', 'b_fighter'], 
                     right_on=['date', 'r_fighter', 'b_fighter'], 
                     how='inner')

# r_fighter와 b_fighter의 순서를 바꿔서 다시 결합
merged_df_reversed = pd.merge(df_lightweight, df_lightweight_json, 
                              left_on=['date', 'b_fighter', 'r_fighter'], 
                              right_on=['date', 'r_fighter', 'b_fighter'], 
                              how='inner')

# r_fighter와 b_fighter의 순서를 df_lightweight에 맞게 조정
merged_df_reversed = merged_df_reversed.rename(columns={'r_fighter_x': 'r_fighter', 'b_fighter_x': 'b_fighter'})

# 두 결과를 합침
final_merged_df = pd.concat([merged_df, merged_df_reversed]).drop_duplicates()

# date 순으로 정렬
final_merged_df = final_merged_df.sort_values(by='date')

final_merged_df = final_merged_df.drop(columns=['r_fighter_y', 'b_fighter_y'])

# 결과 저장
final_merged_df.to_csv(f'results/{args.weight_class}_results.csv', index=False)