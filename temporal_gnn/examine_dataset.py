# large_dataset.csv 파일에서 특정 선수의 모든 경기 데이터를 추출하여 특성 변화 여부를 확인하는 코드입니다.

import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('dataset/large_dataset.csv')
events = pd.read_csv('dataset/completed_events_small.csv')

# 날짜 형식 변환
events['date'] = pd.to_datetime(events['date'])

# 특정 파이터의 모든 경기 데이터 가져오는 함수
def get_fighter_matches(df, fighter_name):
    # Red corner 경기
    red_matches = df[df['r_fighter'] == fighter_name].copy()
    red_matches['corner'] = 'red'
    
    # Blue corner 경기 
    blue_matches = df[df['b_fighter'] == fighter_name].copy()
    blue_matches['corner'] = 'blue'
    
    # 경기 데이터 합치기
    all_matches = pd.concat([red_matches, blue_matches])
    
    # events 데이터셋과 병합하여 날짜 정보 추가
    all_matches = pd.merge(
        all_matches,
        events[['event', 'date']],
        left_on='event_name',
        right_on='event',
        how='left'
    )
    
    # 날짜순 정렬
    all_matches = all_matches.sort_values('date')
    
    return all_matches

# 원하는 통계 컬럼들
# stats_columns = [
#     'kd', 'sig_str', 'sig_str_att', 'sig_str_acc',
#     'str', 'str_att', 'str_acc', 'td', 'td_att', 
#     'td_acc', 'sub_att', 'rev', 'ctrl_sec',
#     'wins_total', 'losses_total', 'age', 'height',
#     'weight', 'reach', 'stance', 'SLpM_total',
#     'SApM_total', 'sig_str_acc_total', 'td_acc_total',
#     'str_def_total', 'td_def_total', 'sub_avg', 'td_avg'
# ]


stats_columns = ['age', 'wins_total', 'losses_total']
# 예시: 특정 파이터의 경기별 통계 보기
fighter_name = "Sean O'Malley"  # 분석하고 싶은 파이터 이름
fighter_matches = get_fighter_matches(df, fighter_name)

# 선택한 통계 컬럼들의 경기별 데이터 출력
for idx, match in fighter_matches.iterrows():
    print(f"\n이벤트: {match['event_name']}")
    print(f"날짜: {match['date'].strftime('%Y-%m-%d')}")
    print(f"코너: {match['corner']}")
    print("---선수 통계---")
    for col in stats_columns:
        prefix = 'r_' if match['corner'] == 'red' else 'b_'
        value = match[f'{prefix}{col}']
        print(f"{col}: {value}")
    print("="*50)
