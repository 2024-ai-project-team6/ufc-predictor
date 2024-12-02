import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
file_path = './large_dataset.csv'  # 파일 경로
large_dataset = pd.read_csv(file_path)

# r_와 b_ 컬럼을 페어로 지정
paired_columns = [
    ('r_kd', 'b_kd'), ('r_sig_str', 'b_sig_str'), ('r_sig_str_att', 'b_sig_str_att'),
    ('r_sig_str_acc', 'b_sig_str_acc'), ('r_str', 'b_str'), ('r_str_att', 'b_str_att'),
    ('r_str_acc', 'b_str_acc'), ('r_td', 'b_td'), ('r_td_att', 'b_td_att'), 
    ('r_td_acc', 'b_td_acc'), ('r_sub_att', 'b_sub_att'), ('r_rev', 'b_rev'), 
    ('r_ctrl_sec', 'b_ctrl_sec'), ('r_wins_total', 'b_wins_total'), 
    ('r_losses_total', 'b_losses_total'), ('r_age', 'b_age'), ('r_height', 'b_height'), 
    ('r_weight', 'b_weight'), ('r_reach', 'b_reach'), ('r_SLpM_total', 'b_SLpM_total'), 
    ('r_SApM_total', 'b_SApM_total'), ('r_sig_str_acc_total', 'b_sig_str_acc_total'), 
    ('r_td_acc_total', 'b_td_acc_total'), ('r_str_def_total', 'b_str_def_total'), 
    ('r_td_def_total', 'b_td_def_total'), ('r_sub_avg', 'b_sub_avg'), ('r_td_avg', 'b_td_avg')
]

# r_fighter와 b_fighter 기준으로 승/패 변환
large_dataset['r_result'] = large_dataset['winner'].apply(lambda x: 'win' if x == 'Red' else 'lose')
large_dataset['b_result'] = large_dataset['winner'].apply(lambda x: 'win' if x == 'Blue' else 'lose')

# 함수: 각 r_/b_ 컬럼 페어에 대해 비교 그래프 생성
def plot_paired_columns(paired_columns):
    for r_feature, b_feature in paired_columns:
        # 분위수 계산
        r_feature_values = large_dataset[r_feature]
        b_feature_values = large_dataset[b_feature]

        # r_ 그룹화
        try:
            r_bins = pd.qcut(r_feature_values, q=[0, 0.25, 0.75, 1], duplicates='drop', retbins=True)[1]
            large_dataset[f'{r_feature}_group'] = pd.cut(r_feature_values, bins=r_bins, labels=['Low', 'Mid', 'High'], include_lowest=True)
        except ValueError:
            print(f"'{r_feature}' 컬럼은 유효한 그룹을 생성할 수 없어 건너뜁니다.")
            continue

        # b_ 그룹화
        try:
            b_bins = pd.qcut(b_feature_values, q=[0, 0.25, 0.75, 1], duplicates='drop', retbins=True)[1]
            large_dataset[f'{b_feature}_group'] = pd.cut(b_feature_values, bins=b_bins, labels=['Low', 'Mid', 'High'], include_lowest=True)
        except ValueError:
            print(f"'{b_feature}' 컬럼은 유효한 그룹을 생성할 수 없어 건너뜁니다.")
            continue

        # r_ 승률 계산
        r_win_rate = large_dataset.groupby(f'{r_feature}_group')['r_result'].apply(lambda x: (x == 'win').mean()).reset_index()

        # b_ 승률 계산
        b_win_rate = large_dataset.groupby(f'{b_feature}_group')['b_result'].apply(lambda x: (x == 'win').mean()).reset_index()

        # 그래프 그리기
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        sns.barplot(ax=axes[0], data=r_win_rate, x=f'{r_feature}_group', y='r_result', palette='cool')
        sns.barplot(ax=axes[1], data=b_win_rate, x=f'{b_feature}_group', y='b_result', palette='cool')

        # r_ 그래프 설정
        axes[0].set_title(f"r_fighter Win Rate by {r_feature}")
        axes[0].set_xlabel(f"{r_feature} Range (Low ~ High)")
        axes[0].set_ylabel("Win Rate")
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # b_ 그래프 설정
        axes[1].set_title(f"b_fighter Win Rate by {b_feature}")
        axes[1].set_xlabel(f"{b_feature} Range (Low ~ High)")
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

# 실행
plot_paired_columns(paired_columns)

