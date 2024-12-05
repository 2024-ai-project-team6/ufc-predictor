import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = './large_dataset.csv'
large_dataset = pd.read_csv(file_path)

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

# Red 승 -> / 1 Blue 승 -> 0
large_dataset['result'] = large_dataset['winner'].apply(lambda x: 1 if x == 'Red' else 0)  

def plot_paired_columns(paired_columns):
    for r_feature, b_feature in paired_columns:
        # r_와 b_ 값을 승패에 맞게 하나로 통합
        large_dataset[f'{r_feature}_combined'] = large_dataset.apply(
            lambda x: x[r_feature] if x['result'] == 1 else x[b_feature], axis=1
        )
        
        # 최솟값과 최댓값을 기준으로 4등분
        min_val = large_dataset[f'{r_feature}_combined'].min()
        max_val = large_dataset[f'{r_feature}_combined'].max()
        
        # 4등분 구간 계산
        bin_edges = [min_val, (min_val + max_val) / 4, (min_val + 2 * max_val) / 4, 
                     (min_val + 3 * max_val) / 4, max_val]
        
        # 구간을 오름차순으로 정렬
        bin_edges.sort()
        
        # 경계값을 소수점 2자리로 포맷
        bin_labels = [f"{bin_edges[0]:.2f}-{bin_edges[1]:.2f}", 
                      f"{bin_edges[1]:.2f}-{bin_edges[2]:.2f}", 
                      f"{bin_edges[2]:.2f}-{bin_edges[3]:.2f}",
                      f"{bin_edges[3]:.2f}-{bin_edges[4]:.2f}"]
        
        large_dataset[f'{r_feature}_combined_group'] = pd.cut(
            large_dataset[f'{r_feature}_combined'], bins=bin_edges, 
            labels=bin_labels,
            include_lowest=True
        )

        win_rate = large_dataset.groupby(f'{r_feature}_combined_group')['result'].apply(lambda x: x.mean()).reset_index()

        plt.figure(figsize=(8, 5))
        sns.barplot(data=win_rate, x=f'{r_feature}_combined_group', y='result', palette='cool')
        plt.title(f'Win Rate by Combined {r_feature}')
        plt.xlabel(f"{r_feature} Combined Range (Low ~ High)")
        plt.ylabel("Win Rate")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

plot_paired_columns(paired_columns)

