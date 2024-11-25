import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = './large_dataset.csv'
large_dataset = pd.read_csv(file_path)

# r_fighter 기준으로 승/패 변환
large_dataset['result'] = large_dataset['winner'].apply(lambda x: 'win' if x == 'Red' else 'lose')

diff_columns = [col for col in large_dataset.columns if '_diff' in col]

#  박스플롯 시각화
for col in diff_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=large_dataset, x='result', y=col, palette='cool')
    plt.title(f"Distribution of {col} by r_fighter Win/Lose")
    plt.xlabel("Result (Win = r_fighter Wins, Lose = r_fighter Loses)")
    plt.ylabel(col)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 선 그래프 시각화
for col in diff_columns:
    # diff 값을 구간별로 나누고 각 구간에서의 승률 계산
    bins = pd.qcut(large_dataset[col], q=10, duplicates='drop') 
    win_rate_by_bin = large_dataset.groupby(bins)['result'].apply(lambda x: (x == 'win').mean())

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=win_rate_by_bin.index.astype(str), y=win_rate_by_bin.values, marker='o', color='blue')
    plt.title(f"Win Rate by {col} (r_fighter)")
    plt.xlabel(col)
    plt.ylabel("Win Rate")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

