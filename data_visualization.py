import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fighter_stats_path = './fighter_stats.csv'
event_dataset_path = './event_dataset.csv'

fighter_stats = pd.read_csv(fighter_stats_path)
event_dataset = pd.read_csv(event_dataset_path)


# 승리/패배 정보 결합
r_fighter_stats = event_dataset[['r_fighter', 'weight_class']].merge(
    fighter_stats, left_on='r_fighter', right_on='name', how='left'
).drop(columns=['name'])
r_fighter_stats['result'] = 'win'

b_fighter_stats = event_dataset[['b_fighter', 'weight_class']].merge(
    fighter_stats, left_on='b_fighter', right_on='name', how='left'
).drop(columns=['name'])
b_fighter_stats['result'] = 'lose'

fighter_results = pd.concat([r_fighter_stats, b_fighter_stats], ignore_index=True)

### box-plot ###

## height와 승패 시각화 (체급 별)
plt.figure(figsize=(10, 6))
sns.boxplot(data=fighter_results, x='result', y='height', hue='weight_class', palette='cool')
plt.title("Height by Win/Lose and Weight Class")
plt.xlabel("Result")
plt.ylabel("Height (cm)")
plt.legend(title="Weight Class", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

## weight와 승패 비교 (체급 별)
plt.figure(figsize=(10, 6))
sns.boxplot(data=fighter_results, x='result', y='weight', hue='weight_class', palette='cool')
plt.title("Weight by Win/Lose and Weight Class")
plt.xlabel("Result")
plt.ylabel("Weight (kg)")
plt.legend(title="Weight Class", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

## reach와 승패 비교 (체급 별)
plt.figure(figsize=(10, 6))
sns.boxplot(data=fighter_results, x='result', y='reach', hue='weight_class', palette='cool')
plt.title("Reach by Win/Lose and Weight Class")
plt.xlabel("Result")
plt.ylabel("Reach (cm)")
plt.legend(title="Weight Class", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

## height와 승패 비교 (stance 별)
plt.figure(figsize=(10, 6))
sns.boxplot(data=fighter_results, x='result', y='height', hue='stance', palette='cool')
plt.title("Height by Win/Lose and Stance")
plt.xlabel("Result")
plt.ylabel("Height (cm)")
plt.legend(title="Stance", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

## weight와 승패 비교 (stance 별)
plt.figure(figsize=(10, 6))
sns.boxplot(data=fighter_results, x='result', y='weight', hue='stance', palette='cool')
plt.title("Weight by Win/Lose and Stance")
plt.xlabel("Result")
plt.ylabel("Weight (kg)")
plt.legend(title="Stance", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

## reach와 승패 비교 (stance 별)
plt.figure(figsize=(10, 6))
sns.boxplot(data=fighter_results, x='result', y='reach', hue='stance', palette='cool')
plt.title("Reach by Win/Lose and Stance")
plt.xlabel("Result")
plt.ylabel("Reach (cm)")
plt.legend(title="Stance", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


## stance 별 승률 시각화
stance_summary = (
    fighter_results.groupby(['stance', 'result'])
    .size()
    .reset_index(name='count')
)

stance_total = stance_summary.groupby('stance')['count'].sum().reset_index(name='total')
stance_summary = stance_summary.merge(stance_total, on='stance')
stance_summary['ratio'] = stance_summary['count'] / stance_summary['total']
stance_pivot = stance_summary.pivot(index='stance', columns='result', values='ratio').fillna(0)
plt.figure(figsize=(12, 6))
stance_pivot.plot(
    kind='barh', stacked=True, color=['#1f77b4', '#ff7f0e'], figsize=(12, 6), alpha=0.8
)
plt.title("Win/Lose Ratio by Stance (Normalized to 100%)")
plt.xlabel("Ratio")
plt.ylabel("Stance")
plt.legend(title="Result", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

