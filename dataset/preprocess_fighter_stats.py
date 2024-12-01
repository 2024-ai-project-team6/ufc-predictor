import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

first_row = pd.read_csv('dataset/raw/fighter_stats.csv', nrows=1, header=None)

# CSV 파일 읽기
df = pd.read_csv('dataset/raw/fighter_stats.csv', header=None, 
                 names=['name','wins','losses','height','weight','reach','stance','age','SLpM','sig_str_acc','SApM','str_def','td_avg','td_acc','td_def','sub_avg'],
                 skiprows=1)

# 전처리 전 결측치 확인
print("=== 전처리 전 결측치 개수 ===")
print(df.isnull().sum())

# 전처리 전 Stance 값 분포 확인
print("\n=== 전처리 전 Stance 값 분포 ===")
print(df['stance'].value_counts())
print(f"Stance 결측치 개수: {df['stance'].isnull().sum()}")

# 수치형 데이터 변환 및 결측치 처리
df['height'] = pd.to_numeric(df['height'], errors='coerce')
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
df['reach'] = pd.to_numeric(df['reach'], errors='coerce')

# 숫자 변환 후 결측치 다시 확인
print("\n=== 숫자 변환 후 결측치 개수 ===")
print(df.isnull().sum())

# 평균값 계산
height_mean = round(df['height'].mean(), 2)
weight_mean = round(df['weight'].mean(), 2)
reach_mean = round(df['reach'].mean(), 2)
age_mean = round(df['age'].mean(), 2)

# 결측치 대체
df['height'] = df['height'].fillna(height_mean)
df['weight'] = df['weight'].fillna(weight_mean)
df['reach'] = df['reach'].fillna(reach_mean)
df['age'] = df['age'].fillna(age_mean)

# 결측치 대체 값 확인
print("\n=== 결측치 대체 값 ===")
print(f"height: {height_mean}, weight: {weight_mean}, reach: {reach_mean}, age: {age_mean}")

# Stance 결측치 처리
stance_mode = df['stance'].mode()[0]
df['stance'] = df['stance'].fillna(stance_mode)

label_encoder = LabelEncoder()
df['stance'] = label_encoder.fit_transform(df['stance'])

print("\n=== Stance 레이블 인코딩 매핑 ===")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {i}")

print("\n=== 전처리 후 데이터 샘플 ===")
print(df.head())

print("\n=== 전처리 후 결측치 개수 ===")
print(df.isnull().sum())

# # age가 결측치인 행의 인덱스 찾기
# missing_age_indices = df[df['age'].isna()].index.tolist()

# # 결과 출력
# print("age가 결측치인 행의 인덱스:")
# print(missing_age_indices)

# # 해당 행들의 전체 데이터 확인
# print("\n해당 행들의 전체 데이터:")
# print(df.loc[missing_age_indices])

# # 총 결측치 개수
# print(f"\n총 age 결측치 개수: {len(missing_age_indices)}")

# 결과 저장
first_row.to_csv('dataset/preprocessed/fighter_stats.csv', index=False, header=False, mode='w')
df.to_csv('dataset/preprocessed/fighter_stats.csv', index=False, header=False, mode='a')
