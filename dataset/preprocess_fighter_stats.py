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
df['reach'] = pd.to_numeric(df['reach'], errors='coerce')
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# 숫자 변환 후 결측치 다시 확인
print("\n=== 숫자 변환 후 결측치 개수 ===")
print(df.isnull().sum())

# 평균값 계산
reach_mean = round(df['reach'].mean(), 2)
age_mean = round(df['age'].mean(), 2)

# 결측치 대체
df['reach'] = df['reach'].fillna(reach_mean)
df['age'] = df['age'].fillna(age_mean)

# 결측치 대체 값 확인
print("\n=== 결측치 대체 값 ===")
print(f"reach: {reach_mean}, age: {age_mean}")

# Stance 결측치 처리
stance_mode = df['stance'].mode()[0]
df['stance'] = df['stance'].fillna(stance_mode)

# 원핫인코딩 후 int 타입으로 변환
stance_dummies = pd.get_dummies(df['stance'], prefix='stance').astype(int)
df = df.drop('stance', axis=1)  # 기존 stance 열 삭제
df = pd.concat([df, stance_dummies], axis=1)  # 원핫인코딩된 열들 추가

print("\n=== Stance 원핫인코딩 결과 ===")
print("생성된 stance 열:", list(stance_dummies.columns))

print("\n=== 전처리 후 데이터 샘플 ===")
print(df.head())

print("\n=== 전처리 후 결측치 개수 ===")
print(df.isnull().sum())

# 결과 저장
first_row.to_csv('dataset/preprocessed/fighter_stats.csv', index=False, header=False, mode='w')
df.to_csv('dataset/preprocessed/fighter_stats.csv', index=False, header=False, mode='a')
