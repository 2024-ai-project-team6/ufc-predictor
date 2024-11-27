import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

first_row = pd.read_csv('fighter_stats.csv', nrows=1, header=None)

# CSV 파일 읽기
df = pd.read_csv('fighter_stats.csv', header=None, 
                 names=['Name', 'Height', 'Weight', 'Reach', 'Stance'],
                 skiprows=1)

# 전처리 전 결측치 확인
print("=== 전처리 전 결측치 개수 ===")
print(df.isnull().sum())

# 전처리 전 Stance 값 분포 확인
print("\n=== 전처리 전 Stance 값 분포 ===")
print(df['Stance'].value_counts())
print(f"Stance 결측치 개수: {df['Stance'].isnull().sum()}")

# 수치형 데이터 변환 및 결측치 처리
df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
df['Reach'] = pd.to_numeric(df['Reach'], errors='coerce')

# 숫자 변환 후 결측치 다시 확인
print("\n=== 숫자 변환 후 결측치 개수 ===")
print(df.isnull().sum())

# 평균값 계산
height_mean = round(df['Height'].mean(), 2)
weight_mean = round(df['Weight'].mean(), 2)
reach_mean = round(df['Reach'].mean(), 2)

# 결측치 대체
df['Height'] = df['Height'].fillna(height_mean)
df['Weight'] = df['Weight'].fillna(weight_mean)
df['Reach'] = df['Reach'].fillna(reach_mean)

# Stance 결측치 처리
stance_mode = df['Stance'].mode()[0]
df['Stance'] = df['Stance'].fillna(stance_mode)

label_encoder = LabelEncoder()
df['Stance'] = label_encoder.fit_transform(df['Stance'])

print("\n=== Stance 레이블 인코딩 매핑 ===")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {i}")

print("\n=== 전처리 후 데이터 샘플 ===")
print(df.head())

print("\n=== 전처리 후 결측치 개수 ===")
print(df.isnull().sum())

# 결과 저장
first_row.to_csv('preprocessed_fighter_stats.csv', index=False, header=False, mode='w')
df.to_csv('preprocessed_fighter_stats.csv', index=False, header=False, mode='a')