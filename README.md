
    weight class별로 시간에 따른 dynamic graph를 생성합니다.
    edge의 방향은 추가된 경기의 승패를 나타냅니다.
    새롭게 추가된 데이터가 업데이트한 노드 사이의 엣지 방향을 예측하는 프로그램을 작성해주세요.

## 컬럼 의미
| Column Name              | Description (설명)                                                                                   |
|--------------------------|------------------------------------------------------------------------------------------------------|
| `kd_diff`                | Knockdown Difference: Red 선수가 Blue 선수보다 기록한 넉다운 수 차이.                                |
| `sig_str_diff`           | Significant Strike Difference: Red 선수가 Blue 선수보다 기록한 유효 타격 수 차이.                     |
| `sig_str_att_diff`       | Significant Strike Attempt Difference: Red 선수가 Blue 선수보다 시도한 유효 타격 수 차이.            |
| `sig_str_acc_diff`       | Significant Strike Accuracy Difference: Red 선수의 유효 타격 정확도가 Blue 선수보다 얼마나 높은지.    |
| `str_diff`               | Total Strike Difference: Red 선수가 Blue 선수보다 기록한 총 타격 수 차이.                            |
| `str_att_diff`           | Total Strike Attempt Difference: Red 선수가 Blue 선수보다 시도한 총 타격 수 차이.                    |
| `str_acc_diff`           | Total Strike Accuracy Difference: Red 선수의 총 타격 정확도가 Blue 선수보다 얼마나 높은지.           |
| `td_diff`                | Takedown Difference: Red 선수가 Blue 선수보다 성공한 테이크다운 수 차이.                             |
| `td_att_diff`            | Takedown Attempt Difference: Red 선수가 Blue 선수보다 시도한 테이크다운 수 차이.                     |
| `td_acc_diff`            | Takedown Accuracy Difference: Red 선수의 테이크다운 성공률이 Blue 선수보다 얼마나 높은지.            |
| `sub_att_diff`           | Submission Attempt Difference: Red 선수가 Blue 선수보다 시도한 서브미션 수 차이.                     |
| `rev_diff`               | Reversal Difference: Red 선수가 Blue 선수보다 기록한 리버설 수 차이.                                |
| `ctrl_sec_diff`          | Control Time Difference: Red 선수가 Blue 선수보다 컨트롤한 총 시간(초) 차이.                         |
| `wins_total_diff`        | Total Wins Difference: Red 선수가 Blue 선수보다 기록한 총 승리 횟수 차이.                            |
| `losses_total_diff`      | Total Losses Difference: Red 선수가 Blue 선수보다 기록한 총 패배 횟수 차이.                          |
| `age_diff`               | Age Difference: Red 선수가 Blue 선수보다 얼마나 나이가 많은지(또는 적은지).                         |
| `height_diff`            | Height Difference: Red 선수가 Blue 선수보다 얼마나 큰지(또는 작은지).                               |
| `weight_diff`            | Weight Difference: Red 선수가 Blue 선수보다 얼마나 무거운지(또는 가벼운지).                         |
| `reach_diff`             | Reach Difference: Red 선수가 Blue 선수보다 리치가 얼마나 긴지(또는 짧은지).                         |
| `SLpM_total_diff`        | Significant Strikes Landed per Minute Difference: Red 선수가 Blue 선수보다 분당 얼마나 더 많은 유효 타격을 성공시켰는지. |
| `SApM_total_diff`        | Significant Strikes Absorbed per Minute Difference: Red 선수가 Blue 선수보다 분당 얼마나 적게 맞았는지.|
| `sig_str_acc_total_diff` | Total Significant Strike Accuracy Difference: Red 선수의 전체 유효 타격 정확도가 Blue 선수보다 얼마나 높은지. |
| `td_acc_total_diff`      | Total Takedown Accuracy Difference: Red 선수의 전체 테이크다운 성공률이 Blue 선수보다 얼마나 높은지.  |
| `str_def_total_diff`     | Total Strike Defense Difference: Red 선수의 타격 방어율이 Blue 선수보다 얼마나 높은지.               |
| `td_def_total_diff`      | Total Takedown Defense Difference: Red 선수의 테이크다운 방어율이 Blue 선수보다 얼마나 높은지.        |
| `sub_avg_diff`           | Submission Attempts per Fight Difference: Red 선수가 경기당 평균적으로 Blue 선수보다 얼마나 더 많은 서브미션을 시도했는지. |
| `td_avg_diff`            | Takedown Attempts per Fight Difference: Red 선수가 경기당 평균적으로 Blue 선수보다 얼마나 더 많은 테이크다운을 시도했는지. |

## 데이터 시각화
- 원본 large_dataset.csv를 이용하였습니다.
- 가로축은 (나의 stat - 상대방의 stat)을 의미합니다


![age_diff_boxplot](https://github.com/user-attachments/assets/f751d389-f67b-4e0b-bec4-28b38944ad9c)
![wins_total_diff_lineplot](https://github.com/user-attachments/assets/175c62ae-6afa-484d-aa2c-1c84b0d85a38)
![wins_total_diff_boxplot](https://github.com/user-attachments/assets/5eb4523d-e99b-4e89-8d7d-fdbd006bc920)
![weight_diff_lineplot](https://github.com/user-attachments/assets/0d37edb2-dc56-4b64-ae6e-1b785d947bd6)
![weight_diff_boxplot](https://github.com/user-attachments/assets/f51e2066-e80f-46c6-922e-8926d6a181df)
![td_diff_lineplot](https://github.com/user-attachments/assets/bb0e36c7-2d4f-492c-8834-282335dd27f1)
![td_diff_boxplot](https://github.com/user-attachments/assets/5a29aea3-287b-468a-9d78-b34bc723dad6)
![td_def_total_diff_lineplot](https://github.com/user-attachments/assets/2c60ac02-5d89-4922-bdca-dd5ef91cf68f)
![td_def_total_diff_boxplot](https://github.com/user-attachments/assets/da16132a-cc78-4569-a8e0-83be0d59e1f9)
![td_avg_diff_lineplot](https://github.com/user-attachments/assets/13c59024-d5d0-48d5-a10b-5f7087d5c290)
![td_avg_diff_boxplot](https://github.com/user-attachments/assets/20f8fba2-298e-4e23-9473-d4a6bbb9a3f6)
![td_att_diff_lineplot](https://github.com/user-attachments/assets/6ab732cc-3b60-49ec-8a5e-0e5f4ab17b61)
![td_att_diff_boxplot](https://github.com/user-attachments/assets/88d31702-b595-414a-89ed-c535f8b2ebb6)
![td_acc_total_diff_lineplot](https://github.com/user-attachments/assets/52cd8182-ca33-4551-93a2-ec9549f26cd5)
![td_acc_total_diff_boxplot](https://github.com/user-attachments/assets/38693ac1-e58c-4baf-91d0-58c3b7bcc0d1)
![td_acc_diff_lineplot](https://github.com/user-attachments/assets/72e0c6f7-08e8-46a8-beb2-99217607ebba)
![td_acc_diff_boxplot](https://github.com/user-attachments/assets/34c1f2ec-0eb4-4a56-9108-6546dbfc36f6)
![sub_avg_diff_lineplot](https://github.com/user-attachments/assets/f68d2d19-d9da-4c22-9aab-a90d607ac2f2)
![sub_avg_diff_boxplot](https://github.com/user-attachments/assets/a966e92c-887a-4fd8-89b2-41a3803cf5e7)
![sub_att_diff_lineplot](https://github.com/user-attachments/assets/e63ef73b-d251-4c1a-b25c-7fb26783add6)
![sub_att_diff_boxplot](https://github.com/user-attachments/assets/d8633def-1e32-4b1c-a105-e26620b34f52)
![str_diff_lineplot](https://github.com/user-attachments/assets/53e5840d-22cb-447c-bd4f-785287a033d2)
![str_diff_boxplot](https://github.com/user-attachments/assets/07b2aac7-7004-4abb-a3f3-4e2919a8997a)
![str_def_total_diff_lineplot](https://github.com/user-attachments/assets/ecf63f7d-a699-4db6-9376-72d1c6128b6a)
![str_def_total_diff_boxplot](https://github.com/user-attachments/assets/e5c36635-00dc-4d2e-8e8e-276d6b99961e)
![str_att_diff_lineplot](https://github.com/user-attachments/assets/4e66d2d8-ae8e-4e45-bf35-2f89f427207b)
![str_att_diff_boxplot](https://github.com/user-attachments/assets/acafb405-2177-4b88-b348-9a56c8f9b4e4)
![str_acc_diff_lineplot](https://github.com/user-attachments/assets/526d7ae9-09ac-4a72-8dec-2ecad2a5df93)
![str_acc_diff_boxplot](https://github.com/user-attachments/assets/d6b2dab2-20b3-4182-88fe-3c681b1c48f0)
![SLpM_total_diff_lineplot](https://github.com/user-attachments/assets/2b58ec5f-60be-4875-950b-091200ba9625)
![SLpM_total_diff_boxplot](https://github.com/user-attachments/assets/5749df3c-f623-4253-8d47-e7f08cd7fa32)
![sig_str_diff_lineplot](https://github.com/user-attachments/assets/39511574-b39e-4b22-8a98-b348d14ad50a)
![sig_str_diff_boxplot](https://github.com/user-attachments/assets/4fab5eb8-3612-4d72-88e4-fb0040821a76)
![sig_str_att_diff_lineplot](https://github.com/user-attachments/assets/4560de8d-e5be-47a5-b839-ec7927fc7108)
![sig_str_att_diff_boxplot](https://github.com/user-attachments/assets/035fb9a7-7298-4189-afc6-b45608e9aa30)
![sig_str_acc_total_diff_lineplot](https://github.com/user-attachments/assets/e01d09b6-3ef1-426a-b378-5a1bb67cb276)
![sig_str_acc_total_diff_boxplot](https://github.com/user-attachments/assets/1832eecf-609a-4d98-90e4-56552ccb869e)
![sig_str_acc_diff_lineplot](https://github.com/user-attachments/assets/1291b34f-844e-49fe-9236-4880ee2ab29f)
![sig_str_acc_diff_boxplot](https://github.com/user-attachments/assets/7f8a42e7-0c39-4e85-85fe-eaef980ee492)
![SApM_total_diff_lineplot](https://github.com/user-attachments/assets/1c155169-7a3a-455b-9606-22346c330ef3)
![SApM_total_diff_boxplot](https://github.com/user-attachments/assets/1b12b779-20ca-4f3d-9796-b4c9126649ab)
![rev_diff_lineplot](https://github.com/user-attachments/assets/7e593de4-83ff-42cb-b86a-e6ffec93eacd)
![rev_diff_boxplot](https://github.com/user-attachments/assets/958c5db4-e7ac-4ac5-9310-cf37725b16ff)
![reach_diff_lineplot](https://github.com/user-attachments/assets/3954ab9b-0382-4ff3-96d2-889c0467c427)
![reach_diff_boxplot](https://github.com/user-attachments/assets/20110882-de8a-4d2f-8d64-5593c8e0ad64)
![losses_total_diff_lineplot](https://github.com/user-attachments/assets/3f852e57-6507-49e1-8ba5-853886d68d44)
![losses_total_diff_boxplot](https://github.com/user-attachments/assets/f36114fa-ffc3-4881-98c3-b571a1b78523)
![kd_diff_lineplot](https://github.com/user-attachments/assets/c2832abc-9a1c-4239-800a-f19f7e180def)
![kd_diff_boxplot](https://github.com/user-attachments/assets/0e1df7b5-9762-4710-90ae-68aecdaad373)
![height_diff_lineplot](https://github.com/user-attachments/assets/5c01d381-fd18-40ee-a125-4a411fed1903)
![height_diff_boxplot](https://github.com/user-attachments/assets/fb044e25-a532-48d8-b32b-b5ebf388165d)
![ctrl_sec_diff_lineplot](https://github.com/user-attachments/assets/a2c13b7e-fe92-4904-aeba-6f233c730395)
![ctrl_sec_diff_boxplot](https://github.com/user-attachments/assets/58cf35c6-c110-45d1-b730-d89bb95a05ff)
![age_diff_lineplot](https://github.com/user-attachments/assets/fca98754-f600-4495-805d-fb39730ad999)

# column을 그룹화 하여 분류
- 하위 25% (Low) / 상위 25% (High) / 그 중간(MID) 총 3단계로 분류
- 하위 25%란 수치상으로 하위 25%를 의미함. (하위로갈수록 적은 수치, 상위로 갈 수록 높은 수치를 의미)
- red , blue 각각을 분리해서 집계한 결과가 포함

## 결과
![r_SApM_total_vs_b_SApM_total](https://github.com/user-attachments/assets/eb50bbb6-22ee-407f-8f2c-60d8b7d7601b)
![r_SLpM_total_vs_b_SLpM_total](https://github.com/user-attachments/assets/ca2c72b6-f589-4915-afcc-b24440da08d0)
![r_age_vs_b_age](https://github.com/user-attachments/assets/ce5ad7da-6914-4f5a-a9b1-0595f77fba95)
![r_ctrl_sec_vs_b_ctrl_sec](https://github.com/user-attachments/assets/0009df3c-84c6-4042-8aff-21cf4f58e2d2)
![r_height_vs_b_height](https://github.com/user-attachments/assets/eb8c9e18-5d96-4c5d-a361-0d5b57b640a2)
![r_losses_total_vs_b_losses_total](https://github.com/user-attachments/assets/49d959a0-02a4-44b4-801f-d419f3015593)
![r_reach_vs_b_reach](https://github.com/user-attachments/assets/30d1720f-261c-4806-94ef-b89409e44eb4)
![r_sig_str_acc_total_vs_b_sig_str_acc_total](https://github.com/user-attachments/assets/8f5b89cd-d84b-4cbe-9e42-0c41e26b0eda)
![r_sig_str_acc_vs_b_sig_str_acc](https://github.com/user-attachments/assets/26c5da34-a960-4abb-bee0-6c4ff30bbc36)
![r_sig_str_att_vs_b_sig_str_att](https://github.com/user-attachments/assets/d0e88ef9-f697-40f9-9aa8-7cc38bc752cd)
![r_sig_str_vs_b_sig_str](https://github.com/user-attachments/assets/ea898e30-c1bd-43b0-bff5-36df61864004)
![r_str_acc_vs_b_str_acc](https://github.com/user-attachments/assets/766ef747-28af-49cc-a7c5-eb095bd6d39a)
![r_str_att_vs_b_str_att](https://github.com/user-attachments/assets/98dc7638-7865-4d05-bb17-4cdc87e917fc)
![r_str_def_total_vs_b_str_def_total](https://github.com/user-attachments/assets/4e3b5783-2d17-470f-926d-d258fadd7057)
![r_str_vs_b_str](https://github.com/user-attachments/assets/388a666b-fa09-4c5d-a471-60e545d01a2a)
![r_td_acc_total_vs_b_td_acc_total](https://github.com/user-attachments/assets/3ddc6601-6a8a-48f9-ac22-613bebb8cf69)
![r_td_avg_vs_b_td_avg](https://github.com/user-attachments/assets/8e0829ff-7fc9-4f50-9216-27e863b44e51)
![r_td_def_total_vs_b_td_def_total](https://github.com/user-attachments/assets/56b3cc61-6ff5-44bf-a93a-8e6d1a16b4c4)
![r_weight_vs_b_weight](https://github.com/user-attachments/assets/d11abc85-ff55-4a3a-9dea-ff4e45c7bdca)
![r_wins_total_vs_b_wins_total](https://github.com/user-attachments/assets/a600d86f-beac-4466-9ae3-54fb61ea410f)


## F1-Score explain
F1-Score 공식
F1-Score는 Precision과 Recall의 조화 평균(Harmonic Mean)입니다:

F1 = 2 * (Precision*Recall) / (Precision+Recall)
​
Precision (정밀도): Positive로 예측한 것 중에서 실제로 Positive인 비율.

Precision = True Positive (TP) / (True Positive (TP) + False Positive (FP))
​

Recall (재현율): 실제 Positive 중에서 Positive로 정확히 예측한 비율.

Recall = True Positive (TP) / (True Positive (TP) + False Negative (FN))

Positive F1-Score
Positive Class (1)에 대해 Precision과 Recall을 사용하여 계산:
Positive F1-Score = 2 * (Precision(positive) * Recall(positive)) / (Precision(positive) + Recall(positive))
​
Positive Class의 TP, FP, FN:
True Positive (TP): 실제 1이고 모델이 1로 예측한 경우.
False Positive (FP): 실제 0인데 모델이 1로 잘못 예측한 경우.
False Negative (FN): 실제 1인데 모델이 0으로 잘못 예측한 경우.


Negative F1-Score
Negative Class (0)에 대해 Precision과 Recall을 사용하여 계산:
Negative F1-Score = 2 * (Precision(negative) * Recall(negative)) / (Precision(negative) + Recall(negative))
 
Negative Class의 TP, FP, FN:
True Negative (TN): 실제 0이고 모델이 0으로 예측한 경우.
False Positive (FP): 실제 0인데 모델이 1로 잘못 예측한 경우.
False Negative (FN): 실제 1인데 모델이 0으로 잘못 예측한 경우.
​
