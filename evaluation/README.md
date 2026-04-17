# Evaluation Guide

## 1. 목적

이 폴더는 스마트 터널 시스템의 성능평가 전용 코드를 관리한다.
실행 파이프라인과 분리하여, 기능별 정답(GT)과 예측 결과를 비교할 수 있도록 구성한다.

평가 대상:

* 차선 추정 / ROI 자동설정
* 교통 상태 판단
* 사고 판단

## 2. 폴더 구조

### 실행 코드

```
smart_tunnel_V3/
└─ evaluation/
   ├─ eval_utils.py
   ├─ eval_state_logger.py
   ├─ eval_accident_logger.py
   ├─ eval_lane_roi_logger.py
   └─ README.md
```

### 외부 평가 데이터 및 결과

```
smart_tunnel_V3_eval/
├─ ground_truth/
├─ outputs/
└─ outputs_summaries/
```

* `ground_truth/` : 정답 CSV 저장
* `outputs/` : 프레임별 상세 평가 로그 저장
* `outputs_summaries/` : 성능평가 요약 결과 저장

## 3. GT 파일 형식

### 3-1. 상태 평가 GT

파일 예:

* `state_gt_test_normal_2.csv`
* `state_gt_test_congestion_1.csv`
* `state_gt_test_accident_1.csv`

형식:
frame_id,gt_state
0,NORMAL
1,NORMAL
2,CONGESTION
3,JAM

사용 라벨:

* `NORMAL`
* `CONGESTION`
* `JAM`

### 3-2. 사고 평가 GT

파일 예:

* `accident_gt_test_accident_1.csv`

형식:
frame_id,gt_accident
0,NON_ACCIDENT
1,NON_ACCIDENT
2,ACCIDENT
3,ACCIDENT

사용 라벨:

* `NON_ACCIDENT`
* `ACCIDENT`

### 3-3. 차선 평가 GT

파일 예:

* `lane_gt_test_normal_2.csv`

형식:
frame_id,gt_lane_count
0,4
100,4
200,4

### 3-4. ROI 평가 GT

파일 예:

* `roi_gt_test_normal_2.csv`

형식:
frame_id,gt_roi_y1,gt_roi_y2
0,159,589
100,159,589
200,159,589

## 4. 실행 방법

### 4-1. 상태 평가

`eval_state_logger.py` 상단에서 아래 경로를 현재 평가할 영상에 맞게 수정 후 실행한다.

* `PIPELINE_LOG_CSV`
* `GT_CSV`

실행:
python eval_state_logger.py

출력:

* `smart_tunnel_V3_eval/outputs/state_eval_log_<video>.csv`
* `smart_tunnel_V3_eval/outputs_summaries/state_summary_<video>.csv`

### 4-2. 사고 평가

`eval_accident_logger.py` 상단에서 아래 경로를 수정 후 실행한다.

* `PIPELINE_LOG_CSV`
* `GT_CSV`

실행:
python eval_accident_logger.py

출력:

* `smart_tunnel_V3_eval/outputs/accident_eval_log_<video>.csv`
* `smart_tunnel_V3_eval/outputs_summaries/accident_summary_<video>.csv`

### 4-3. 차선 / ROI 평가

`eval_lane_roi_logger.py` 상단에서 아래 경로를 수정 후 실행한다.

* `PIPELINE_LOG_CSV`
* `LANE_GT_CSV`
* `ROI_GT_CSV`

실행:
python eval_lane_roi_logger.py

출력:

* `smart_tunnel_V3_eval/outputs/lane_roi_eval_log_<video>.csv`
* `smart_tunnel_V3_eval/outputs_summaries/lane_roi_summary_<video>.csv`

## 5. 주요 평가 지표

### 상태 평가

* Candidate Accuracy
* Final Accuracy
* GT 분포 / 예측 분포
* final_speed 평균 및 분포

활용 목적:

* 속도 임계값(`JAM`, `CONGESTION`) 튜닝
* 상태 로직 안정성 확인

### 사고 평가

* Accuracy
* TP / TN / FP / FN
* Precision / Recall / F1-score

활용 목적:

* 사고 검출 민감도 확인
* 미탐 / 오탐 비율 분석

### 차선 / ROI 평가

* Lane Accuracy
* ROI Match
* ROI y1 MAE
* ROI y2 MAE
* ROI Mean MAE

활용 목적:

* 차선 추정 정확도 확인
* ROI 자동설정 오차 확인

## 6. 운영 원칙

* 실행 파이프라인 코드는 수정하지 않고, 성능평가 코드는 별도 관리한다.
* GT 파일은 영상별로 따로 생성한다.
* 상태 / 사고 / 차선 / ROI 평가는 각각 독립적으로 수행한다.
* 평가 결과 파일은 GitHub 업로드 대상이 아니므로 외부 폴더(`smart_tunnel_V3_eval`)에 저장한다.

## 7. 현재 상태

* 상태 평가 코드 작성 완료
* 사고 평가 코드 작성 완료
* 차선/ROI 평가 코드 작성 완료
* 다음 단계: 상태 임계값 튜닝 및 사고 로직 보완
