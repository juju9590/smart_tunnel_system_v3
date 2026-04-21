# 🚧 Smart Tunnel AI System (V3)

## 📌 Overview
터널 CCTV 영상을 기반으로 차량을 탐지·추적하고,
교통 상태와 사고 이상징후를 분석하여
관제용 대시보드 및 위험도 판단에 활용하는 시스템입니다.

## 🧠 Main Features
- 차량 탐지 (YOLO)
- 차량 추적 (ByteTrack)
- 교통 상태 분석 (NORMAL / CONGESTION / JAM)
- 사고 및 이상징후 감지
- 위험도 분석 및 로그 저장
- 웹 대시보드 연동

## 🏗️ Pipeline
Detection → Tracking → State Analysis → Accident Detection → Risk / Dashboard

## 📁 Project Structure
```text
SMART_TUNNEL_V3/
├─ backend_flask/        # Flask backend
├─ data/                 # input videos, images, labels
├─ outputs/              # output videos, logs, csv
├─ scr/
│  ├─ analysis/          # analysis scripts
│  ├─ pipe/              # temporary / experimental pipeline code
│  ├─ pipeline/          # main pipeline versions
│  ├─ pipeline_V4/
│  ├─ pipeline_V4_1/
│  ├─ pipeline_V4.0/
│  ├─ pipeline_V5/
│  └─ pipeline_V5_1/
├─ scripts/              # utility scripts
├─ tests/                # test code
├─ AGENTS.md             # Codex project guidance
├─ RULES.md              # development rules
├─ TODO.md               # current progress
└─ README.md

```

## 🚀 Version History
- V1: 초기 실험
- V2: 로직 개선 시도
- V3: 구조 재설계 + 통합 시스템
- V4~V5: 사고/상태 판단 로직 개선 및 파이프라인 실험


## 🛠️ Development Notes
- 모든 차량 클래스는 학습 및 분석 과정에서 class 0 (car) 기준으로 통합 관리
- 속도는 픽셀 기반으로 계산
- 사고 판단과 상태 판단은 객체탐지 결과 이후의 로직 기반 분석으로 처리
- 로그는 이후 튜닝 및 평가를 위해 최대한 보존

## 📌 Current Status
- 데이터 수집 완료
- 자동 라벨링 완료
- 수동 라벨링 진행 중
- YOLO 학습 및 로직 튜닝 예정