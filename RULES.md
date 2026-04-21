# RULES.md

# 개발 규칙

## 1. Class rule
- 모든 차량은 class 0 (car) 기준으로 통합한다.
- 분석 로직에서는 차종 분리보다 차량 수, 추적, 상태 판단을 우선한다.

## 2. TDD / test workflow
- 가능하면 먼저 테스트 코드를 작성한다.
- 그 다음 scripts 또는 실제 로직 코드를 작성한다.
- 테스트 통과 여부를 확인한 뒤 반영한다.

## 3. Test file naming
- 테스트 파일명은 `test_*.py` 형식을 사용한다.
- 예시:
  - `test_state.py`
  - `test_speed.py`
  - `test_collision.py`

## 4. Speed logic
- 속도는 픽셀 기반 속도를 사용한다.
- 필요 시 이동 평균을 적용한다.
- 임의의 실제 km/h 환산값을 고정 상수처럼 쓰지 않는다.

## 5. State labels
- 기본 상태 값:
  - `NORMAL`
  - `CONGESTION`
  - `JAM`
  - `ACCIDENT`

## 6. ROI rule
- ROI는 자동 설정 로직을 우선 고려한다.
- ROI 변경 시 변경 이유를 기록한다.

## 7. Logging rule
- 상태 판단, 사고 판단, 디버깅에 필요한 로그는 최대한 유지한다.
- 가능하면 아래 값들을 보존한다:
  - frame index
  - vehicle ID
  - speed-related values
  - state
  - risk
  - accident-related signals

## 8. Threshold rule
- threshold는 임의로 자주 바꾸지 않는다.
- threshold를 바꿀 경우:
  1. 변경 이유
  2. 기대 효과
  3. 결과
  를 함께 기록한다.

## 9. Structure rule
- 기존 폴더 구조와 모듈 경계를 임의로 바꾸지 않는다.
- 구조 변경이 필요한 경우 반드시 기록을 남긴다.

## 10. Output rule
- 분석 결과는 가능한 한 영상, CSV, 로그 형태로 저장해 나중에 검토 가능하게 한다.
