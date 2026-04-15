2. RULES.md

# 개발 규칙

## 클래스
- 모든 차량은 class 0 (car)

## TDD설계
- 먼저 tests 코드 작성
- scripts 코드 작성
- 테스트 통과 확인

## 테스트 파일 이름 규칙
- test_*.py
(예시) test_state.py, test_speed.py, test_collision.py

## 속도
- 픽셀 기반 속도 사용
- 이동 평균 적용

## 상태 기준
- NORMAL / CONGESTION / JAM / ACCIDENT

## ROI
- 자동 설정 사용

## 금지사항
- 임의 threshold 변경 금지
- 코드 구조 변경 시 반드시 기록


