# 테스트 코드 (unittest 사용)
# 이미 yolo11n 학습결과 mAP50 = 0.910 이상으로 나와서,테스트 필요 없음
# 이제 교통흐름 중심 상태 판단 로직 테스트 필요

import unittest
# 파이썬에서 테스트하기 위한 표준 라이브러리 
# assert : 값이 참인지 확인하는 함수 (예: assertEqual, assertTrue 등)
# unittest.TestCase : 테스트 케이스를 작성하기 위한 기본 클래스 (테스트 묶음)
# unittest.main() : 테스트 실행 함수 (스크립트가 직접 실행될 때 테스트를 실행하도록 함)

# 교통 상태 판단 로직(차량 수 기준 판단_아주단순) 
def classify_state(vehicle_count):
    if vehicle_count >= 10:
        return "JAM"
    elif vehicle_count >= 5:
        return "CONGESTION"
    else:
        return "NORMAL"


class TestTrafficLogic(unittest.TestCase):

    def test_normal(self):
        self.assertEqual(classify_state(2), "NORMAL")

    def test_congestion(self):
        self.assertEqual(classify_state(6), "CONGESTION")

    def test_jam(self):
        self.assertEqual(classify_state(12), "JAM")


if __name__ == "__main__":
    unittest.main()