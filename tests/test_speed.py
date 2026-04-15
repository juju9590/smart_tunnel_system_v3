
# 파이썬에서 테스트하기 위한 표준 라이브러리
import unittest 

# 나중에 구현할 함수 import
# from speed import calculate_speed

# 임시 함수 (테스트 먼저 돌리기 위해)
def calculate_speed(prev_point, curr_point):
    if prev_point is None or curr_point is None:
        return 0

    prev_x, prev_y = prev_point
    curr_x, curr_y = curr_point

    dx = curr_x - prev_x
    dy = curr_y - prev_y

    distance = (dx**2 + dy**2) ** 0.5 
    # 유클리드 거리 계산 (픽셀 단위) - 실제로는 시간 간격도 고려해야 하지만, 
    # 여기서는 거리 계산에 초점 (아직 기초 속도)
    return distance


class TestSpeedCalculation(unittest.TestCase):

    # 1️⃣ 이전 좌표 없음
    def test_no_previous_point(self):
        speed = calculate_speed(None, (100, 200))
        self.assertEqual(speed, 0)

    # 2️⃣ 이동 없음
    def test_no_movement(self):
        speed = calculate_speed((100, 200), (100, 200))
        self.assertEqual(speed, 0)

    # 3️⃣ 정상 이동
    def test_normal_movement(self):
        speed = calculate_speed((100, 200), (110, 200))
        self.assertGreater(speed, 0)

    # 4️⃣ 대각선 이동
    def test_diagonal_movement(self):
        speed = calculate_speed((100, 200), (110, 210))
        self.assertGreater(speed, 0)

    # 5️⃣ 큰 점프 (이상값)
    def test_large_jump(self):
        speed = calculate_speed((100, 200), (1000, 200))
        self.assertGreater(speed, 0)

    # 6️⃣ 음수 이동 (뒤로 이동)
    def test_backward_movement(self):
        speed = calculate_speed((110, 200), (100, 200))
        self.assertGreaterEqual(speed, 0)

    # 7️⃣ 동일 x, y만 이동
    def test_vertical_movement(self):
        speed = calculate_speed((100, 200), (100, 250))
        self.assertGreater(speed, 0)


if __name__ == "__main__":
    unittest.main()

# 👉 "속도 함수 테스트 구조"
# 이 단계는 속도 계산 함수의 기본적인 동작을 검증하기 위한 테스트
# 실제로는 시간 간격을 고려한 속도 계산이 필요하지만, 여기서는  거리 계산에 초점 맞춤

# 나중에 반드시 바꿔야 할 부분:
# - y2 기준 속도
# - 이동 평균 (5프레임)
# - max speed 제한
