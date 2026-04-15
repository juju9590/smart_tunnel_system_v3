# test_accident_v2.py

from tests.accident import AccidentDetectorV2

detector = AccidentDetectorV2()

# 시뮬레이션 데이터 (사고 패턴)
test_data = [
    # dist, speed, front_speed
    (100, 30, 20),  # 정상
    (80, 32, 20),   # 접근
    (50, 35, 20),   # 급접근 + 속도차 ↑
    (20, 36, 20),   # 거의 붙음
    (10, 10, 20),   # 💥 충돌 후 급감
    (10, 0, 0),     # 정지
]

for i, (dist, speed, front_speed) in enumerate(test_data):
    state = detector.update(
        id=1,
        dist=dist,
        speed=speed,
        front_speed=front_speed
    )

    print(f"[{i}] dist={dist}, speed={speed} → {state}")