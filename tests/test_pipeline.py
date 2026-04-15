# test_pipeline.py

from pipeline import process_frame

# 프레임별 차량 상황 (사고 시뮬레이션)
frames = [
    [
        {"id": 1, "bbox": [100, 100, 150, 150]},
        {"id": 2, "bbox": [100, 200, 150, 250]},
    ],
    [
        {"id": 1, "bbox": [100, 120, 150, 170]},
        {"id": 2, "bbox": [100, 210, 150, 260]},
    ],
    [
        {"id": 1, "bbox": [100, 150, 150, 200]},
        {"id": 2, "bbox": [100, 220, 150, 270]},
    ],
    [
        {"id": 1, "bbox": [100, 180, 150, 230]},
        {"id": 2, "bbox": [100, 230, 150, 280]},
    ],
    [
        {"id": 1, "bbox": [100, 200, 150, 250]},  # 앞차 느림
        {"id": 2, "bbox": [100, 210, 150, 260]},  # 뒤차 급접근
    ],
    [
        {"id": 1, "bbox": [100, 200, 150, 250]},
        {"id": 2, "bbox": [100, 205, 150, 255]},  # 충돌 후 정지
    ],
]

for i, frame in enumerate(frames):
    results = process_frame(frame)

    print(f"\n[FRAME {i}]")
    for r in results:
        print(r)