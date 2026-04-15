# pipeline.py

from tests.accident import AccidentDetectorV2
import numpy as np

detector = AccidentDetectorV2()

prev_positions = {}

def process_frame(tracks):
    """
    tracks: [
        {id: 1, bbox: [x1,y1,x2,y2]},
        {id: 2, bbox: [...]}
    ]
    """

    results = []

    # 👉 y 기준 정렬 (앞차/뒷차 구분)
    tracks = sorted(tracks, key=lambda x: x["bbox"][3])

    for i, car in enumerate(tracks):
        id = car["id"]
        x1, y1, x2, y2 = car["bbox"]

        cx = (x1 + x2) // 2
        cy = y2  # 👈 바닥 기준 (중요)

        # -------------------------
        # 1️⃣ 속도 계산
        # -------------------------
        speed = 0
        if id in prev_positions:
            px, py = prev_positions[id]
            speed = np.sqrt((cx - px)**2 + (cy - py)**2)

        prev_positions[id] = (cx, cy)

        # -------------------------
        # 2️⃣ 앞차 찾기
        # -------------------------
        front_car = None

        if i > 0:
            front_car = tracks[i-1]

        if front_car is None:
            continue

        fx1, fy1, fx2, fy2 = front_car["bbox"]
        fcx = (fx1 + fx2) // 2
        fcy = fy2

        # -------------------------
        # 3️⃣ 거리 계산
        # -------------------------
        dist = abs(cy - fcy)

        # -------------------------
        # 4️⃣ 앞차 속도
        # -------------------------
        front_speed = 0
        fid = front_car["id"]

        if fid in prev_positions:
            fpx, fpy = prev_positions[fid]
            front_speed = np.sqrt((fcx - fpx)**2 + (fcy - fpy)**2)

        # -------------------------
        # 5️⃣ 사고 판단
        # -------------------------
        state = detector.update(
            id=id,
            dist=dist,
            speed=speed,
            front_speed=front_speed
        )

        results.append({
            "id": id,
            "state": state,
            "speed": speed,
            "dist": dist
        })

    return results