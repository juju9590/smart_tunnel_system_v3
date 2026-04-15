# accident.py _V2
# STEP1: 급접근 발생 → flag 저장
# STEP2: 속도차 증가 → flag 저장
# STEP3: 이후 속도 급감 → 사고




class AccidentDetectorV2:
    def __init__(self):
        self.history = {}

        self.DIST_THRESHOLD = 15
        self.SPEED_GAP_THRESHOLD = 5
        self.SPEED_DROP_THRESHOLD = -5

    def update(self, id, dist, speed, front_speed):
        if id not in self.history:
            self.history[id] = {
                "dist": [],
                "speed": [],
                "approach": False,
                "speeding": False,
                "state": "NORMAL"
            }

        h = self.history[id]

        h["dist"].append(dist)
        h["speed"].append(speed)

        if len(h["dist"]) < 2:
            return "NORMAL"

        # 계산
        dist_diff = h["dist"][-2] - h["dist"][-1]
        speed_gap = speed - front_speed
        speed_drop = h["speed"][-1] - h["speed"][-2]

        # 1️⃣ 급접근 감지
        if dist_diff > self.DIST_THRESHOLD:
            h["approach"] = True

        # 2️⃣ 속도 차이 감지
        if speed_gap > self.SPEED_GAP_THRESHOLD:
            h["speeding"] = True

        # 3️⃣ 충돌 후 급감 → 사고
        if h["approach"] and h["speeding"] and speed_drop < self.SPEED_DROP_THRESHOLD:
            h["state"] = "ACCIDENT"
            return "ACCIDENT"

        return "NORMAL"