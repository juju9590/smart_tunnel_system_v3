import numpy as np
from tests.accident import AccidentDetectorV2
from traffic_state import TrafficStateDetector

class TunnelPipeline:
    def __init__(self):
        self.accident = AccidentDetectorV2()
        self.state_detector = TrafficStateDetector()

        self.prev_positions = {}

    def process(self, tracks):
        results = []
        speeds = {}

        # 정렬 (앞차 기준)
        tracks = sorted(tracks, key=lambda x: x["bbox"][3])

        for i, car in enumerate(tracks):
            id = car["id"]
            x1,y1,x2,y2 = car["bbox"]

            cx = (x1+x2)//2
            cy = y2

            # 속도
            speed = 0
            if id in self.prev_positions:
                px,py = self.prev_positions[id]
                speed = np.sqrt((cx-px)**2 + (cy-py)**2)

            self.prev_positions[id] = (cx,cy)
            speeds[id] = speed

            # 앞차 없음 skip
            if i == 0:
                continue

            front = tracks[i-1]
            fid = front["id"]

            fx1,fy1,fx2,fy2 = front["bbox"]
            fcx = (fx1+fx2)//2
            fcy = fy2

            # 같은 차선
            if abs(cx - fcx) > 50:
                continue

            dist = abs(cy - fcy)

            front_speed = 0
            if fid in self.prev_positions:
                fpx,fpy = self.prev_positions[fid]
                front_speed = np.sqrt((fcx-fpx)**2 + (fcy-fpy)**2)

            event = self.accident.update(
                id=id,
                dist=dist,
                speed=speed,
                front_speed=front_speed
            )

            results.append({
                "id": id,
                "speed": speed,
                "dist": dist,
                "event": event
            })

        # 상태 판단
        state = self.state_detector.update(list(speeds.values()))

        return results, state