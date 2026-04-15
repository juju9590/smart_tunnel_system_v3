# ==========================================
# 파일명 예시: traffic_accident_V4.py
# 용도: 파이프라인용 사고 감지 로직 클래스
# ==========================================
# [설명]
# 이 파일은 "단독 실행형 사고 감지 스크립트"를
# "파이프라인에서 호출하는 클래스형 구조"로 바꾼 버전이다.
#
# 즉,
#   cap = cv2.VideoCapture(...)
#   model.track(...)
#   cv2.imshow(...)
# 같은 코드는 여기 없다.
#
# 대신 PipelineCore 쪽에서 이미 추적된 tracks 정보를 받아서
# 사고 여부만 판단한다.
#
# 사용 흐름:
#   detector = AccidentDetectorV4()
#   result = detector.update(frame_idx, frame, tracks)
#
# 여기서 tracks는 보통 아래와 같은 형태라고 가정한다:
#   [
#       {
#           "id": 3,
#           "bbox": [x1, y1, x2, y2]
#       },
#       {
#           "id": 7,
#           "bbox": [x1, y1, x2, y2]
#       }
#   ]
#
# 필요하면 나중에 class, conf, lane 등을 추가해서 확장 가능하다.
# ==========================================

import cv2
import numpy as np
from sklearn.cluster import KMeans


class AccidentDetector:
    def __init__(
        self,
        alpha=0.3,              # 속도 smoothing 비율
        accident_hold=5,        # 사고 확정 hold 프레임
        iou_stop_hold=5,        # IoU 정지 hold 프레임
        init_frames=300,        # 차선 학습 시작 프레임
        lane_break_threshold=50,# 차선이탈 판단 임계값
        vertical_threshold=30,  # 세로 정렬 판단 임계값
        flow_break_threshold=0.3
    ):
        """
        사고 감지기 초기화

        [왜 필요한가?]
        사고 판단은 한 프레임만 보고 판단하면 흔들리기 쉽다.
        그래서 이전 프레임 정보, 차량 속도, 차량쌍 거리 변화 등을
        모두 메모리에 저장해 두고 누적 판단한다.
        """

        # ------------------------------
        # 추적 궤적 메모리
        # tid -> [(cx, cy), ...]
        # ------------------------------
        self.track_history = {}

        # ------------------------------
        # 속도 메모리
        # tid -> 이전 smoothing speed
        # ------------------------------
        self.speed_memory = {}

        # ------------------------------
        # 차량쌍(pair) 메모리
        # "id1-id2" -> {"dist": ..., "gap": ...}
        # ------------------------------
        self.pair_memory = {}

        # ------------------------------
        # 사고 hold 카운터
        # "id1-id2" -> int
        # ------------------------------
        self.accident_counter = {}

        # ------------------------------
        # IoU 정지 카운터
        # "id1-id2" -> int
        # ------------------------------
        self.iou_stop_counter = {}

        # ------------------------------
        # 차선 저장
        # tid -> lane index
        # ------------------------------
        self.lanes = {}

        # ------------------------------
        # 차선 학습 관련 변수
        # ------------------------------
        self.lane_initialized = False
        self.lane_centers = []
        self.cos_a = 1
        self.sin_a = 0

        # ------------------------------
        # 프레임 변화량(파편 감지용)
        # ------------------------------
        self.prev_frame = None
        self.noise_history = []

        # ------------------------------
        # 파라미터 저장
        # ------------------------------
        self.alpha = alpha
        self.accident_hold = accident_hold
        self.iou_stop_hold = iou_stop_hold
        self.init_frames = init_frames
        self.lane_break_threshold = lane_break_threshold
        self.vertical_threshold = vertical_threshold
        self.flow_break_threshold = flow_break_threshold

    # ==========================================
    # 1. IoU 계산 함수
    # ==========================================
    def compute_iou(self, box1, box2):
        """
        두 바운딩 박스의 IoU 계산

        [IoU란?]
        두 박스가 얼마나 겹치는지 나타내는 값
        0이면 안 겹침, 1이면 완전히 겹침

        사고 상황에서는 차량이 충돌 후 서로 겹쳐 보이는 경우가 있어
        중요한 힌트가 된다.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    # ==========================================
    # 2. 차선 이탈량 계산
    # ==========================================
    def get_lane_break(self, track):
        """
        차량 궤적(track)을 보고 옆 방향 이동량을 계산한다.

        [핵심]
        그냥 x좌표 차이만 보면 터널 방향이 기울어진 영상에서 부정확하다.
        그래서 평균 진행 방향(angle)을 기준으로 회전 좌표계를 만든 뒤,
        그 축 기준 옆 이동량을 측정한다.

        결과값이 크면:
        -> 차선이탈 / 비정상 횡이동 가능성
        """
        if len(track) < 10:
            return 0

        x0, y0 = track[0]
        x1, y1 = track[-1]

        xr0 = x0 * self.cos_a + y0 * self.sin_a
        xr1 = x1 * self.cos_a + y1 * self.sin_a

        return abs(xr1 - xr0)

    # ==========================================
    # 3. 파편(프레임 급변) 감지
    # ==========================================
    def detect_fragment(self, frame):
        """
        이전 프레임과 현재 프레임 차이를 이용해
        화면 변화량이 급격히 커졌는지 확인한다.

        [의도]
        사고 직후:
        - 차량 자세가 급격히 바뀌거나
        - 충돌 흔들림, 파편, 화면 변화가 커질 수 있음

        너무 민감하면 오탐이 늘기 때문에
        baseline 대비 상대값으로 판단한다.
        """
        fragment = False

        if self.prev_frame is not None:
            diff = cv2.absdiff(self.prev_frame, frame)
            noise = np.mean(diff)

            self.noise_history.append(noise)
            if len(self.noise_history) > 300:
                self.noise_history.pop(0)

            baseline = np.mean(self.noise_history) if self.noise_history else 0

            if len(self.noise_history) >= 50 and baseline > 0:
                fragment = noise > baseline * 1.5

        self.prev_frame = frame.copy()
        return fragment

    # ==========================================
    # 4. 차선 자동 학습
    # ==========================================
    def learn_lane_direction(self, frame_idx):
        """
        일정 프레임이 지난 뒤, 지금까지 쌓인 궤적을 보고
        도로 진행 방향과 차선 중심을 학습한다.

        [동작 순서]
        1) 여러 차량의 이동 벡터 평균으로 도로 진행 방향 추정
        2) 그 방향으로 좌표를 회전
        3) 회전된 좌표에서 차선 중심을 KMeans로 군집화
        """
        if self.lane_initialized:
            return

        if frame_idx <= self.init_frames:
            return

        directions = []

        for track in self.track_history.values():
            if len(track) >= 10:
                dx = track[-1][0] - track[0][0]
                dy = track[-1][1] - track[0][1]

                if abs(dx) + abs(dy) > 5:
                    directions.append([dx, dy])

        if len(directions) <= 5:
            return

        avg = np.mean(directions, axis=0)
        angle = np.arctan2(avg[1], avg[0])

        self.cos_a = np.cos(angle)
        self.sin_a = np.sin(angle)

        rotated = []
        for track in self.track_history.values():
            for x, y in track:
                xr = x * self.cos_a + y * self.sin_a
                rotated.append(xr)

        if len(rotated) < 4:
            return

        data = np.array(rotated).reshape(-1, 1)
        k = min(4, len(data))

        kmeans = KMeans(n_clusters=k, n_init=10).fit(data)
        self.lane_centers = sorted([c[0] for c in kmeans.cluster_centers_])

        self.lane_initialized = True

    # ==========================================
    # 5. 차선 번호 할당
    # ==========================================
    def assign_lanes(self, boxes):
        """
        각 차량 bbox에 대해 차선 번호를 붙인다.

        차선 학습이 아직 안 되었으면 -1 처리한다.
        """
        for tid, box in boxes.items():
            cx = int((box[0] + box[2]) / 2)
            cy = box[3]

            lane = -1

            if self.lane_initialized and len(self.lane_centers) > 0:
                xr = cx * self.cos_a + cy * self.sin_a
                dists = [abs(xr - c) for c in self.lane_centers]
                lane = dists.index(min(dists)) + 1

            self.lanes[tid] = lane

    # ==========================================
    # 6. 속도 계산
    # ==========================================
    def update_tracks_and_speed(self, tracks):
        """
        tracks를 받아서:
        - 차량 bbox 정리
        - 궤적 저장
        - 픽셀 속도 계산
        를 수행한다.

        [속도 기준]
        기존처럼 바운딩 박스 하단 y2 기준의 변화량을 사용한다.
        터널 영상에서는 y2가 중심점보다 상대적으로 덜 흔들리는 편이라
        V4 기준으로 이 방식이 더 안정적이었다.
        """
        boxes = {}
        speeds = {}

        for obj in tracks:
            tid = int(obj["id"])
            x1, y1, x2, y2 = map(int, obj["bbox"])

            cx = int((x1 + x2) / 2)
            cy = y2

            boxes[tid] = (x1, y1, x2, y2)

            # 궤적 저장
            self.track_history.setdefault(tid, []).append((cx, cy))
            if len(self.track_history[tid]) > 30:
                self.track_history[tid].pop(0)

            # 속도 계산
            speed = 0
            if len(self.track_history[tid]) >= 2:
                dy = self.track_history[tid][-1][1] - self.track_history[tid][-2][1]
                speed = abs(dy)

                # smoothing 적용
                if tid in self.speed_memory:
                    speed = self.alpha * speed + (1 - self.alpha) * self.speed_memory[tid]

                self.speed_memory[tid] = speed

            speeds[tid] = speed

        return boxes, speeds

    # ==========================================
    # 7. 메인 업데이트 함수
    # ==========================================
    def update(self, frame_idx, frame, tracks):
        """
        파이프라인에서 매 프레임마다 호출하는 핵심 함수

        입력:
            frame_idx : 현재 프레임 번호
            frame     : 현재 영상 프레임
            tracks    : 추적 결과 리스트
                       예) [{"id":1, "bbox":[x1,y1,x2,y2]}, ...]

        반환:
            dict 형태의 사고 판단 결과
        """

        # ------------------------------
        # 1) 파편 감지
        # ------------------------------
        fragment = self.detect_fragment(frame)

        # ------------------------------
        # 2) bbox / speed 업데이트
        # ------------------------------
        boxes, speeds = self.update_tracks_and_speed(tracks)

        # ------------------------------
        # 3) 차선 학습 + 차선 할당
        # ------------------------------
        self.learn_lane_direction(frame_idx)
        self.assign_lanes(boxes)

        # ------------------------------
        # 4) 전체 평균 속도
        # ------------------------------
        avg_speed = np.mean(list(speeds.values())) if len(speeds) > 0 else 0

        # ------------------------------
        # 5) flow break 계산
        # 화면 상/하 차량 분포 차이
        # ------------------------------
        h = frame.shape[0]
        upper = 0
        lower = 0

        for tid, box in boxes.items():
            cy = box[3]
            if cy < h / 2:
                upper += 1
            else:
                lower += 1

        total = len(boxes)
        occupancy_diff = abs(upper - lower) / total if total > 0 else 0
        flow_break = occupancy_diff > self.flow_break_threshold

        # ------------------------------
        # 6) 차량쌍 비교
        # ------------------------------
        ids = list(boxes.keys())

        accident_pairs = []
        final_accident = False
        stop_confirm_any = False
        abnormal_any = False

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]

                box1 = boxes[id1]
                box2 = boxes[id2]

                cx1 = int((box1[0] + box1[2]) / 2)
                cy1 = box1[3]
                cx2 = int((box2[0] + box2[2]) / 2)
                cy2 = box2[3]

                dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

                s1 = speeds.get(id1, 0)
                s2 = speeds.get(id2, 0)

                gap = abs(s1 - s2)
                iou = self.compute_iou(box1, box2)

                lane1 = self.lanes.get(id1, -1)
                lane2 = self.lanes.get(id2, -1)
                same_lane = (lane1 == lane2 and lane1 != -1)

                pair_key = f"{id1}-{id2}"

                prev = self.pair_memory.get(pair_key, {
                    "dist": dist,
                    "gap": gap
                })

                # ------------------------------
                # 충돌 전후 변화 특징
                # ------------------------------
                dist_drop = dist < prev["dist"] * 0.6
                gap_up = gap > prev["gap"] * 1.2 or gap > 3
                vertical = abs(cx1 - cx2) < self.vertical_threshold

                lane_break = (
                    self.get_lane_break(self.track_history.get(id1, [])) > self.lane_break_threshold
                    or
                    self.get_lane_break(self.track_history.get(id2, [])) > self.lane_break_threshold
                )

                # ------------------------------
                # IoU 정지 유지 판단
                # ------------------------------
                self.iou_stop_counter.setdefault(pair_key, 0)

                if iou > 0.9:
                    self.iou_stop_counter[pair_key] += 1
                else:
                    self.iou_stop_counter[pair_key] = 0

                stop_confirm = self.iou_stop_counter[pair_key] > self.iou_stop_hold
                stop_confirm_any = stop_confirm_any or stop_confirm

                # ------------------------------
                # 전체 흐름은 빠른데 특정 차량만 느릴 때
                # -> 이상 차량 가능성
                # ------------------------------
                abnormal = ((s1 < 2 or s2 < 2) and avg_speed > 5)
                abnormal_any = abnormal_any or abnormal

                # ------------------------------
                # 사고 패턴 세부 조건
                # ------------------------------
                rear = dist_drop and gap_up and vertical
                side = (iou > 0.3) and gap_up
                lane_break_acc = lane_break and gap_up

                # ------------------------------
                # 최종 사고 의심
                # 기존 로직 최대한 유지
                # ------------------------------
                accident = (
                    same_lane and (
                        rear
                        or side
                        or (iou > 0.9 and stop_confirm)
                        or lane_break_acc
                        or (fragment and abnormal and gap_up)
                    )
                )

                # ------------------------------
                # HOLD 안정화
                # 순간 오탐 방지
                # ------------------------------
                self.accident_counter.setdefault(pair_key, 0)

                if accident:
                    self.accident_counter[pair_key] += 1
                else:
                    self.accident_counter[pair_key] = 0

                final = self.accident_counter[pair_key] > self.accident_hold

                if final:
                    final_accident = True

                # ------------------------------
                # 결과 저장
                # ------------------------------
                accident_pairs.append({
                    "id1": id1,
                    "id2": id2,
                    "cx1": cx1,
                    "cy1": cy1,
                    "cx2": cx2,
                    "cy2": cy2,
                    "dist": round(dist, 2),
                    "s1": round(s1, 2),
                    "s2": round(s2, 2),
                    "gap": round(gap, 2),
                    "iou": round(iou, 3),
                    "lane1": lane1,
                    "lane2": lane2,
                    "prev_dist": round(prev["dist"], 2),
                    "prev_gap": round(prev["gap"], 2),
                    "dist_drop": dist_drop,
                    "gap_up": gap_up,
                    "vertical": vertical,
                    "lane_break": lane_break,
                    "stop_confirm": stop_confirm,
                    "abnormal": abnormal,
                    "rear": rear,
                    "side": side,
                    "lane_break_acc": lane_break_acc,
                    "accident": accident,
                    "final": final
                })

                # 다음 프레임 비교용 갱신
                self.pair_memory[pair_key] = {"dist": dist, "gap": gap}

        # ------------------------------
        # 7) 최종 반환
        # ------------------------------
        result = {
            "accident": final_accident,        # 이번 프레임에서 확정 사고가 있는가
            "flow_break": flow_break,          # 흐름 붕괴 여부
            "fragment": fragment,              # 프레임 급변 여부
            "abnormal": abnormal_any,          # 이상 차량 존재 여부
            "stop_confirm": stop_confirm_any,  # IoU 정지 유지 쌍 존재 여부
            "avg_speed": round(float(avg_speed), 2),
            "vehicle_count": len(boxes),
            "pairs": accident_pairs,           # 차량쌍 상세 로그용
            "boxes": boxes,                    # 디버깅/시각화용
            "speeds": speeds,                  # 디버깅/상태모델 공유용
            "lanes": dict(self.lanes)          # 차선 정보
        }

        return result