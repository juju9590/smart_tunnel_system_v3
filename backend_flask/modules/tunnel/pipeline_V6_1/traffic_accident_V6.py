# ==========================================
# 파일명: traffic_accident_V6.py
# 설명:
# SMART TUNNEL 사고탐지 V6
#
# V6 핵심 방향
# --------------------------------------------------
# 기존 V5.5:
#   pair 조건 중심
#   dist_drop + gap_up + same_lane/vertical
#   후보가 strong_suspect로 바로 이어지고
#   strong_suspect가 accident lock까지 연결됨
#
# 문제:
#   1) 사고 영상에서는 pair 후보가 잘 안 생김
#   2) 혼잡+정체에서는 저속/후보 반복이 사고로 오탐됨
#   3) 평균속도/차량수/정체만으로는 사고와 혼잡을 구분하기 어려움
#
# V6:
#   사고를 "고정 장애물 + 후방 흐름 단절"로 판단한다.
#
# 사고 판단 핵심:
#   1) 특정 위치에 오래 고정된 차량/객체가 있는가?
#   2) bbox jump 이후 같은 위치에 고정되는가?
#   3) 그 고정 지점 뒤쪽에 차량이 쌓이거나 속도가 떨어지는가?
#   4) 차량 전체가 천천히 흐르는 혼잡인지, 특정 지점만 막힌 사고인지 구분한다.
#
# 중요:
#   - 평균속도 낮음 단독으로 사고 판단 금지
#   - 차량 수 많음 단독으로 사고 판단 금지
#   - 혼잡/정체 상태 단독으로 사고 판단 금지
#   - weak candidate 단독으로 accident lock 금지
# ==========================================

from collections import deque, defaultdict
import math
import numpy as np


class AccidentDetector:
    def __init__(self):
        # =====================================================
        # 1) 사고 상태 lock
        # =====================================================
        # 사고가 한 번 확정되면 관제사가 수동 해제하기 전까지 유지한다.
        self.accident_locked = False
        self.accident_start_frame = None

        # =====================================================
        # 2) 차량별 추적 메모리
        # =====================================================
        # track_id별 최근 위치/속도/bbox 정보를 저장한다.
        # V6에서는 track_age가 짧다고 무조건 버리지 않는다.
        # 대신 "짧은 track이라도 같은 위치에 반복적으로 고정되는지"를 본다.
        self.track_memory = {}

        # track_id가 사라진 뒤에도 너무 오래 메모리에 남지 않도록 정리한다.
        self.TRACK_STALE_GAP = 90

        # =====================================================
        # 3) 위치 셀 기반 메모리
        # =====================================================
        # 사고는 특정 위치가 막히는 현상이다.
        # 그래서 화면을 작은 격자 cell로 나누고,
        # 같은 cell에 저속/정지 객체가 반복해서 나타나는지 본다.
        self.cell_stationary_history = defaultdict(deque)

        # cell 크기. 너무 작으면 흔들림에 약하고, 너무 크면 구분이 둔해진다.
        self.CELL_SIZE = 60

        # =====================================================
        # 4) 사고 후보 history
        # =====================================================
        # 한 프레임에서 사고처럼 보여도 바로 lock하지 않는다.
        # 여러 프레임 동안 반복될 때만 사고 의심/확정으로 간다.
        self.frame_candidate_history = deque()
        self.strong_candidate_history = deque()
        self.weak_candidate_history = deque()
        self.strong_reason_history = deque()
        self.pair_collision_valid_history = deque()

        # 대형 차량이 화면에 들어오면 작은 차량 bbox가 순간적으로 가려져
        # detector의 차량 수가 줄어들 수 있다. 이 감소량은 사고 증거가 아니라
        # 혼잡/정체 영상의 가림 패턴일 수 있으므로 별도 buffer로 추적한다.
        self.vehicle_count_history = deque(maxlen=120)
        self.prev_vehicle_count = None

        # =====================================================
        # 5) 임계값 - 움직임/고정성
        # =====================================================
        # 차량 bottom-center 이동량이 이 값 이하이면 거의 고정으로 본다.
        self.STATIONARY_MOVE_THR = 4.0

        # 속도가 이 값 이하이면 저속/정지 후보로 본다.
        # 교통상태의 혼잡 기준과 사고판단의 정지 기준은 분리해야 한다.
        self.STOP_SPEED_THR = 1.4

        # 약한 저속 기준.
        # 사고 후 후방 정체 판단에는 조금 더 넓은 저속 기준을 사용한다.
        self.SLOW_SPEED_THR = 2.2

        # 같은 track이 최소 몇 프레임 이상 보여야 고정 판단에 의미를 둘지.
        self.MIN_TRACK_AGE_FOR_FIXED = 6

        # 몇 프레임 이상 거의 움직이지 않으면 고정 장애물 후보인지.
        self.FIXED_HOLD_FRAMES = 8

        # bbox jump 이후 몇 프레임 안에 고정되면 사고 증거로 볼지.
        self.JUMP_RECENT_WINDOW = 20

        # bbox 중심 이동이 이 값 이상이면 jump로 본다.
        # 단, V6에서는 jump를 무조건 제거하지 않고,
        # jump 후 고정이면 사고 증거로 사용한다.
        self.CENTER_JUMP_ABS_THR = 75.0

        # bbox 면적 변화가 이 배율 이상이면 bbox jump로 본다.
        self.AREA_JUMP_RATIO_THR = 2.8

        # =====================================================
        # 6) 임계값 - 공간/흐름 단절
        # =====================================================
        # 고정 장애물 뒤쪽에 차량이 몇 대 이상 있으면 후방 누적으로 본다.
        self.REAR_QUEUE_COUNT_THR = 1

        # 고정 지점 주변 반경
        self.OBSTACLE_NEAR_RADIUS = 120.0

        # 고정 장애물 후보 최소 개수
        self.MIN_FIXED_OBSTACLE_COUNT = 1

        # 사고 판단에 필요한 최소 차량 수
        self.MIN_VEHICLE_COUNT_FOR_ACCIDENT = 2

        # 혼잡 방어:
        # 정지/고정 차량 비율이 너무 높으면 특정 장애물이 아니라
        # 전체 정체일 가능성이 크다.
        self.CONGESTION_FIXED_RATIO_THR = 0.65

        # =====================================================
        # 7) 임계값 - 후보 누적
        # =====================================================
        # 후보 누적 window
        self.CANDIDATE_WINDOW = 120

        # strong 후보 확정 반복 수
        self.STRONG_CONFIRM_COUNT = 3

        # weak 후보는 바로 lock하지 않는다.
        # 팝업/의심 단계 확인용으로만 사용한다.
        self.WEAK_CONFIRM_COUNT = 8

        # 최종 사고 lock에 필요한 frame candidate 반복 수
        self.ACCIDENT_CONFIRM_COUNT = 4

        # pair_collision_valid 반복 사고 보강:
        # 사매터널처럼 bbox jump/sudden stop은 약하지만 충돌성 pair가
        # 일정 구간에 반복되는 사고 영상을 확정 후보 누적에 포함한다.
        self.PAIR_REPEAT_WINDOW = 150
        self.PAIR_REPEAT_CONFIRM_COUNT = 5

        # =====================================================
        # 8) pair 충돌 보조 로직
        # =====================================================
        # V6에서도 pair 로직을 완전히 없애지는 않는다.
        # 다만 사고 판단의 중심이 아니라 보조 증거로만 사용한다.
        self.pair_memory = {}
        self.PAIR_DIST_DROP_RATIO = 0.65
        self.PAIR_GAP_UP_THR = 4.0
        self.PAIR_NEAR_DIST_THR = 65.0

        # =====================================================
        # 8-1) V6_1 실험: 원거리 작은 bbox pair 방어
        # =====================================================
        # 화면 상단 원거리 영역의 작은 bbox 두 개가 가까워 보이면
        # 실제 충돌이 아니라 중복탐지/라이트/반사광일 수 있다.
        # 큰 bbox나 중하단/근거리 pair는 기존 사고 근거를 보호하기 위해
        # 이 방어에서 제외한다.
        self.SMALL_PAIR_BBOX_HEIGHT_THR = 45
        self.SMALL_PAIR_BBOX_AREA_THR = 2500
        self.FAR_REGION_Y_RATIO = 0.45

        # =====================================================
        # 9) 디버그 정보
        # =====================================================
        self.last_debug = {
            "frame_id": 0,
            "accident": False,
            "accident_locked": False,
            "accident_candidate_only": False,
            "accident_score": 0,
            "reasons": [],
            "weak_suspect": False,
            "strong_suspect": False,
            "confirm_candidate": False,
        }

    # =========================================================
    # 외부 수동 해제
    # =========================================================
    def clear_accident(self):
        """
        관제사가 사고 해제 버튼을 눌렀을 때 호출한다.
        """
        self.accident_locked = False
        self.accident_start_frame = None

        self.track_memory.clear()
        self.cell_stationary_history.clear()
        self.pair_memory.clear()
        self.frame_candidate_history.clear()
        self.strong_candidate_history.clear()
        self.weak_candidate_history.clear()
        self.strong_reason_history.clear()
        self.pair_collision_valid_history.clear()
        self.vehicle_count_history.clear()
        self.prev_vehicle_count = None

        self.last_debug = {
            "frame_id": 0,
            "accident": False,
            "acc_ratio": 0.0,
            "frame_accident_prediction": False,
            "recent_prediction_count": 0,
            "accident_locked": False,
            "accident_candidate_only": False,
            "accident_score": 0,
            "reasons": ["manual_clear"],
            "weak_suspect": False,
            "strong_suspect": False,
            "confirm_candidate": False,
            "strong_candidate": False,
            "weak_candidate": False,
            "strong_confirmed": False,
            "weak_confirmed": False,
            "frame_confirmed": False,
            "has_real_accident_evidence": False,
            "has_final_accident_evidence": False,
            "final_accumulation_blocked": False,
            "history_info": {
                "frame_candidate_count": 0,
                "strong_candidate_count": 0,
                "weak_candidate_count": 0,
            },
        }

        return True

    # =========================================================
    # 기본 유틸
    # =========================================================
    def _center_bottom(self, box):
        """
        bbox에서 bottom-center 좌표를 구한다.
        터널 CCTV에서는 차량의 실제 도로상 위치를 bbox 중심보다
        bottom-center가 더 잘 대표하는 경우가 많다.
        """
        x1, y1, x2, y2 = box
        cx = float((x1 + x2) / 2.0)
        by = float(y2)
        return cx, by

    def _bbox_area(self, box):
        x1, y1, x2, y2 = box
        return float(max(x2 - x1, 1) * max(y2 - y1, 1))

    def _dist(self, p1, p2):
        return float(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

    def _cell_key(self, cx, by):
        """
        화면을 cell 단위로 나눈다.
        같은 위치에 정지 객체가 반복적으로 나타나는지 확인하기 위해 사용한다.
        """
        return int(cx // self.CELL_SIZE), int(by // self.CELL_SIZE)

    def _cleanup_old_tracks(self, frame_id, current_ids):
        """
        오래 사라진 track 메모리를 제거한다.
        """
        stale_ids = []

        for tid, mem in self.track_memory.items():
            last_frame = mem.get("last_frame", frame_id)
            if tid not in current_ids and frame_id - last_frame > self.TRACK_STALE_GAP:
                stale_ids.append(tid)

        for tid in stale_ids:
            self.track_memory.pop(tid, None)

    def _cleanup_cell_history(self, frame_id):
        """
        cell별 정지 history도 오래된 기록은 제거한다.
        """
        stale_cells = []

        for key, dq in self.cell_stationary_history.items():
            while dq and frame_id - dq[0] > self.CANDIDATE_WINDOW:
                dq.popleft()

            if not dq:
                stale_cells.append(key)

        for key in stale_cells:
            self.cell_stationary_history.pop(key, None)

    # =========================================================
    # 차량별 상태 업데이트
    # =========================================================
    def _update_track_memory(self, frame_id, tracks, analysis):
        """
        각 차량의 이동량, 정지 지속, bbox jump 여부를 계산한다.

        반환:
            vehicle_infos = [
                {
                    "id": tid,
                    "box": box,
                    "cx": cx,
                    "by": by,
                    "speed": speed,
                    "move": move,
                    "age": age,
                    "stationary_count": ...,
                    "is_stationary": bool,
                    "is_fixed_obstacle": bool,
                    "jump": bool,
                    "jump_then_fixed": bool,
                    "cell": (cx_cell, by_cell),
                    ...
                }
            ]
        """
        boxes = analysis.get("boxes", {})
        speeds = analysis.get("speeds", {})

        current_ids = set()
        vehicle_infos = []

        for t in tracks:
            tid = t.get("id")
            if tid is None:
                continue

            box = boxes.get(tid, t.get("bbox"))
            if box is None:
                continue

            current_ids.add(tid)

            x1, y1, x2, y2 = box
            box = (float(x1), float(y1), float(x2), float(y2))

            cx, by = self._center_bottom(box)
            area = self._bbox_area(box)
            speed = float(speeds.get(tid, 0.0))
            cell = self._cell_key(cx, by)

            prev = self.track_memory.get(tid)

            # ----------------------------------------------
            # 최초 등장 차량
            # ----------------------------------------------
            if prev is None:
                move = 0.0
                age = 1
                stationary_count = 1 if speed <= self.STOP_SPEED_THR else 0
                jump = False
                jump_recent_frames = deque(maxlen=self.JUMP_RECENT_WINDOW)

            # ----------------------------------------------
            # 기존 차량
            # ----------------------------------------------
            else:
                prev_pos = prev.get("pos", (cx, by))
                prev_area = max(float(prev.get("area", area)), 1.0)

                move = self._dist((cx, by), prev_pos)
                age = int(prev.get("age", 0)) + 1

                # bbox 중심 이동 jump
                center_jump = move >= self.CENTER_JUMP_ABS_THR

                # bbox 면적 변화 jump
                area_ratio = max(area / prev_area, prev_area / max(area, 1.0))
                area_jump = area_ratio >= self.AREA_JUMP_RATIO_THR

                jump = bool(center_jump or area_jump)

                prev_jump_recent = prev.get("jump_recent_frames", deque(maxlen=self.JUMP_RECENT_WINDOW))
                jump_recent_frames = deque(prev_jump_recent, maxlen=self.JUMP_RECENT_WINDOW)

                if jump:
                    jump_recent_frames.append(frame_id)

                # 정지/저속 판단
                # speed가 낮거나 bottom-center 이동량이 아주 작으면 stationary로 본다.
                is_current_stationary = (
                    speed <= self.STOP_SPEED_THR
                    or move <= self.STATIONARY_MOVE_THR
                )

                if is_current_stationary:
                    stationary_count = int(prev.get("stationary_count", 0)) + 1
                else:
                    stationary_count = 0

            # ----------------------------------------------
            # 현재 차량 상태 판정
            # ----------------------------------------------
            is_stationary = (
                speed <= self.STOP_SPEED_THR
                or move <= self.STATIONARY_MOVE_THR
            )

            is_fixed_obstacle = (
                age >= self.MIN_TRACK_AGE_FOR_FIXED
                and stationary_count >= self.FIXED_HOLD_FRAMES
            )

            # bbox jump가 최근에 있었고, 이후 고정되면 사고의 중요한 증거
            jump_then_fixed = (
                len(jump_recent_frames) > 0
                and frame_id - jump_recent_frames[-1] <= self.JUMP_RECENT_WINDOW
                and stationary_count >= max(4, self.FIXED_HOLD_FRAMES // 2)
            )

            # cell 기반 정지 history 기록
            if is_stationary:
                self.cell_stationary_history[cell].append(frame_id)

            info = {
                "id": tid,
                "box": box,
                "cx": cx,
                "by": by,
                "area": area,
                "speed": speed,
                "move": move,
                "age": age,
                "cell": cell,
                "stationary_count": stationary_count,
                "is_stationary": bool(is_stationary),
                "is_fixed_obstacle": bool(is_fixed_obstacle),
                "jump": bool(jump),
                "jump_then_fixed": bool(jump_then_fixed),
            }

            vehicle_infos.append(info)

            # 메모리 갱신
            self.track_memory[tid] = {
                "pos": (cx, by),
                "area": area,
                "age": age,
                "stationary_count": stationary_count,
                "last_frame": frame_id,
                "jump_recent_frames": jump_recent_frames,
            }

        self._cleanup_old_tracks(frame_id, current_ids)
        self._cleanup_cell_history(frame_id)

        return vehicle_infos

    # =========================================================
    # pair 충돌 후보 분석
    # =========================================================
    def _analyze_pair_collision(self, frame_id, vehicle_infos, frame_height):
        """
        기존 pair 기반 사고 후보를 보조 증거로 사용한다.

        V6에서는 same_lane에 강하게 의존하지 않는다.
        이유:
            - 구봉처럼 차선 추정이 흔들리면 same_lane 조건이 사고 후보를 막을 수 있음
            - 따라서 거리 급감, 속도 gap, 근접성 중심으로만 보조 판단한다.
        """
        pair_collision_count = 0
        small_far_pair_count = 0
        pair_debug = []
        small_far_pair_debug = {
            "defense_small_far_pair": False,
            "small_bbox_pair": False,
            "small_area_pair": False,
            "far_region_pair": False,
            "pair_bbox_h1": 0.0,
            "pair_bbox_h2": 0.0,
            "pair_bbox_area1": 0.0,
            "pair_bbox_area2": 0.0,
            "pair_bottom_y": 0.0,
        }

        for i in range(len(vehicle_infos)):
            for j in range(i + 1, len(vehicle_infos)):
                a = vehicle_infos[i]
                b = vehicle_infos[j]

                id1 = a["id"]
                id2 = b["id"]
                key = tuple(sorted((id1, id2)))

                p1 = (a["cx"], a["by"])
                p2 = (b["cx"], b["by"])

                dist = self._dist(p1, p2)
                gap = abs(float(a["speed"]) - float(b["speed"]))

                prev = self.pair_memory.get(key, {"dist": dist, "gap": gap})

                dist_drop = dist < float(prev["dist"]) * self.PAIR_DIST_DROP_RATIO
                gap_up = gap >= self.PAIR_GAP_UP_THR
                near = dist <= self.PAIR_NEAR_DIST_THR

                # 충돌성 후보:
                # 거리 급감 + 속도 차이 + 근접성 중 2개 이상
                evidence_count = int(dist_drop) + int(gap_up) + int(near)

                pair_collision = evidence_count >= 2

                ax1, ay1, ax2, ay2 = a["box"]
                bx1, by1, bx2, by2 = b["box"]
                bbox_h1 = max(float(ay2 - ay1), 1.0)
                bbox_h2 = max(float(by2 - by1), 1.0)
                bbox_area1 = float(a.get("area", bbox_h1))
                bbox_area2 = float(b.get("area", bbox_h2))
                pair_bottom_y = max(float(a["by"]), float(b["by"]))

                small_bbox_pair = (
                    bbox_h1 < self.SMALL_PAIR_BBOX_HEIGHT_THR
                    and bbox_h2 < self.SMALL_PAIR_BBOX_HEIGHT_THR
                )

                small_area_pair = (
                    bbox_area1 < self.SMALL_PAIR_BBOX_AREA_THR
                    and bbox_area2 < self.SMALL_PAIR_BBOX_AREA_THR
                )

                far_region_pair = (
                    frame_height > 0
                    and pair_bottom_y < float(frame_height) * self.FAR_REGION_Y_RATIO
                )

                small_far_pair = (
                    pair_collision
                    and far_region_pair
                    and (small_bbox_pair or small_area_pair)
                )

                if pair_collision:
                    pair_collision_count += 1

                if small_far_pair:
                    small_far_pair_count += 1
                    if not small_far_pair_debug["defense_small_far_pair"]:
                        small_far_pair_debug = {
                            "defense_small_far_pair": True,
                            "small_bbox_pair": bool(small_bbox_pair),
                            "small_area_pair": bool(small_area_pair),
                            "far_region_pair": bool(far_region_pair),
                            "pair_bbox_h1": round(bbox_h1, 2),
                            "pair_bbox_h2": round(bbox_h2, 2),
                            "pair_bbox_area1": round(bbox_area1, 2),
                            "pair_bbox_area2": round(bbox_area2, 2),
                            "pair_bottom_y": round(pair_bottom_y, 2),
                        }

                self.pair_memory[key] = {
                    "dist": dist,
                    "gap": gap,
                    "last_frame": frame_id,
                }

                pair_debug.append({
                    "pair": key,
                    "dist": round(dist, 2),
                    "gap": round(gap, 2),
                    "dist_drop": bool(dist_drop),
                    "gap_up": bool(gap_up),
                    "near": bool(near),
                    "pair_collision": bool(pair_collision),
                    "small_bbox_pair": bool(small_bbox_pair),
                    "small_area_pair": bool(small_area_pair),
                    "far_region_pair": bool(far_region_pair),
                    "small_far_pair": bool(small_far_pair),
                    "bbox_h1": round(bbox_h1, 2),
                    "bbox_h2": round(bbox_h2, 2),
                    "bbox_area1": round(bbox_area1, 2),
                    "bbox_area2": round(bbox_area2, 2),
                    "pair_bottom_y": round(pair_bottom_y, 2),
                })

        # 오래된 pair 제거
        stale_keys = []
        for key, mem in self.pair_memory.items():
            last_frame = mem.get("last_frame", frame_id)
            if frame_id - last_frame > self.TRACK_STALE_GAP:
                stale_keys.append(key)

        for key in stale_keys:
            self.pair_memory.pop(key, None)

        return pair_collision_count, pair_debug, small_far_pair_count, small_far_pair_debug

    # =========================================================
    # 고정 장애물 / 후방 누적 / 혼잡 방어 분석
    # =========================================================
    def _analyze_obstacle_pattern(self, vehicle_infos, avg_speed):
        """
        사고와 혼잡을 가르는 핵심 분석.

        혼잡:
            전체 차량이 느리게 움직이거나 멈춰 있음.
            특정한 고정 장애물 하나 때문에 흐름이 끊긴 것이 아님.

        사고:
            특정 위치에 고정 장애물이 있고,
            그 지점 뒤로 차량이 쌓이거나 흐름이 끊김.
        """
        vehicle_count = len(vehicle_infos)

        fixed_obstacles = [
            v for v in vehicle_infos
            if v["is_fixed_obstacle"]
        ]

        stationary_vehicles = [
            v for v in vehicle_infos
            if v["is_stationary"]
        ]

        jump_fixed = [
            v for v in vehicle_infos
            if v["jump_then_fixed"]
        ]

        fixed_count = len(fixed_obstacles)
        stationary_count = len(stationary_vehicles)
        jump_fixed_count = len(jump_fixed)

        fixed_ratio = fixed_count / vehicle_count if vehicle_count > 0 else 0.0
        stationary_ratio = stationary_count / vehicle_count if vehicle_count > 0 else 0.0

        # -----------------------------------------------------
        # 혼잡 방어 조건
        # -----------------------------------------------------
        # 전체 차량 대부분이 느리거나 고정되어 있으면
        # 특정 사고 지점이 아니라 전체 정체일 수 있다.
        congestion_like = (
            vehicle_count >= 4
            and stationary_ratio >= self.CONGESTION_FIXED_RATIO_THR
            and jump_fixed_count == 0
        )

        # -----------------------------------------------------
        # localized obstacle 판단
        # -----------------------------------------------------
        # 고정 객체가 너무 많으면 전체 정체일 가능성이 있고,
        # 1~2개의 고정 객체가 흐름을 막고 있으면 사고 가능성이 높다.
        localized_obstacle = (
            fixed_count >= self.MIN_FIXED_OBSTACLE_COUNT
            and not congestion_like
        )

        # -----------------------------------------------------
        # 후방 누적 판단
        # -----------------------------------------------------
        # 터널 CCTV에서는 보통 화면 아래쪽이 카메라에 가까운 후방인 경우가 많다.
        # 따라서 고정 장애물보다 by가 더 큰 차량을 후방 차량으로 본다.
        # 단, 모든 영상에서 완벽한 방향 정보는 아니므로 보조 증거로만 사용한다.
        rear_queue_count = 0

        for obs in fixed_obstacles:
            obs_by = obs["by"]

            rear_vehicles = [
                v for v in vehicle_infos
                if v["id"] != obs["id"]
                and v["by"] > obs_by + 20
            ]

            if len(rear_vehicles) > rear_queue_count:
                rear_queue_count = len(rear_vehicles)

        rear_queue = rear_queue_count >= self.REAR_QUEUE_COUNT_THR

        # -----------------------------------------------------
        # 흐름 대비 판단
        # -----------------------------------------------------
        # 사고는 고정 지점과 움직이는 차량/후방 차량이 섞이는 경우가 많다.
        # 전체가 다 멈춘 혼잡과 구분하기 위해 사용한다.
        moving_count = len([
            v for v in vehicle_infos
            if not v["is_stationary"]
        ])

        flow_contrast = (
            fixed_count >= 1
            and moving_count >= 1
        )

        return {
            "fixed_count": fixed_count,
            "stationary_count": stationary_count,
            "jump_fixed_count": jump_fixed_count,
            "fixed_ratio": round(fixed_ratio, 4),
            "stationary_ratio": round(stationary_ratio, 4),
            "localized_obstacle": bool(localized_obstacle),
            "rear_queue": bool(rear_queue),
            "rear_queue_count": rear_queue_count,
            "flow_contrast": bool(flow_contrast),
            "congestion_like": bool(congestion_like),
            "avg_speed": round(float(avg_speed), 4),
        }

    # =========================================================
    # cell 고정성 분석
    # =========================================================
    def _analyze_cell_persistence(self, frame_id):
        """
        같은 위치 cell에 정지 객체가 반복적으로 나타나는지 확인한다.

        track_id가 바뀌어도 같은 위치에 계속 객체가 잡히면
        사고로 인한 고정 장애물일 가능성이 있다.
        """
        max_cell_count = 0
        hot_cell = None

        for key, dq in self.cell_stationary_history.items():
            recent = [f for f in dq if frame_id - f <= self.CANDIDATE_WINDOW]
            count = len(recent)

            if count > max_cell_count:
                max_cell_count = count
                hot_cell = key

        cell_persistent = max_cell_count >= 10

        return {
            "cell_persistent": bool(cell_persistent),
            "hot_cell": hot_cell,
            "hot_cell_count": max_cell_count,
        }

    # =========================================================
    # 사고 점수 계산
    # =========================================================
    def _score_accident_candidate(
        self,
        vehicle_count,
        obstacle_info,
        cell_info,
        pair_collision_count,
        smoke_fire=False,
    ):
        """
        사고 후보 점수 계산.

        중요한 원칙:
            - 평균속도 낮음은 점수를 거의 주지 않는다.
            - 차량 수 많음도 점수를 주지 않는다.
            - 고정 장애물, jump 후 고정, 후방 누적, 연기/화재가 핵심이다.
        """
        score = 0
        reasons = []

        # -----------------------------------------------------
        # 1) 특정 위치 고정 장애물
        # -----------------------------------------------------
        if obstacle_info["localized_obstacle"]:
            score += 3
            reasons.append("localized_fixed_obstacle")

        # -----------------------------------------------------
        # 2) bbox jump 이후 고정
        # -----------------------------------------------------
        if obstacle_info["jump_fixed_count"] >= 1:
            score += 2
            reasons.append("bbox_jump_then_fixed")
            # 사고 확정에는 "갑자기 멈춘 뒤 고정됨"처럼 사고성이 있는
            # 직접 증거가 필요하다. 기존 jump_fixed 신호를 확정용 reason으로
            # 함께 남겨 localized/cell 단독 확정을 막는다.
            reasons.append("sudden_stop_after_moving")

        # -----------------------------------------------------
        # 3) 후방 차량 누적
        # -----------------------------------------------------
        if obstacle_info["rear_queue"]:
            score += 2
            reasons.append("rear_queue_after_obstacle")

        # -----------------------------------------------------
        # 4) 고정 지점과 이동 흐름의 대비
        # -----------------------------------------------------
        if obstacle_info["flow_contrast"]:
            score += 1
            reasons.append("flow_contrast")

        # -----------------------------------------------------
        # 5) cell 기반 위치 지속성
        # -----------------------------------------------------
        if cell_info["cell_persistent"]:
            score += 2
            reasons.append("persistent_stationary_cell")

        # -----------------------------------------------------
        # 6) pair 충돌 후보
        # -----------------------------------------------------
        if pair_collision_count >= 1:
            score += 2
            reasons.append("pair_collision_evidence")

        # -----------------------------------------------------
        # 7) 연기/화재
        # 현재 pipeline에서 smoke_fire_map이 들어오면 확장 가능.
        # 아직 없으면 False로 유지된다.
        # -----------------------------------------------------
        if smoke_fire:
            score += 3
            reasons.append("smoke_or_fire")

        # -----------------------------------------------------
        # 8) 혼잡 방어
        # -----------------------------------------------------
        # 전체 정체처럼 보이면 강하게 감점한다.
        # 단, jump_fixed나 smoke_fire가 있으면 사고 가능성이 있으므로
        # 너무 강하게 막지는 않는다.
        if obstacle_info["congestion_like"]:
            score -= 4
            reasons.append("defense_congestion_like")

        # -----------------------------------------------------
        # 9) 차량 수 부족 방어
        # -----------------------------------------------------
        if vehicle_count < self.MIN_VEHICLE_COUNT_FOR_ACCIDENT:
            score -= 3
            reasons.append("defense_too_few_vehicles")

        return score, reasons

    # =========================================================
    # 후보 history 업데이트
    # =========================================================
    def _update_histories(
        self,
        frame_id,
        score,
        strong_candidate,
        weak_candidate,
        confirm_candidate,
        reasons,
    ):
        """
        strong/weak/confirm 후보를 각각 누적한다.

        strong:
            accident lock까지 갈 수 있는 후보

        weak:
            사고 의심 팝업 후보
            단독으로 accident lock은 금지

        confirm:
            실제 사고 확정 누적에 들어가는 후보.
            weak_suspect는 이 history에 절대 넣지 않는다.
        """
        if confirm_candidate:
            self.frame_candidate_history.append(frame_id)

        if strong_candidate:
            self.strong_candidate_history.append(frame_id)
            self.strong_reason_history.append({
                "frame_id": frame_id,
                "reasons": list(reasons),
            })

        if weak_candidate:
            self.weak_candidate_history.append(frame_id)

        # 오래된 기록 제거
        for dq in [
            self.frame_candidate_history,
            self.strong_candidate_history,
            self.weak_candidate_history,
        ]:
            while dq and frame_id - dq[0] > self.CANDIDATE_WINDOW:
                dq.popleft()

        while (
            self.strong_reason_history
            and frame_id - self.strong_reason_history[0]["frame_id"] > self.CANDIDATE_WINDOW
        ):
            self.strong_reason_history.popleft()

        return {
            "frame_candidate_count": len(self.frame_candidate_history),
            "strong_candidate_count": len(self.strong_candidate_history),
            "weak_candidate_count": len(self.weak_candidate_history),
        }

    # =========================================================
    # 메인 update
    # =========================================================
    def update(self, frame_id, tracks, analysis):
        """
        PipelineCore에서 매 프레임 호출하는 사고 판단 함수.

        입력:
            frame_id : 현재 프레임 번호
            tracks   : [{"id": tid, "bbox": (...)}, ...]
            analysis : TrackAnalyzer + LaneTemplate 결과 dict

        출력:
            기존 V5.5와 호환되는 dict 구조 유지
        """

        boxes = analysis.get("boxes", {})
        avg_speed = float(analysis.get("avg_speed", 0.0))
        vehicle_count = int(analysis.get("vehicle_count", len(tracks)))
        traffic_state = str(analysis.get("traffic_state", "NORMAL"))
        frame_height = int(analysis.get("frame_height", 0) or 0)

        # -----------------------------------------------------
        # 1) 차량별 이동/고정/jump 상태 계산
        # -----------------------------------------------------
        vehicle_infos = self._update_track_memory(
            frame_id=frame_id,
            tracks=tracks,
            analysis=analysis,
        )

        # 실제 분석 가능한 차량 수
        visible_vehicle_count = len(vehicle_infos)

        # -----------------------------------------------------
        # 차량 수 감소량 계산
        # -----------------------------------------------------
        # 혼잡/정체에서 대형 차량이 진입하면 주변 차량 bbox가 가려지며
        # detector 기준 vehicle_count가 갑자기 줄어들 수 있다.
        # 직전 프레임과 최근 평균 중 더 큰 감소량을 사용해 방어 신호로 쓴다.
        if self.prev_vehicle_count is None:
            prev_vehicle_count = vehicle_count
        else:
            prev_vehicle_count = int(self.prev_vehicle_count)

        if len(self.vehicle_count_history) > 0:
            recent_vehicle_avg = float(np.mean(self.vehicle_count_history))
        else:
            recent_vehicle_avg = float(prev_vehicle_count)

        vehicle_drop_from_prev = max(0, prev_vehicle_count - vehicle_count)
        vehicle_drop_from_avg = max(0, int(round(recent_vehicle_avg - vehicle_count)))
        vehicle_drop = max(vehicle_drop_from_prev, vehicle_drop_from_avg)

        # -----------------------------------------------------
        # 2) pair 충돌 후보 분석
        # -----------------------------------------------------
        pair_collision_count, pair_debug, small_far_pair_count, small_far_pair_debug = self._analyze_pair_collision(
            frame_id=frame_id,
            vehicle_infos=vehicle_infos,
            frame_height=frame_height,
        )

        # -----------------------------------------------------
        # 3) 고정 장애물 / 후방 누적 / 혼잡 방어 분석
        # -----------------------------------------------------
        obstacle_info = self._analyze_obstacle_pattern(
            vehicle_infos=vehicle_infos,
            avg_speed=avg_speed,
        )

        # -----------------------------------------------------
        # 4) cell 위치 지속성 분석
        # -----------------------------------------------------
        cell_info = self._analyze_cell_persistence(frame_id)

        # -----------------------------------------------------
        # 5) smoke/fire 보조 증거
        # -----------------------------------------------------
        # 현재 analysis에 smoke_fire_map이 있다면 사용 가능.
        # 지금 없더라도 코드가 깨지지 않도록 안전 처리.
        smoke_fire_map = analysis.get("smoke_fire_map", {})
        smoke_fire = False

        if isinstance(smoke_fire_map, dict) and len(smoke_fire_map) > 0:
            smoke_fire = any(bool(v) for v in smoke_fire_map.values())

        # -----------------------------------------------------
        # 6) 사고 후보 점수 계산
        # -----------------------------------------------------
        score, reasons = self._score_accident_candidate(
            vehicle_count=visible_vehicle_count,
            obstacle_info=obstacle_info,
            cell_info=cell_info,
            pair_collision_count=pair_collision_count,
            smoke_fire=smoke_fire,
        )

        pair_evidence_raw = "pair_collision_evidence" in reasons
        sudden_stop_reason = "sudden_stop_after_moving" in reasons
        bbox_jump_then_fixed_reason = "bbox_jump_then_fixed" in reasons
        localized_fixed_reason = "localized_fixed_obstacle" in reasons
        rear_queue_reason = "rear_queue_after_obstacle" in reasons
        persistent_cell_reason = "persistent_stationary_cell" in reasons
        flow_contrast_reason = "flow_contrast" in reasons

        # -----------------------------------------------------
        # 6-1) bbox 가림에 의한 가짜 sudden stop 방어
        # -----------------------------------------------------
        # 혼잡/정체에서 대형차가 들어오면 bbox가 크게 흔들리거나 ID가
        # 바뀌면서 "jump 후 고정"처럼 보일 수 있다. 이 경우의
        # sudden_stop_after_moving은 사고 증거가 아니라 가림/추적 흔들림으로 본다.
        false_stop_by_occlusion = (
            traffic_state in ["JAM", "CONGESTION"]
            and sudden_stop_reason
            and bbox_jump_then_fixed_reason
            and vehicle_drop >= 1
            and not pair_evidence_raw
        )

        jam_bbox_obstacle_combo = (
            traffic_state in ["JAM", "CONGESTION"]
            and localized_fixed_reason
            and bbox_jump_then_fixed_reason
            and rear_queue_reason
            and persistent_cell_reason
            and not pair_evidence_raw
        )

        if false_stop_by_occlusion or jam_bbox_obstacle_combo:
            score -= 5
            if "defense_bbox_occlusion_false_stop" not in reasons:
                reasons.append("defense_bbox_occlusion_false_stop")

        # sudden stop은 위의 가짜 급정지 방어를 통과한 경우에만
        # real evidence 후보로 인정한다.
        sudden_stop_valid = (
            sudden_stop_reason
            and not false_stop_by_occlusion
        )

        defense_small_far_pair = (
            pair_evidence_raw
            and pair_collision_count > 0
            and small_far_pair_count == pair_collision_count
            and not sudden_stop_valid
        )

        if defense_small_far_pair:
            score -= 4
            if "defense_small_far_pair" not in reasons:
                reasons.append("defense_small_far_pair")

        # -----------------------------------------------------
        # 6-2) 혼잡/정체 pair false positive 방어
        # -----------------------------------------------------
        # JAM/CONGESTION에서는 차량 간격이 원래 좁고 bbox가 겹쳐 보여
        # pair_collision_evidence가 쉽게 발생한다. localized/cell/flow 조합만
        # 같이 있는 pair는 real evidence로 인정하지 않는다.
        pair_false_in_congestion = (
            traffic_state in ["JAM", "CONGESTION"]
            and pair_evidence_raw
            and localized_fixed_reason
            and persistent_cell_reason
            and flow_contrast_reason
            and not sudden_stop_reason
        )

        if pair_false_in_congestion:
            score -= 5
            if "defense_pair_false_in_congestion" not in reasons:
                reasons.append("defense_pair_false_in_congestion")

        if traffic_state in ["JAM", "CONGESTION"]:
            pair_collision_valid = (
                pair_evidence_raw
                and sudden_stop_valid
                and not false_stop_by_occlusion
                and not pair_false_in_congestion
                and not defense_small_far_pair
            )
        else:
            pair_collision_valid = (
                pair_evidence_raw
                and not defense_small_far_pair
            )

        real_evidence_in_frame = (
            pair_collision_valid
            or sudden_stop_valid
        )

        # -----------------------------------------------------
        # 6-3) JAM / CONGESTION 정체 방어
        # -----------------------------------------------------
        # localized_fixed_obstacle + persistent_stationary_cell 조합은
        # 사고 영상뿐 아니라 정체 영상에서도 자연스럽게 반복된다.
        # pair 충돌이나 신뢰 가능한 sudden stop 증거가 없으면 감점한다.

        defense_jam_stationary_cell = (
            traffic_state in ["JAM", "CONGESTION"]
            and "localized_fixed_obstacle" in reasons
            and "persistent_stationary_cell" in reasons
            and not real_evidence_in_frame
        )

        if defense_jam_stationary_cell:
            score -= 4
            reasons.append("defense_jam_stationary_cell")

        # -----------------------------------------------------
        # 6-4) 대형차/가림 방어
        # -----------------------------------------------------
        # 저속 정체 상태에서 차량 수가 줄고 bbox jump + 고정 장애물처럼
        # 보이면, sudden_stop_after_moving이 있더라도 대형차 가림을 우선한다.
        defense_large_vehicle_occlusion = (
            traffic_state in ["JAM", "CONGESTION"]
            and not pair_collision_valid
            and (
                (
                    vehicle_drop >= 1
                    and bbox_jump_then_fixed_reason
                    and localized_fixed_reason
                )
                or false_stop_by_occlusion
                or jam_bbox_obstacle_combo
                or (
                    vehicle_drop >= 2
                    and avg_speed <= 3.5
                    and not real_evidence_in_frame
                )
            )
        )

        if defense_large_vehicle_occlusion:
            score -= 5
            if "defense_large_vehicle_occlusion" not in reasons:
                reasons.append("defense_large_vehicle_occlusion")

        # defense_large_vehicle_occlusion이 켜진 뒤에는 sudden stop을
        # 다시 무효화한다. 같은 프레임에서 방어 reason이 추가된 뒤
        # confirm_candidate가 살아나는 것을 막기 위한 최종 정리다.
        sudden_stop_valid = (
            sudden_stop_reason
            and not false_stop_by_occlusion
            and not defense_large_vehicle_occlusion
        )

        if traffic_state in ["JAM", "CONGESTION"]:
            pair_collision_valid = (
                pair_evidence_raw
                and sudden_stop_valid
                and not defense_large_vehicle_occlusion
                and not false_stop_by_occlusion
                and not pair_false_in_congestion
                and not defense_small_far_pair
            )
        else:
            pair_collision_valid = (
                pair_evidence_raw
                and not defense_small_far_pair
            )

        real_evidence_in_frame = (
            pair_collision_valid
            or sudden_stop_valid
        )

        pair_only_without_direct_stop = (
            pair_collision_valid
            and not sudden_stop_reason
            and not bbox_jump_then_fixed_reason
            and not smoke_fire
        )

        if pair_only_without_direct_stop:
            real_evidence_in_frame = False
            if "defense_pair_only_without_direct_stop" not in reasons:
                reasons.append("defense_pair_only_without_direct_stop")

        false_stop_by_bbox_split = (
            bbox_jump_then_fixed_reason
            and sudden_stop_reason
            and not pair_collision_valid
            and not pair_evidence_raw
            and not smoke_fire
        )

        congestion_pair_only_false = (
            traffic_state in ["JAM", "CONGESTION"]
            and pair_evidence_raw
            and not sudden_stop_valid
            and localized_fixed_reason
            and persistent_cell_reason
        )

        if congestion_pair_only_false:
            score -= 5
            real_evidence_in_frame = False
            pair_collision_valid = False
            if "defense_pair_only_congestion" not in reasons:
                reasons.append("defense_pair_only_congestion")

        if pair_collision_valid:
            self.pair_collision_valid_history.append(frame_id)

        while (
            self.pair_collision_valid_history
            and frame_id - self.pair_collision_valid_history[0] > self.PAIR_REPEAT_WINDOW
        ):
            self.pair_collision_valid_history.popleft()

        pair_collision_repeat_count = len(self.pair_collision_valid_history)

        defense_congestion_like = "defense_congestion_like" in reasons
        defense_too_few_vehicles = "defense_too_few_vehicles" in reasons

        # -----------------------------------------------------
        # 7) 후보 단계 분리
        # -----------------------------------------------------
        # strong_suspect:
        #   사고 lock까지 갈 수 있는 후보.
        #   고정 장애물/후방 누적/위치 지속성 등 사고 고유 특징이 있어야 함.
        #
        # weak_suspect:
        #   사고 의심 후보.
        #   단독으로 lock 금지.
        # -----------------------------------------------------
        strong_suspect = (
            score >= 6
            and obstacle_info["localized_obstacle"]
            and (
                obstacle_info["rear_queue"]
                or obstacle_info["jump_fixed_count"] >= 1
                or cell_info["cell_persistent"]
                or smoke_fire
            )
            and not (
                obstacle_info["congestion_like"]
                and obstacle_info["jump_fixed_count"] == 0
                and not smoke_fire
            )
        )

        weak_suspect = (
            score >= 4
            and not strong_suspect
            and (
                obstacle_info["localized_obstacle"]
                or cell_info["cell_persistent"]
                or pair_collision_count >= 1
            )
        )

        # JAM/CONGESTION에서 localized + cell 조합만으로는 strong까지
        # 올라가지 못하게 한다. 사고 확정은 아래 confirm_candidate가 담당한다.
        if defense_jam_stationary_cell or defense_large_vehicle_occlusion:
            strong_suspect = False

        # -----------------------------------------------------
        # 7-1) queue/fixed/persistent 단독 확정 방어
        # -----------------------------------------------------
        # traffic_state가 NORMAL인지 여부와 무관하게, 강한 사고 증거 없이
        # fixed obstacle + rear queue + persistent cell만 반복되는 경우는
        # 확정 누적(confirm history)에 넣지 않는다. 다만 디버그와 화면의
        # weak/strong suspect 표시는 유지해서 원인 분석은 가능하게 둔다.
        defense_queue_only_without_real_evidence = False
        queue_only_without_real_evidence = (
            localized_fixed_reason
            and rear_queue_reason
            and persistent_cell_reason
            and not pair_collision_valid
            and not sudden_stop_reason
            and not bbox_jump_then_fixed_reason
            and not smoke_fire
        )

        if queue_only_without_real_evidence:
            defense_queue_only_without_real_evidence = True
            score -= 4
            real_evidence_in_frame = False
            if "defense_queue_only_without_real_evidence" not in reasons:
                reasons.append("defense_queue_only_without_real_evidence")

        # -----------------------------------------------------
        # 7-2) bbox/ID 분할에 의한 가짜 급정지 방어
        # -----------------------------------------------------
        # 큰 차량이 여러 bbox나 ID로 쪼개지면 bbox jump + sudden stop이
        # 동시에 잡힐 수 있다. pair 충돌 증거가 없으면 확정 후보와
        # confirm history 누적에서 제외하되, weak/strong suspect 로그는 유지한다.
        if false_stop_by_bbox_split:
            score -= 5
            sudden_stop_valid = False
            real_evidence_in_frame = False
            if "defense_bbox_split_false_stop" not in reasons:
                reasons.append("defense_bbox_split_false_stop")

        # real accident evidence는 현재 프레임 기준으로 계산한다.
        # pair 충돌은 직접 증거로 인정하지만, sudden stop은 대형차/가림
        # 방어를 통과한 경우에만 인정한다.
        pair_collision_repeat_evidence = (
            pair_collision_repeat_count >= self.PAIR_REPEAT_CONFIRM_COUNT
            and vehicle_count >= 3
            and visible_vehicle_count >= 3
            and not defense_too_few_vehicles
            and not defense_queue_only_without_real_evidence
            and not defense_congestion_like
            and not defense_jam_stationary_cell
            and not defense_large_vehicle_occlusion
            and not false_stop_by_occlusion
            and not false_stop_by_bbox_split
            and not congestion_pair_only_false
            and not defense_small_far_pair
            and not pair_only_without_direct_stop
        )

        has_real_accident_evidence = bool(
            real_evidence_in_frame
        )

        # 방어 reason이 찍힌 프레임은 score/의심 표시는 유지하되,
        # final accident 확정용 history에는 누적하지 않는다.
        final_accumulation_blocked = (
            not has_real_accident_evidence
            or defense_queue_only_without_real_evidence
            or defense_too_few_vehicles
            or defense_congestion_like
            or defense_jam_stationary_cell
            or pair_only_without_direct_stop
            or vehicle_count < 3
            or visible_vehicle_count < 3
        )

        pair_collision_repeat_confirm = (
            pair_collision_repeat_evidence
            and not final_accumulation_blocked
            and not defense_queue_only_without_real_evidence
            and vehicle_count >= 3
            and visible_vehicle_count >= 3
        )

        if pair_collision_repeat_confirm and "pair_collision_repeat" not in reasons:
            reasons.append("pair_collision_repeat")

        base_confirm_candidate = (
            strong_suspect
            and score >= 7
            and has_real_accident_evidence
            and not defense_congestion_like
            and not defense_jam_stationary_cell
            and not defense_large_vehicle_occlusion
            and not false_stop_by_occlusion
            and not queue_only_without_real_evidence
            and not false_stop_by_bbox_split
            and not final_accumulation_blocked
        )

        confirm_candidate = (
            base_confirm_candidate
        )

        history_info = self._update_histories(
            frame_id=frame_id,
            score=score,
            strong_candidate=strong_suspect and not final_accumulation_blocked,
            weak_candidate=weak_suspect,
            confirm_candidate=confirm_candidate,
            reasons=reasons,
        )

        # -----------------------------------------------------
        # 8) accident lock 판단
        # -----------------------------------------------------
        # V6 핵심:
        #   weak 후보만으로는 lock하지 않는다.
        #   strong 후보가 반복되고, frame 후보도 누적되어야 lock한다.
        # -----------------------------------------------------
        strong_confirmed = (
            history_info["strong_candidate_count"] >= self.STRONG_CONFIRM_COUNT
        )

        frame_confirmed = (
            history_info["frame_candidate_count"] >= self.ACCIDENT_CONFIRM_COUNT
        )

        weak_confirmed = (
            history_info["weak_candidate_count"] >= self.WEAK_CONFIRM_COUNT
        )

        has_final_accident_evidence = (
            pair_collision_valid
            or has_real_accident_evidence
            or confirm_candidate
        )

        # V6_1 마감 버전에서는 pair 반복은 보완 후보 지표로만 사용한다.
        # 성채/경부동탄 실시간 CCTV 오탐 방지를 위해 직접 lock 경로는 비활성화한다.
        pair_collision_repeat_lock = False

        new_lock_allowed = (
            vehicle_count >= 3
            and visible_vehicle_count >= 3
            and not final_accumulation_blocked
            and not defense_queue_only_without_real_evidence
            and not defense_too_few_vehicles
            and not defense_congestion_like
            and not defense_jam_stationary_cell
            and not defense_large_vehicle_occlusion
            and not false_stop_by_occlusion
            and not false_stop_by_bbox_split
            and not congestion_pair_only_false
            and not defense_small_far_pair
            and not pair_only_without_direct_stop
        )

        # 사고 확정 조건:
        # strong 후보와 confirm 후보가 반복되어야 하며,
        # 최근 strong 후보 안에 실제 사고성 증거가 반드시 있어야 한다.
        # localized/cell/rear_queue만 반복되는 정체 패턴은 lock하지 않는다.
        if (
            not self.accident_locked
            and strong_confirmed
            and frame_confirmed
            and has_final_accident_evidence
            and new_lock_allowed
        ):
            self.accident_locked = True
            self.accident_start_frame = frame_id
            print("🚨 V6 accident locked:", frame_id, reasons)

        accident_flag = self.accident_locked

        # candidate_only:
        # 사고 lock은 아니지만 의심 후보가 있는 상태.
        accident_candidate_only = (
            not accident_flag
            and (
                strong_suspect
                or weak_suspect
                or weak_confirmed
            )
        )

        # -----------------------------------------------------
        # 9) acc_ratio
        # 기존 UI/CSV 호환용.
        # V6에서는 pair positive ratio 대신 score 기반 ratio로 사용한다.
        # -----------------------------------------------------
        acc_ratio = max(0.0, min(float(score) / 10.0, 1.0))

        # -----------------------------------------------------
        # 10) 디버그 저장
        # -----------------------------------------------------
        self.last_debug = {
            "frame_id": frame_id,
            "accident": accident_flag,
            "acc_ratio": round(acc_ratio, 4),
            "frame_accident_prediction": bool(confirm_candidate),
            "recent_prediction_count": history_info["frame_candidate_count"],
            "accident_locked": self.accident_locked,
            "accident_start_frame": self.accident_start_frame,
            "accident_candidate_only": bool(accident_candidate_only),

            # V6 신규 디버그
            "accident_score": score,
            "reasons": reasons,
            "weak_suspect": bool(weak_suspect),
            "strong_suspect": bool(strong_suspect),
            "confirm_candidate": bool(confirm_candidate),
            "strong_candidate": bool(strong_suspect),
            "weak_candidate": bool(weak_suspect),
            "strong_confirmed": bool(strong_confirmed),
            "weak_confirmed": bool(weak_confirmed),
            "frame_confirmed": bool(frame_confirmed),
            "has_real_accident_evidence": bool(has_real_accident_evidence),
            "has_final_accident_evidence": bool(has_final_accident_evidence),
            "final_accumulation_blocked": bool(final_accumulation_blocked),
            "new_lock_allowed": bool(new_lock_allowed),
            "pair_collision_repeat_count": int(pair_collision_repeat_count),
            "pair_collision_repeat_count_window": int(pair_collision_repeat_count),
            "pair_collision_repeat_window": int(self.PAIR_REPEAT_WINDOW),
            "pair_collision_repeat_evidence": bool(pair_collision_repeat_evidence),
            "pair_collision_repeat_confirm": bool(pair_collision_repeat_confirm),
            "pair_collision_repeat_lock": bool(pair_collision_repeat_lock),

            "traffic_state": traffic_state,
            "defense_congestion_like": bool(defense_congestion_like),
            "defense_too_few_vehicles": bool(defense_too_few_vehicles),
            "defense_jam_stationary_cell": bool(defense_jam_stationary_cell),
            "defense_large_vehicle_occlusion": bool(defense_large_vehicle_occlusion),
            "defense_queue_only_without_real_evidence": bool(defense_queue_only_without_real_evidence),
            "queue_only_without_real_evidence": bool(queue_only_without_real_evidence),
            "defense_pair_only_without_direct_stop": bool(pair_only_without_direct_stop),
            "false_stop_by_bbox_split": bool(false_stop_by_bbox_split),
            "pair_evidence_raw": bool(pair_evidence_raw),
            "pair_collision_valid": bool(pair_collision_valid),
            "pair_false_in_congestion": bool(pair_false_in_congestion),
            "congestion_pair_only_false": bool(congestion_pair_only_false),
            "defense_small_far_pair": bool(defense_small_far_pair),
            "small_bbox_pair": bool(small_far_pair_debug.get("small_bbox_pair", False)),
            "far_region_pair": bool(small_far_pair_debug.get("far_region_pair", False)),
            "pair_bbox_h1": small_far_pair_debug.get("pair_bbox_h1", 0.0),
            "pair_bbox_h2": small_far_pair_debug.get("pair_bbox_h2", 0.0),
            "pair_bbox_area1": small_far_pair_debug.get("pair_bbox_area1", 0.0),
            "pair_bbox_area2": small_far_pair_debug.get("pair_bbox_area2", 0.0),
            "pair_bottom_y": small_far_pair_debug.get("pair_bottom_y", 0.0),
            "vehicle_drop": int(vehicle_drop),
            "prev_vehicle_count": int(prev_vehicle_count),

            "visible_vehicle_count": visible_vehicle_count,
            "vehicle_count": vehicle_count,
            "avg_speed": round(avg_speed, 4),

            "obstacle_info": obstacle_info,
            "cell_info": cell_info,
            "history_info": history_info,

            "pair_collision_count": pair_collision_count,
            "small_far_pair_count": small_far_pair_count,
            "pairs": pair_debug,
        }

        self.vehicle_count_history.append(vehicle_count)
        self.prev_vehicle_count = vehicle_count

        return {
            "accident": accident_flag,
            "acc_ratio": round(acc_ratio, 4),
            "frame_accident_prediction": bool(confirm_candidate),
            "recent_prediction_count": history_info["frame_candidate_count"],
            "accident_locked": self.accident_locked,
            "accident_candidate_only": bool(accident_candidate_only),

            # V6 추가 반환값
            "weak_suspect": bool(weak_suspect),
            "strong_suspect": bool(strong_suspect),
            "confirm_candidate": bool(confirm_candidate),
            "weak_confirmed": bool(weak_confirmed),
            "has_real_accident_evidence": bool(has_real_accident_evidence),
            "has_final_accident_evidence": bool(has_final_accident_evidence),
            "final_accumulation_blocked": bool(final_accumulation_blocked),
            "new_lock_allowed": bool(new_lock_allowed),
            "defense_too_few_vehicles": bool(defense_too_few_vehicles),
            "defense_jam_stationary_cell": bool(defense_jam_stationary_cell),
            "defense_large_vehicle_occlusion": bool(defense_large_vehicle_occlusion),
            "defense_queue_only_without_real_evidence": bool(defense_queue_only_without_real_evidence),
            "queue_only_without_real_evidence": bool(queue_only_without_real_evidence),
            "defense_pair_only_without_direct_stop": bool(pair_only_without_direct_stop),
            "false_stop_by_bbox_split": bool(false_stop_by_bbox_split),
            "pair_evidence_raw": bool(pair_evidence_raw),
            "pair_collision_valid": bool(pair_collision_valid),
            "pair_collision_repeat_count": int(pair_collision_repeat_count),
            "pair_collision_repeat_count_window": int(pair_collision_repeat_count),
            "pair_collision_repeat_window": int(self.PAIR_REPEAT_WINDOW),
            "pair_collision_repeat_evidence": bool(pair_collision_repeat_evidence),
            "pair_collision_repeat_confirm": bool(pair_collision_repeat_confirm),
            "pair_collision_repeat_lock": bool(pair_collision_repeat_lock),
            "congestion_pair_only_false": bool(congestion_pair_only_false),
            "defense_small_far_pair": bool(defense_small_far_pair),
            "small_bbox_pair": bool(small_far_pair_debug.get("small_bbox_pair", False)),
            "far_region_pair": bool(small_far_pair_debug.get("far_region_pair", False)),
            "pair_bbox_h1": small_far_pair_debug.get("pair_bbox_h1", 0.0),
            "pair_bbox_h2": small_far_pair_debug.get("pair_bbox_h2", 0.0),
            "pair_bbox_area1": small_far_pair_debug.get("pair_bbox_area1", 0.0),
            "pair_bbox_area2": small_far_pair_debug.get("pair_bbox_area2", 0.0),
            "pair_bottom_y": small_far_pair_debug.get("pair_bottom_y", 0.0),
            "vehicle_drop": int(vehicle_drop),
            "accident_score": score,
            "reasons": reasons,
            "filter_skip": None,
        }

    def get_debug_info(self):
        """
        eval_accident_logger나 디버그 CSV에서 호출할 수 있는 상세 정보.
        """
        return self.last_debug
