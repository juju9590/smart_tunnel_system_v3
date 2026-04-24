# ==========================================
# 파일명: traffic_accident_V5_5.py
# 설명:
# V5.5 사고 판단 로직
# - IoU 제외 유지
# - pair 후보 + frame 누적 + 사고 상태 lock 구조
# - 큰 gap 절대값 기준 추가
# - 최근 300프레임 내 사고 예측 5회 이상이면 사고 확정
# - 사고 확정 후 수동 해제 전까지 유지
# ==========================================

from collections import deque
import numpy as np
import time


class AccidentDetector:
    def __init__(self):
        # -----------------------------
        # pair 메모리
        # -----------------------------
        self.pair_memory = {}                 # (id1,id2) -> {"dist": ..., "gap": ...}
        self.pair_score = {}                  # pair별 사고 누적 점수
        self.impact_persist_counter = {}      # 근접/이상 지속 카운터

        # 반복성 메모리
        self.pair_candidate_frames = {}       # key -> [frame_id, ...]
        self.pair_last_candidate_frame = {}   # key -> 마지막 candidate frame

        # frame 사고 예측 history
        self.frame_prediction_history = deque()
        self.strong_candidate_history = deque()

        # 사고 상태 lock
        self.accident_locked = False
        self.accident_start_frame = None

        # -----------------------------
        # 임계값
        # -----------------------------
        self.VERTICAL_X_THR = 30
        self.SAME_LANE_X_THR = 55

        self.DIST_DROP_RATIO = 0.6
        self.GAP_UP_RATIO = 1.2
        self.GAP_MIN_ABS = 3.0

        # gap 절대값 기준
        self.GAP_ABS_WEAK_THR = 4.0
        self.GAP_ABS_STRONG_THR = 6.0

        # impact persist 보조 임계값
        self.IMPACT_DIST_THR = 45
        self.IMPACT_PERSIST_HOLD = 3

        # pair 반복성
        self.PAIR_REPEAT_WINDOW = 20
        self.PAIR_REPEAT_COUNT = 3
        self.PAIR_CONSEC_GAP = 2

        # pair 점수
        self.PAIR_SCORE_FRAME_THR = 3
        self.PAIR_SCORE_MAX = 10

        # frame 누적 사고 확정
        self.ACCIDENT_WINDOW = 150
        self.ACCIDENT_CONFIRM_COUNT = 1

        # 실시간 CCTV 방어 필터
        self.MAX_FRAME_TIME_GAP = 0.5
        self.FRAME_GAP_FACTOR = 3.0
        self.MIN_TRACK_AGE = 10
        self.CENTER_JUMP_FACTOR = 4.0
        self.MIN_CENTER_JUMP_ABS = 80.0
        self.BBOX_AREA_JUMP_RATIO = 2.0
        self.STRONG_WINDOW_FRAMES = 150
        self.STRONG_WINDOW_SEC = 5.0
        self.STRONG_REPEAT_COUNT = 5
        self.STOP_SPEED_THR = 2.0
        self.STOP_HOLD_SEC = 2.0
        self.MIN_STRONG_VEHICLE_COUNT = 2
        self.SKIP_LOG_COOLDOWN_SEC = 5.0

        # 메모리 정리용
        self.STALE_FRAME_GAP = 60

        self.last_update_ts = None
        self.recent_frame_intervals = deque(maxlen=20)
        self.prev_track_metrics = {}
        self.track_seen_count = {}
        self.low_speed_started_at = None
        self.last_skip_log_ts = {}

        # 디버그 정보
        self.last_debug = {
            "frame_id": 0,
            "accident": False,
            "acc_ratio": 0.0,
            "frame_accident_prediction": False,
            "recent_prediction_count": 0,
            "accident_locked": False,
            "accident_candidate_only": False,
            "accident_filter_skip": None,
            "pairs": []
        }

    # =========================================================
    # 외부 수동 해제용
    # =========================================================
    def clear_accident(self):
        self.accident_locked = False
        self.accident_start_frame = None
        self.frame_prediction_history.clear()
        self.strong_candidate_history.clear()
        self.low_speed_started_at = None

    # =========================================================
    # 내부 유틸
    # =========================================================
    def _cleanup_stale_pairs(self, frame_id):
        stale_keys = []

        for key, last in self.pair_last_candidate_frame.items():
            if frame_id - last > self.STALE_FRAME_GAP:
                stale_keys.append(key)

        for key in stale_keys:
            self.pair_candidate_frames.pop(key, None)
            self.pair_last_candidate_frame.pop(key, None)

    def _update_candidate_repeat(self, key, frame_id, pair_accident_candidate):
        """
        같은 pair의 candidate 반복성 관리
        반환:
            pair_repeat_candidate
            pair_consecutive_candidate
            repeat_count_window
        """
        self.pair_candidate_frames.setdefault(key, [])

        pair_repeat_candidate = False
        pair_consecutive_candidate = False

        if pair_accident_candidate:
            last_frame = self.pair_last_candidate_frame.get(key, None)

            if last_frame is not None and (frame_id - last_frame) <= self.PAIR_CONSEC_GAP:
                pair_consecutive_candidate = True

            self.pair_candidate_frames[key].append(frame_id)
            self.pair_last_candidate_frame[key] = frame_id

        recent_frames = [
            f for f in self.pair_candidate_frames.get(key, [])
            if frame_id - f <= self.PAIR_REPEAT_WINDOW
        ]
        self.pair_candidate_frames[key] = recent_frames

        repeat_count_window = len(recent_frames)
        pair_repeat_candidate = repeat_count_window >= self.PAIR_REPEAT_COUNT

        return pair_repeat_candidate, pair_consecutive_candidate, repeat_count_window

    def _update_frame_prediction_history(self, frame_id, frame_accident_prediction):
        self.frame_prediction_history.append((frame_id, frame_accident_prediction))

        while self.frame_prediction_history and (frame_id - self.frame_prediction_history[0][0] > self.ACCIDENT_WINDOW):
            self.frame_prediction_history.popleft()

        recent_prediction_count = sum(
            1 for _, pred in self.frame_prediction_history if pred
        )

        return recent_prediction_count

    def _log_skip(self, reason):
        now = time.time()
        last = self.last_skip_log_ts.get(reason, 0.0)
        if now - last >= self.SKIP_LOG_COOLDOWN_SEC:
            print(f"🧊 사고 판단 skip: {reason}")
            self.last_skip_log_ts[reason] = now

    def _is_frame_gap_abnormal(self, now_ts):
        if self.last_update_ts is None:
            self.last_update_ts = now_ts
            return False, 0.0

        time_gap = now_ts - self.last_update_ts
        avg_interval = None
        if self.recent_frame_intervals:
            avg_interval = sum(self.recent_frame_intervals) / len(self.recent_frame_intervals)

        dynamic_thr = self.MAX_FRAME_TIME_GAP
        if avg_interval and avg_interval > 0:
            dynamic_thr = max(dynamic_thr, avg_interval * self.FRAME_GAP_FACTOR)

        self.last_update_ts = now_ts

        if time_gap > dynamic_thr:
            return True, time_gap

        self.recent_frame_intervals.append(time_gap)
        return False, time_gap

    def _filter_valid_track_ids(self, tracks, analysis):
        boxes = analysis.get("boxes", {})
        history = analysis.get("track_history", {})

        current_ids = set()
        valid_ids = set()
        skipped_age = 0
        skipped_jump = 0

        for t in tracks:
            tid = t.get("id")
            if tid is None:
                continue

            current_ids.add(tid)
            self.track_seen_count[tid] = self.track_seen_count.get(tid, 0) + 1

            box = boxes.get(tid, t.get("bbox"))
            if not box:
                continue

            x1, y1, x2, y2 = box
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            area = float(max(int(x2) - int(x1), 1) * max(int(y2) - int(y1), 1))

            hist_age = len(history.get(tid, [])) if isinstance(history, dict) else 0
            age = max(hist_age, self.track_seen_count.get(tid, 0))

            prev = self.prev_track_metrics.get(tid)
            is_outlier = False

            if prev:
                dx = cx - prev["cx"]
                dy = cy - prev["cy"]
                move = float(np.sqrt(dx ** 2 + dy ** 2))
                avg_move = prev.get("avg_move", 0.0)
                jump_thr = max(self.MIN_CENTER_JUMP_ABS, avg_move * self.CENTER_JUMP_FACTOR)

                prev_area = max(float(prev.get("area", area)), 1.0)
                area_ratio = max(area / prev_area, prev_area / max(area, 1.0))

                if move > jump_thr or area_ratio >= self.BBOX_AREA_JUMP_RATIO:
                    is_outlier = True
                    skipped_jump += 1

                smooth_move = move if avg_move <= 0 else (0.8 * avg_move + 0.2 * move)
            else:
                smooth_move = 0.0

            self.prev_track_metrics[tid] = {
                "cx": cx,
                "cy": cy,
                "area": area,
                "avg_move": smooth_move,
            }

            if age < self.MIN_TRACK_AGE:
                skipped_age += 1
                continue

            if is_outlier:
                continue

            valid_ids.add(tid)

        stale_ids = [tid for tid in self.prev_track_metrics.keys() if tid not in current_ids]
        for tid in stale_ids:
            self.prev_track_metrics.pop(tid, None)
            self.track_seen_count.pop(tid, None)

        if skipped_age:
            self._log_skip("track_age 부족")
        if skipped_jump:
            self._log_skip("bbox jump")

        return valid_ids, {
            "track_age": skipped_age,
            "bbox_jump": skipped_jump,
        }

    def _update_strong_suspect(self, frame_id, now_ts, candidate, avg_speed, vehicle_count):
        if avg_speed < self.STOP_SPEED_THR:
            if self.low_speed_started_at is None:
                self.low_speed_started_at = now_ts
        else:
            self.low_speed_started_at = None

        low_speed_hold = (
            self.low_speed_started_at is not None
            and now_ts - self.low_speed_started_at >= self.STOP_HOLD_SEC
        )

        if candidate:
            self.strong_candidate_history.append((frame_id, now_ts))

        while self.strong_candidate_history:
            old_frame, old_ts = self.strong_candidate_history[0]
            if frame_id - old_frame <= self.STRONG_WINDOW_FRAMES and now_ts - old_ts <= self.STRONG_WINDOW_SEC:
                break
            self.strong_candidate_history.popleft()

        repeat_count = len(self.strong_candidate_history)
        strong_suspect = (
            repeat_count >= self.STRONG_REPEAT_COUNT
            and (avg_speed < self.STOP_SPEED_THR or low_speed_hold)
            and vehicle_count >= self.MIN_STRONG_VEHICLE_COUNT
        )

        return strong_suspect, repeat_count, low_speed_hold

    # =========================================================
    # 사고 판단
    # =========================================================
    def update(self, frame_id, tracks, analysis):
        """
        입력:
            frame_id : 현재 프레임 번호
            tracks   : [{"id": tid, "bbox": (...)}, ...]
            analysis : TrackAnalyzer + LaneTemplate 결과 dict

        출력:
            {
                "accident": bool,
                "acc_ratio": float,
                "frame_accident_prediction": bool,
                "accident_locked": bool
            }
        """

        now_ts = time.time()
        frame_gap_skip, time_gap = self._is_frame_gap_abnormal(now_ts)
        if frame_gap_skip:
            self._log_skip("frame_gap")
            self.last_debug = {
                "frame_id": frame_id,
                "accident": False,
                "acc_ratio": 0.0,
                "frame_accident_prediction": False,
                "recent_prediction_count": 0,
                "accident_locked": self.accident_locked,
                "accident_candidate_only": False,
                "accident_filter_skip": "frame_gap",
                "time_gap": round(time_gap, 3),
                "pairs": []
            }
            return {
                "accident": False,
                "acc_ratio": 0.0,
                "frame_accident_prediction": False,
                "recent_prediction_count": 0,
                "accident_locked": self.accident_locked,
                "accident_candidate_only": False,
                "strong_suspect": False,
                "filter_skip": "frame_gap",
            }

        self._cleanup_stale_pairs(frame_id)

        boxes = analysis.get("boxes", {})
        speeds = analysis.get("speeds", {})
        avg_speed = float(analysis.get("avg_speed", 0.0))
        lane_map = analysis.get("lane_map", {})
        smoke_fire_map = analysis.get("smoke_fire_map", {})
        vehicle_count = int(analysis.get("vehicle_count", len(tracks)))

        valid_ids, filter_counts = self._filter_valid_track_ids(tracks, analysis)
        ids = [tid for tid in boxes.keys() if tid in valid_ids]

        pair_debug = []
        total_pairs = 0
        positive_pairs = 0

        frame_has_strong_candidate = False
        frame_has_repeat_strong_candidate = False
        frame_has_high_score_pair = False

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                total_pairs += 1

                box1 = boxes[id1]
                box2 = boxes[id2]

                cx1 = int((box1[0] + box1[2]) / 2)
                cy1 = int(box1[3])
                cx2 = int((box2[0] + box2[2]) / 2)
                cy2 = int(box2[3])

                dist = float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))

                s1 = float(speeds.get(id1, 0.0))
                s2 = float(speeds.get(id2, 0.0))
                gap = abs(s1 - s2)

                key = tuple(sorted((id1, id2)))

                prev = self.pair_memory.get(key, {
                    "dist": dist,
                    "gap": gap
                })

                # -----------------------------
                # 변화량
                # -----------------------------
                dist_drop = dist < prev["dist"] * self.DIST_DROP_RATIO
                gap_up = (gap > prev["gap"] * self.GAP_UP_RATIO) or (gap > self.GAP_MIN_ABS)

                # -----------------------------
                # 차선 / 배치 관계
                # -----------------------------
                lane1 = lane_map.get(id1, None)
                lane2 = lane_map.get(id2, None)

                same_lane = (lane1 is not None and lane2 is not None and lane1 == lane2)

                vertical = abs(cx1 - cx2) < self.VERTICAL_X_THR
                vertical_or_lane = vertical or (same_lane and abs(cx1 - cx2) < self.SAME_LANE_X_THR)

                lane_break = (same_lane and abs(cx1 - cx2) > self.SAME_LANE_X_THR)

                lane_break_acc = (
                    same_lane and
                    (dist < self.IMPACT_DIST_THR) and
                    (not vertical_or_lane)
                )

                # -----------------------------
                # impact persist
                # -----------------------------
                self.impact_persist_counter.setdefault(key, 0)

                if (dist < self.IMPACT_DIST_THR) and (gap_up or dist_drop):
                    self.impact_persist_counter[key] += 1
                else:
                    self.impact_persist_counter[key] = 0

                impact_persist = self.impact_persist_counter[key] >= self.IMPACT_PERSIST_HOLD

                # -----------------------------
                # 비정상 정지
                # -----------------------------
                abnormal = ((s1 < 2.0 or s2 < 2.0) and avg_speed > 3.0)
                abnormal_stop = abnormal and gap_up

                # -----------------------------
                # 핵심 패턴
                # -----------------------------
                rear_core = dist_drop and gap_up and (same_lane or vertical)

                gap_weak = gap >= self.GAP_ABS_WEAK_THR
                gap_strong = gap >= self.GAP_ABS_STRONG_THR

                persist_evidence = impact_persist and rear_core
                stop_evidence = abnormal_stop
                lane_break_evidence = lane_break_acc

                smoke_fire = False
                if isinstance(smoke_fire_map, dict):
                    smoke_fire = bool(smoke_fire_map.get(key, False))

                post_evidence = (
                    persist_evidence or
                    stop_evidence or
                    lane_break_evidence or
                    smoke_fire
                )

                abnormal_pose = lane_break_acc or abnormal_stop

                # -----------------------------
                # pair 후보
                # -----------------------------
                weak_pair_candidate = rear_core and gap_weak
                strong_pair_candidate = rear_core and post_evidence and gap_strong

                pair_accident_candidate = weak_pair_candidate or strong_pair_candidate

                # -----------------------------
                # pair 반복성
                # -----------------------------
                pair_repeat_candidate, pair_consecutive_candidate, repeat_count_window = \
                    self._update_candidate_repeat(key, frame_id, pair_accident_candidate)

                repeat_strong_candidate = pair_repeat_candidate or pair_consecutive_candidate

                # -----------------------------
                # pair 점수
                # strong / repeat strong +2
                # weak                  +1
                # else                  +0
                # -----------------------------
                self.pair_score.setdefault(key, 0)

                if strong_pair_candidate or repeat_strong_candidate:
                    self.pair_score[key] += 2
                elif weak_pair_candidate:
                    self.pair_score[key] += 1
                else:
                    self.pair_score[key] += 0

                self.pair_score[key] = max(
                    0,
                    min(self.PAIR_SCORE_MAX, self.pair_score[key])
                )

                pair_high_score = self.pair_score[key] >= self.PAIR_SCORE_FRAME_THR

                if strong_pair_candidate:
                    frame_has_strong_candidate = True
                if repeat_strong_candidate:
                    frame_has_repeat_strong_candidate = True
                if pair_high_score:
                    frame_has_high_score_pair = True

                if pair_high_score:
                    positive_pairs += 1

                # -----------------------------
                # pair 메모리 업데이트
                # -----------------------------
                self.pair_memory[key] = {
                    "dist": dist,
                    "gap": gap
                }

                pair_debug.append({
                    "pair": key,
                    "lane1": lane1,
                    "lane2": lane2,
                    "same_lane": same_lane,

                    "dist": round(dist, 2),
                    "gap": round(gap, 2),

                    "dist_drop": dist_drop,
                    "gap_up": gap_up,
                    "vertical": vertical,
                    "vertical_or_lane": vertical_or_lane,

                    "lane_break": lane_break,
                    "lane_break_acc": lane_break_acc,

                    "impact_persist": impact_persist,
                    "persist_evidence": persist_evidence,
                    "stop_evidence": stop_evidence,
                    "lane_break_evidence": lane_break_evidence,
                    "smoke_fire": smoke_fire,

                    "abnormal": abnormal,
                    "abnormal_stop": abnormal_stop,
                    "abnormal_pose": abnormal_pose,

                    "rear_core": rear_core,
                    "gap_weak": gap_weak,
                    "gap_strong": gap_strong,
                    "post_evidence": post_evidence,

                    "weak_pair_candidate": weak_pair_candidate,
                    "strong_pair_candidate": strong_pair_candidate,
                    "pair_accident_candidate": pair_accident_candidate,

                    "pair_repeat_candidate": pair_repeat_candidate,
                    "pair_consecutive_candidate": pair_consecutive_candidate,
                    "repeat_count_window": repeat_count_window,
                    "repeat_strong_candidate": repeat_strong_candidate,

                    "pair_score": self.pair_score[key],
                    "pair_high_score": pair_high_score
                })

        # =====================================================
        # frame 사고 예측
        # =====================================================
        candidate_prediction = (
            frame_has_strong_candidate
            or frame_has_repeat_strong_candidate
            or frame_has_high_score_pair
        )

        strong_suspect, strong_repeat_count, low_speed_hold = self._update_strong_suspect(
            frame_id=frame_id,
            now_ts=now_ts,
            candidate=candidate_prediction,
            avg_speed=avg_speed,
            vehicle_count=vehicle_count,
        )

        frame_accident_prediction = strong_suspect

        recent_prediction_count = self._update_frame_prediction_history(
            frame_id, frame_accident_prediction
        )

        # =====================================================
        # 사고 확정 / 상태 lock
        # =====================================================
        if (not self.accident_locked) and strong_suspect and (recent_prediction_count >= self.ACCIDENT_CONFIRM_COUNT):
            self.accident_locked = True
            self.accident_start_frame = frame_id
            print("🚨 strong accident suspect 생성")

        accident_flag = self.accident_locked

        acc_ratio = (positive_pairs / total_pairs) if total_pairs > 0 else 0.0

        self.last_debug = {
            "frame_id": frame_id,
            "accident": accident_flag,
            "acc_ratio": round(acc_ratio, 4),
            "frame_accident_prediction": frame_accident_prediction,
            "recent_prediction_count": recent_prediction_count,
            "accident_locked": self.accident_locked,
            "accident_start_frame": self.accident_start_frame,
            "accident_candidate_only": bool(candidate_prediction and not strong_suspect),
            "strong_suspect": strong_suspect,
            "strong_repeat_count": strong_repeat_count,
            "low_speed_hold": low_speed_hold,
            "filter_skip": None,
            "filter_counts": filter_counts,
            "pairs": pair_debug
        }

        return {
            "accident": accident_flag,
            "acc_ratio": round(acc_ratio, 4),
            "frame_accident_prediction": frame_accident_prediction,
            "recent_prediction_count": recent_prediction_count,
            "accident_locked": self.accident_locked,
            "accident_candidate_only": bool(candidate_prediction and not strong_suspect),
            "strong_suspect": strong_suspect,
            "strong_repeat_count": strong_repeat_count,
            "filter_skip": None,
        }

    def get_debug_info(self):
        return self.last_debug
