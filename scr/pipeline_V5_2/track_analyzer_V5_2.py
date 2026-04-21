# ==========================================
# 파일명: track_analyzer_V5_2.py
# 설명:
# 공통 추적 분석기
# - track history 관리
# - 차량 속도 계산
# - 궤적 기반 차선 추정
# - 상태/사고 로직이 공통 사용
#
# [이번 궤적 기반 차선 추정 수정 핵심]
# 기존:
#   최근 5프레임 동일 raw lane -> stable lane 확정
#
# 변경:
#   1) raw lane 최근 50프레임 누적
#   2) 최근 50프레임에서 가장 많이 나온 차선(majority lane) 계산
#   3) 현재 시점에서 같은 raw lane이 몇 프레임 연속인지(streak) 계산
#   4) stable 확정 전:
#        majority lane == streak lane
#        AND streak_len >= 8
#        AND majority_ratio >= 0.40
#        이면 stable lane 확정
#   5) stable 확정 후 freeze 중:
#        raw lane 흔들려도 stable 유지
#   6) freeze 종료 후:
#        다른 차선이 12프레임 연속
#        AND 최근 50프레임 majority도 그 차선
#        AND majority_ratio >= 0.40
#        이면 재배정
# ==========================================

import numpy as np
from collections import defaultdict, deque, Counter


class TrackAnalyzer:
    def __init__(self):
        # -----------------------------
        # 공통 추적 메모리
        # -----------------------------
        self.track_history = {}          # tid -> [(cx, cy), ...]
        self.prev_speeds = {}            # tid -> smoothed speed

        # -----------------------------
        # 차선 추정용 메모리
        # -----------------------------
        self.track_models = {}           # tid -> fitted trajectory model
        self.track_fit_error = {}        # tid -> fit rmse
        self.track_stable_motion = {}    # tid -> stable motion bool

        self.centerlines = []            # [{"lane_id":0, "model_type":"linear", ...}, ...]
        self.current_lane_raw = {}       # tid -> raw nearest lane id
        self.current_lane_stable = {}    # tid -> confirmed lane id

        # [변경]
        # 과거 recent N표 동일 확인용 deque 제거
        # -> 최근 50프레임 raw lane 저장용으로 교체
        self.lane_raw_history = defaultdict(lambda: deque(maxlen=self.LANE_HISTORY_SIZE))   # tid -> 최근 50 raw lane
        self.lane_change_memory = defaultdict(lambda: deque(maxlen=10))  # tid -> stable lane 변경 이력

        # [추가]
        # raw lane 연속 streak 관리
        self.lane_last_raw = {}          # tid -> 직전 raw lane
        self.lane_same_streak = {}       # tid -> 동일 raw lane 연속 길이

        # [추가]
        # stable lane freeze 관리
        self.lane_freeze_until = {}      # tid -> 이 frame_id까지 freeze 유지
        self.lane_stable_since = {}      # tid -> stable lane 확정/재배정된 시점

        # -----------------------------
        # 디버그 정보
        # -----------------------------
        self.last_debug = {
            "vehicle_count": 0,
            "avg_speed": 0.0,
            "lane_count": 0,
            "lane_map": {},
            "raw_lane_map": {},
            "centerlines": [],
            "speeds": {},
            "boxes": {},
            "lane_debug": {},   # tid별 majority/streak/freeze 디버그
        }

        # -----------------------------
        # 파라미터
        # -----------------------------
        self.frame_height = 720
        self.ROI_Y1_RATIO = 0.30
        self.ROI_Y2_RATIO = 0.80

        self.MAX_HISTORY = 60
        self.FIT_MIN_POINTS = 30
        self.MIN_TOTAL_MOTION = 35
        self.MAX_DY_JUMP = 40
        self.MAX_DX_JUMP = 60

        self.MAX_SPEED = 20
        self.SPEED_JUMP_LIMIT = 10

        self.LINEAR_CLUSTER_THR = 0.085
        self.QUAD_CLUSTER_THR = 0.12

        # -----------------------------
        # 차선 안정화 파라미터
        # -----------------------------
        self.LANE_HISTORY_SIZE = 50          # 최근 raw lane 관찰 창
        self.LANE_MIN_SAMPLES = 20            # stable 판정 최소 표본 수
        self.LANE_CONFIRM_STREAK = 8          # 최초 stable 확정용 연속 프레임 수
        self.LANE_REASSIGN_STREAK = 12        # 재배정용 연속 프레임 수
        self.LANE_MAJORITY_RATIO = 0.40       # 최근 history 내 다수결 최소 비율
        self.LANE_FREEZE_FRAMES = 100         # stable lane 확정 후 freeze 프레임

    # =========================================================
    # 기본 유틸
    # =========================================================
    def _roi_bounds(self):
        roi_y1 = int(self.frame_height * self.ROI_Y1_RATIO)
        roi_y2 = int(self.frame_height * self.ROI_Y2_RATIO)
        return roi_y1, roi_y2

    def _clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def _update_frame_height_from_tracks(self, tracks):
        """
        bbox의 y2 최대값을 기반으로 대략 frame_height 보정
        실제 frame shape를 main에서 넘기지 않으므로 안전한 추정만 수행
        """
        max_y2 = 0
        for t in tracks:
            _, _, _, y2 = t["bbox"]
            max_y2 = max(max_y2, int(y2))

        if max_y2 > 0:
            # bbox 하단값만 있으니 약간 여유를 둬서 갱신
            self.frame_height = max(self.frame_height, int(max_y2 * 1.05))

    # =========================================================
    # 1) tracks -> boxes / history / speeds
    # =========================================================
    def _update_tracks_and_speeds(self, tracks):
        boxes = {}
        speeds = {}

        roi_y1, roi_y2 = self._roi_bounds()

        for t in tracks:
            tid = t["id"]
            x1, y1, x2, y2 = t["bbox"]

            cx = int((x1 + x2) / 2)
            cy = int(y2)   # 하단 발점 기준

            boxes[tid] = (x1, y1, x2, y2)

            # history 저장
            self.track_history.setdefault(tid, []).append((cx, cy))
            if len(self.track_history[tid]) > self.MAX_HISTORY:
                self.track_history[tid].pop(0)

            # ROI 밖이면 이전 속도 유지
            if cy < roi_y1 or cy > roi_y2:
                speeds[tid] = self.prev_speeds.get(tid, 0.0)
                continue

            # 초기 보호
            if len(self.track_history[tid]) < 3:
                speed = self.prev_speeds.get(tid, 0.0)
                self.prev_speeds[tid] = speed
                speeds[tid] = speed
                continue

            xp, yp = self.track_history[tid][-2]
            xc, yc = self.track_history[tid][-1]

            dx = abs(xc - xp)
            dy = yc - yp

            # 비정상 점프 제거
            if abs(dy) > self.MAX_DY_JUMP or dx > self.MAX_DX_JUMP:
                speed = self.prev_speeds.get(tid, 0.0)
                speeds[tid] = speed
                continue

            # 원근 보정
            scale = (yc - roi_y1) / (roi_y2 - roi_y1 + 1e-6)
            scale = self._clamp(scale, 0.1, 1.0)

            speed = abs(dy) / (scale + 0.15)
            speed *= (1.2 + scale)

            prev_speed = self.prev_speeds.get(tid, speed)

            if speed > self.MAX_SPEED:
                speed = prev_speed

            if abs(speed - prev_speed) > self.SPEED_JUMP_LIMIT:
                speed = prev_speed

            # smoothing
            speed = 0.3 * speed + 0.7 * prev_speed
            speed = self._clamp(speed, 0, self.MAX_SPEED)

            self.prev_speeds[tid] = speed
            speeds[tid] = speed

        valid = [s for s in speeds.values() if s < self.MAX_SPEED]
        avg_speed = float(np.mean(valid)) if len(valid) > 0 else 0.0

        return boxes, speeds, avg_speed

    # =========================================================
    # 2) 차선 추정: 안정성 검사
    # =========================================================
    def _is_stable_moving_track(self, pts):
        if len(pts) < self.FIT_MIN_POINTS:
            return False

        recent = pts[-self.FIT_MIN_POINTS:]

        total_motion = 0.0
        jump_bad = 0
        ys = []

        for i in range(1, len(recent)):
            x0, y0 = recent[i - 1]
            x1, y1 = recent[i]

            dx = x1 - x0
            dy = y1 - y0

            total_motion += np.sqrt(dx * dx + dy * dy)

            if abs(dy) > self.MAX_DY_JUMP or abs(dx) > self.MAX_DX_JUMP:
                jump_bad += 1

            ys.append(y1)

        y_span = (max(ys) - min(ys)) if ys else 0

        if total_motion < self.MIN_TOTAL_MOTION:
            return False
        if jump_bad > 3:
            return False
        if y_span < 25:
            return False

        return True

    # =========================================================
    # 3) 궤적 피팅
    # x = a*y + b 또는 x = a*y^2 + b*y + c
    # =========================================================
    def _fit_trajectory_model(self, pts):
        recent = pts[-self.FIT_MIN_POINTS:]
        ys = np.array([p[1] for p in recent], dtype=np.float32)
        xs = np.array([p[0] for p in recent], dtype=np.float32)

        if len(np.unique(ys)) < 6:
            return None, 1e9

        # linear
        try:
            lin_coef = np.polyfit(ys, xs, 1)
            lin_pred = np.polyval(lin_coef, ys)
            lin_rmse = float(np.sqrt(np.mean((xs - lin_pred) ** 2)))
        except Exception:
            lin_coef = None
            lin_rmse = 1e9

        # quadratic
        try:
            quad_coef = np.polyfit(ys, xs, 2)
            quad_pred = np.polyval(quad_coef, ys)
            quad_rmse = float(np.sqrt(np.mean((xs - quad_pred) ** 2)))
        except Exception:
            quad_coef = None
            quad_rmse = 1e9

        if lin_coef is None and quad_coef is None:
            return None, 1e9

        if quad_coef is not None and lin_coef is not None:
            improve_ratio = (lin_rmse - quad_rmse) / max(lin_rmse, 1e-6)

            if improve_ratio > 0.18 and abs(quad_coef[0]) > 1e-5:
                return {
                    "type": "quadratic",
                    "coef": quad_coef.astype(float).tolist()
                }, quad_rmse
            else:
                return {
                    "type": "linear",
                    "coef": lin_coef.astype(float).tolist()
                }, lin_rmse

        if lin_coef is not None:
            return {
                "type": "linear",
                "coef": lin_coef.astype(float).tolist()
            }, lin_rmse

        return {
            "type": "quadratic",
            "coef": quad_coef.astype(float).tolist()
        }, quad_rmse

    def _predict_x(self, model, y):
        if model is None:
            return None

        if model["type"] == "linear":
            a, b = model["coef"]
            return a * y + b
        else:
            a, b, c = model["coef"]
            return a * (y ** 2) + b * y + c

    # =========================================================
    # 4) 모델 벡터화 / 거리
    # =========================================================
    def _coef_vector(self, model):
        if model["type"] == "linear":
            a, b = model["coef"]
            return np.array([0.0, a, b / max(self.frame_height, 1)], dtype=np.float32)

        a, b, c = model["coef"]
        return np.array([
            a * self.frame_height,
            b,
            c / max(self.frame_height, 1)
        ], dtype=np.float32)

    def _model_distance(self, model1, model2):
        v1 = self._coef_vector(model1)
        v2 = self._coef_vector(model2)
        coef_dist = float(np.linalg.norm(v1 - v2))

        roi_y1, roi_y2 = self._roi_bounds()
        sample_ys = np.linspace(roi_y1, roi_y2, 5)

        diffs = []
        for y in sample_ys:
            x1 = self._predict_x(model1, y)
            x2 = self._predict_x(model2, y)
            diffs.append(abs(x1 - x2))

        shape_dist = float(np.mean(diffs)) / max(self.frame_height, 1)
        return coef_dist * 0.5 + shape_dist * 0.5

    # =========================================================
    # 5) 군집 중심 모델
    # =========================================================
    def _aggregate_models(self, models):
        if not models:
            return None

        linear_models = [m for m in models if m["type"] == "linear"]
        quad_models = [m for m in models if m["type"] == "quadratic"]

        if len(quad_models) > len(linear_models):
            arr = np.array([m["coef"] for m in quad_models], dtype=np.float32)
            coef = np.median(arr, axis=0).tolist()
            return {"type": "quadratic", "coef": coef}

        arr = np.array([m["coef"] for m in linear_models], dtype=np.float32)
        if len(arr) == 0:
            arr = np.array([m["coef"] for m in models], dtype=np.float32)
            coef = np.median(arr, axis=0).tolist()
            if len(coef) == 2:
                return {"type": "linear", "coef": coef}
            return {"type": "quadratic", "coef": coef}

        coef = np.median(arr, axis=0).tolist()
        return {"type": "linear", "coef": coef}

    # =========================================================
    # 6) 자동 군집화
    # =========================================================
    def _cluster_track_models(self, stable_models):
        if not stable_models:
            return []

        clusters = []

        for tid, model in stable_models:
            assigned = False
            thr = self.QUAD_CLUSTER_THR if model["type"] == "quadratic" else self.LINEAR_CLUSTER_THR

            for cluster in clusters:
                dist = self._model_distance(model, cluster["rep_model"])
                if dist < thr:
                    cluster["items"].append((tid, model))
                    cluster["rep_model"] = self._aggregate_models([m for _, m in cluster["items"]])
                    assigned = True
                    break

            if not assigned:
                clusters.append({
                    "cluster_id": len(clusters),
                    "rep_model": model,
                    "items": [(tid, model)]
                })

        centerlines = []
        roi_y1, roi_y2 = self._roi_bounds()
        y_mid = (roi_y1 + roi_y2) / 2.0

        for c in clusters:
            agg_model = self._aggregate_models([m for _, m in c["items"]])

            centerlines.append({
                "cluster_id": c["cluster_id"],
                "model_type": agg_model["type"],
                "coef": agg_model["coef"],
                "member_ids": [tid for tid, _ in c["items"]],
                "x_mid": self._predict_x(agg_model, y_mid)
            })

        centerlines.sort(key=lambda x: x["x_mid"])

        for idx, cl in enumerate(centerlines):
            cl["lane_id"] = idx

        return centerlines

    # =========================================================
    # 7) 각 차량 raw lane 할당
    # =========================================================
    def _assign_lane_raw(self, current_point):
        if not self.centerlines:
            return None

        x, y = current_point
        best_lane = None
        best_dist = 1e9

        for cl in self.centerlines:
            model = {
                "type": cl["model_type"],
                "coef": cl["coef"]
            }
            cx = self._predict_x(model, y)
            dist = abs(x - cx)

            if dist < best_dist:
                best_dist = dist
                best_lane = cl["lane_id"]

        if best_dist > 80:
            return None

        return best_lane

    # =========================================================
    # 7-1) 최근 raw lane history의 다수결 계산
    # =========================================================
    def _get_lane_majority_info(self, tid):
        """
        최근 100프레임 raw lane history에서
        - 가장 많이 나온 차선번호(majority_lane)
        - 등장 횟수(majority_count)
        - 비율(majority_ratio)
        를 계산한다.

        None은 차선 미할당 상태이므로 다수결 계산에서 제외한다.
        """
        history = self.lane_raw_history[tid]
        valid_lanes = [v for v in history if v is not None]

        if len(valid_lanes) == 0:
            return None, 0, 0.0, 0

        counter = Counter(valid_lanes)
        majority_lane, majority_count = counter.most_common(1)[0]
        total_valid = len(valid_lanes)
        majority_ratio = majority_count / max(total_valid, 1)

        return majority_lane, majority_count, majority_ratio, total_valid

    # =========================================================
    # 7-2) raw lane streak 업데이트
    # =========================================================
    def _update_lane_streak(self, tid, raw_lane):
        """
        raw lane이 직전 프레임과 같으면 streak 증가,
        다르면 streak를 1로 리셋한다.

        예:
            raw: 1,1,1,1 -> streak = 4
            다음 프레임 raw=2 -> streak = 1 (lane 2 시작)
        """
        prev_raw = self.lane_last_raw.get(tid, None)

        if raw_lane == prev_raw:
            self.lane_same_streak[tid] = self.lane_same_streak.get(tid, 0) + 1
        else:
            self.lane_same_streak[tid] = 1
            self.lane_last_raw[tid] = raw_lane

        return self.lane_same_streak[tid]

    # =========================================================
    # 7-3) stable lane 확정 / freeze / 재배정
    # =========================================================
    def _update_lane_confirmation(self, tid, raw_lane, frame_id):
        """
        [최종 규칙]

        1) stable lane이 아직 없을 때
           - 최근 raw lane history(최대 100프레임)
           - majority_lane 계산
           - 현재 raw lane streak 계산
           - 아래 조건을 모두 만족하면 stable lane 확정
             a. 유효 표본 수 >= LANE_MIN_SAMPLES
             b. majority_ratio >= LANE_MAJORITY_RATIO
             c. raw lane이 8프레임 이상 연속
             d. majority_lane == 현재 streak lane(raw_lane)

        2) stable lane이 이미 있을 때
           - freeze 중이면 무조건 stable lane 유지
           - freeze 종료 후
             다른 raw lane이 12프레임 이상 연속
             AND 최근 history majority도 그 lane
             AND majority_ratio도 충분
             이면 재배정
        """
        # 최근 raw lane history 누적
        self.lane_raw_history[tid].append(raw_lane)

        # 현재 raw lane streak 계산
        streak_len = self._update_lane_streak(tid, raw_lane)

        # 최근 history 다수결 계산
        majority_lane, majority_count, majority_ratio, total_valid = self._get_lane_majority_info(tid)

        current_stable = self.current_lane_stable.get(tid, None)
        freeze_until = self.lane_freeze_until.get(tid, -1)

        # -----------------------------------------------------
        # A) stable lane이 아직 없을 때 -> 최초 확정
        # -----------------------------------------------------
        if current_stable is None:
            can_confirm = (
                raw_lane is not None and
                total_valid >= self.LANE_MIN_SAMPLES and
                majority_lane is not None and
                majority_ratio >= self.LANE_MAJORITY_RATIO and
                streak_len >= self.LANE_CONFIRM_STREAK and
                majority_lane == raw_lane
            )

            if can_confirm:
                self.current_lane_stable[tid] = majority_lane
                self.lane_stable_since[tid] = frame_id
                self.lane_freeze_until[tid] = frame_id + self.LANE_FREEZE_FRAMES
                self.lane_change_memory[tid].append(majority_lane)

            return self.current_lane_stable.get(tid, None)

        # -----------------------------------------------------
        # B) stable lane이 이미 있을 때
        # -----------------------------------------------------

        # B-1) freeze 중이면 lane 유지
        if frame_id <= freeze_until:
            return current_stable

        # B-2) freeze 종료 후 -> 강한 조건일 때만 재배정
        can_reassign = (
            raw_lane is not None and
            raw_lane != current_stable and
            total_valid >= self.LANE_MIN_SAMPLES and
            majority_lane is not None and
            majority_ratio >= self.LANE_MAJORITY_RATIO and
            streak_len >= self.LANE_REASSIGN_STREAK and
            majority_lane == raw_lane
        )

        if can_reassign:
            self.current_lane_stable[tid] = raw_lane
            self.lane_stable_since[tid] = frame_id
            self.lane_freeze_until[tid] = frame_id + self.LANE_FREEZE_FRAMES
            self.lane_change_memory[tid].append(raw_lane)

        return self.current_lane_stable.get(tid, None)

    # =========================================================
    # 8) 차선 추정 전체 갱신
    # =========================================================
    def _update_lane_estimation(self, frame_id, tracks):
        stable_models = []

        # 1) 안정적으로 움직인 차량만 궤적 모델 생성
        for t in tracks:
            tid = t["id"]
            pts = self.track_history.get(tid, [])

            stable = self._is_stable_moving_track(pts)
            self.track_stable_motion[tid] = stable

            if not stable:
                continue

            model, rmse = self._fit_trajectory_model(pts)
            if model is None:
                continue

            self.track_models[tid] = model
            self.track_fit_error[tid] = rmse
            stable_models.append((tid, model))

        # 2) 모든 안정 궤적을 군집화해서 centerline 생성
        self.centerlines = self._cluster_track_models(stable_models)

        # 3) 각 track에 raw lane 할당 후 stable lane 업데이트
        lane_map = {}
        raw_lane_map = {}
        lane_debug = {}

        for t in tracks:
            tid = t["id"]
            pts = self.track_history.get(tid, [])

            if len(pts) == 0:
                lane_map[tid] = None
                raw_lane_map[tid] = None
                lane_debug[tid] = {}
                continue

            current_point = pts[-1]

            # 현재 프레임 raw lane 추정
            raw_lane = self._assign_lane_raw(current_point)
            self.current_lane_raw[tid] = raw_lane
            raw_lane_map[tid] = raw_lane

            # stable lane 확정 / freeze / 재배정
            stable_lane = self._update_lane_confirmation(tid, raw_lane, frame_id)
            lane_map[tid] = stable_lane

            # 디버그 정보 저장
            majority_lane, majority_count, majority_ratio, total_valid = self._get_lane_majority_info(tid)
            lane_debug[tid] = {
                "raw_lane": raw_lane,
                "stable_lane": stable_lane,
                "majority_lane": majority_lane,
                "majority_count": majority_count,
                "majority_ratio": round(majority_ratio, 3),
                "total_valid_samples": total_valid,
                "streak_len": self.lane_same_streak.get(tid, 0),
                "freeze_until": self.lane_freeze_until.get(tid, -1),
                "stable_since": self.lane_stable_since.get(tid, None),
            }

        return lane_map, raw_lane_map, lane_debug

    # =========================================================
    # 외부 호출 메인 함수
    # =========================================================
    def update(self, frame_id, tracks):
        """
        입력:
            frame_id : 프레임 번호
            tracks   : [
                {"id": tid, "bbox": (x1, y1, x2, y2)},
                ...
            ]

        출력:
            analysis dict
        """

        # 1) frame_height 대략 보정
        self._update_frame_height_from_tracks(tracks)

        # 2) boxes / speeds / avg_speed
        boxes, speeds, avg_speed = self._update_tracks_and_speeds(tracks)

        # 3) lane estimation
        lane_map, raw_lane_map, lane_debug = self._update_lane_estimation(frame_id, tracks)

        # 4) 결과 정리
        analysis = {
            "frame_id": frame_id,
            "vehicle_count": len(tracks),
            "boxes": boxes,
            "speeds": speeds,
            "avg_speed": round(avg_speed, 2),
            "lane_map": lane_map,
            "raw_lane_map": raw_lane_map,
            "lane_count": len(self.centerlines),
            "centerlines": self.centerlines,
            "lane_debug": lane_debug,
        }

        self.last_debug = analysis
        return analysis

    # =========================================================
    # 디버깅용 getter
    # =========================================================
    def get_debug_info(self):
        return self.last_debug

    def get_lane_of_track(self, tid):
        return self.current_lane_stable.get(tid, None)

    def get_centerlines(self):
        return self.centerlines