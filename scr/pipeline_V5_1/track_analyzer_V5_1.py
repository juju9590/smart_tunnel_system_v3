# ==========================================
# 파일명: track_analyzer_V5_1.py
# 설명:
# 공통 추적 분석기
# - track history 관리
# - 차량 속도 계산
# - 궤적 기반 차선 추정
# - 상태/사고 로직이 공통 사용
# ==========================================

import numpy as np
from collections import defaultdict, deque


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

        self.lane_vote_history = defaultdict(lambda: deque(maxlen=5))
        self.lane_change_memory = defaultdict(lambda: deque(maxlen=10))

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
        }

        # -----------------------------
        # 파라미터
        # -----------------------------
        self.frame_height = 720
        self.ROI_Y1_RATIO = 0.30
        self.ROI_Y2_RATIO = 0.80

        self.MAX_HISTORY = 60
        self.FIT_MIN_POINTS = 30
        self.LANE_CONFIRM_FRAMES = 5
        self.MIN_TOTAL_MOTION = 35
        self.MAX_DY_JUMP = 40
        self.MAX_DX_JUMP = 60

        self.MAX_SPEED = 20
        self.SPEED_JUMP_LIMIT = 10

        self.LINEAR_CLUSTER_THR = 0.085
        self.QUAD_CLUSTER_THR = 0.12

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

    def _update_lane_confirmation(self, tid, raw_lane):
        self.lane_vote_history[tid].append(raw_lane)

        if len(self.lane_vote_history[tid]) < self.LANE_CONFIRM_FRAMES:
            return self.current_lane_stable.get(tid, None)

        last_votes = list(self.lane_vote_history[tid])

        if all(v == last_votes[0] and v is not None for v in last_votes):
            new_lane = last_votes[0]
            self.current_lane_stable[tid] = new_lane
            self.lane_change_memory[tid].append(new_lane)
            return new_lane

        return self.current_lane_stable.get(tid, None)

    # =========================================================
    # 8) 차선 추정 전체 갱신
    # =========================================================
    def _update_lane_estimation(self, tracks):
        stable_models = []

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

        self.centerlines = self._cluster_track_models(stable_models)

        lane_map = {}
        raw_lane_map = {}

        for t in tracks:
            tid = t["id"]
            pts = self.track_history.get(tid, [])

            if len(pts) == 0:
                lane_map[tid] = None
                raw_lane_map[tid] = None
                continue

            current_point = pts[-1]
            raw_lane = self._assign_lane_raw(current_point)
            self.current_lane_raw[tid] = raw_lane
            raw_lane_map[tid] = raw_lane

            stable_lane = self._update_lane_confirmation(tid, raw_lane)
            lane_map[tid] = stable_lane

        return lane_map, raw_lane_map

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
        lane_map, raw_lane_map = self._update_lane_estimation(tracks)

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