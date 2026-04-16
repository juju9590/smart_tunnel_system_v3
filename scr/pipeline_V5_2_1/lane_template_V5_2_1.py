# ==========================================
# 파일명: lane_template_V5_2_1.py
# 설명:
# 차선 추정 전용 모듈
#
# [최종 단순 규칙]
# 1) bootstrap 1회만 수행
#    - 초기 100프레임 동안 차량 궤적 수집
# 2) 차량 궤적끼리 거리 계산
#    - 궤적 거리 < 임계값이면 같은 군집
# 3) 군집 비중 계산
#    - 군집 비중 40% 이상이면 대표 차선
#    - 아니면 잡음
# 4) 대표 차선은 최대 4개만 사용
# 5) bootstrap 이후에는 재평가 없음
# 6) 각 차량은 대표 차선 중 가장 가까운 차선으로 할당
#
# [추가]
# - 차량 궤적 그래프 저장 기능 포함
#   => bootstrap이 끝날 때 한 번 저장
# ==========================================

import os
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt


class LaneTemplateEstimator:
    def __init__(self, output_dir=None):
        # -----------------------------
        # 차선 template 상태
        # -----------------------------
        self.current_template = []          # 최종 대표 차선 목록
        self.template_confirmed = False     # bootstrap 완료 여부
        self.phase = "BOOTSTRAP"            # 현재 단계

        # -----------------------------
        # 수집 메모리
        # -----------------------------
        self.track_models = {}              # tid -> fitted trajectory model
        self.track_fit_error = {}           # tid -> rmse
        self.track_stable_motion = {}       # tid -> 안정 이동 여부
        self.collected_track_ids = set()    # bootstrap 동안 수집된 안정 차량 id

        # 그래프용 원본 포인트 저장
        self.collected_track_points = {}    # tid -> [(x, y), ...]

        # -----------------------------
        # 디버그
        # -----------------------------
        self.last_debug = {
            "phase": "BOOTSTRAP",
            "template_confirmed": False,
            "lane_count": 0,
            "template": [],
            "clusters": [],
            "lane_map": {},
            "raw_lane_map": {},
        }

        # -----------------------------
        # 파라미터
        # -----------------------------
        self.BOOTSTRAP_FRAMES = 100         # 초기 100프레임만 사용
        self.FIT_MIN_POINTS = 30            # 궤적 모델 생성 최소 점 수
        self.MIN_TOTAL_MOTION = 35          # 총 이동량 최소값
        self.MAX_DY_JUMP = 40               # 프레임 간 y 점프 허용 한계
        self.MAX_DX_JUMP = 60               # 프레임 간 x 점프 허용 한계

        # 궤적 거리 임계값
        self.LINEAR_CLUSTER_THR = 0.12
        self.QUAD_CLUSTER_THR = 0.16

        # 대표 차선 선택 규칙
        self.MAIN_CLUSTER_RATIO = 0.40      # 군집 비중 40% 이상
        self.MAX_LANE_COUNT = 4             # 최대 4개 차선만 사용

        # 그래프 저장 경로
        self.output_dir = output_dir or os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)

    # =========================================================
    # 1) 안정적으로 움직이는 차량 판정
    # =========================================================
    def _is_stable_moving_track(self, pts):
        """
        차량 궤적이 '차선 추정용으로 쓸 만한지' 판단
        너무 짧거나, 거의 안 움직였거나, 점프가 심하면 제외
        """
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
    # 2) 차량 궤적을 수학 모델로 피팅
    #    x = a*y + b 또는 x = a*y^2 + b*y + c
    # =========================================================
    def _fit_trajectory_model(self, pts):
        """
        차량 궤적을 선형으로만 표현
        같은 흐름 차량끼리 비교하기 쉽게 모델화하는 단계
        """
        recent = pts[-self.FIT_MIN_POINTS:]
        ys = np.array([p[1] for p in recent], dtype=np.float32)
        xs = np.array([p[0] for p in recent], dtype=np.float32)

        if len(np.unique(ys)) < 6:
            return None, 1e9

        # 선형 모델
        try:
            lin_coef = np.polyfit(ys, xs, 1)
            lin_pred = np.polyval(lin_coef, ys)
            lin_rmse = float(np.sqrt(np.mean((xs - lin_pred) ** 2)))
        except Exception:
            lin_coef = None
            lin_rmse = 1e9

        return {
            "type": "linear",
            "coef": lin_coef.astype(float).tolist()
        }, lin_rmse

    def _predict_x(self, model, y):
        """
        주어진 y 위치에서 해당 궤적 모델의 x 예측값 계산
        """
        if model is None:
            return None

        if model["type"] == "linear":
            a, b = model["coef"]
            return a * y + b
        else:
            a, b, c = model["coef"]
            return a * (y ** 2) + b * y + c

    # =========================================================
    # 3) 궤적 간 거리 계산
    # =========================================================
    def _coef_vector(self, model, frame_height):
        """
        선형/2차 모델을 공통 비교 가능한 벡터로 변환
        """
        if model["type"] == "linear":
            a, b = model["coef"]
            return np.array([0.0, a, b / max(frame_height, 1)], dtype=np.float32)

        a, b, c = model["coef"]
        return np.array([
            a * frame_height,
            b,
            c / max(frame_height, 1)
        ], dtype=np.float32)

    def _model_distance(self, model1, model2, roi_y1, roi_y2, frame_height):
        """
        두 차량 궤적이 얼마나 비슷한지 수치화
        - 계수 벡터 거리
        - ROI 여러 지점에서의 x 차이
        이 둘을 합쳐서 최종 거리 계산
        """
        v1 = self._coef_vector(model1, frame_height)
        v2 = self._coef_vector(model2, frame_height)
        coef_dist = float(np.linalg.norm(v1 - v2))

        sample_ys = np.linspace(roi_y1, roi_y2, 5)
        diffs = []

        for y in sample_ys:
            x1 = self._predict_x(model1, y)
            x2 = self._predict_x(model2, y)
            diffs.append(abs(x1 - x2))

        shape_dist = float(np.mean(diffs)) / max(frame_height, 1)

        # 둘을 반반 반영
        return coef_dist * 0.5 + shape_dist * 0.5

    # =========================================================
    # 4) 군집 대표 모델 계산
    # =========================================================
    def _aggregate_models(self, models):
        """
        같은 군집에 속한 여러 차량 궤적을 대표하는 중심 모델 생성
        """
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
    # 5) bootstrap 동안 안정 차량 궤적 수집
    # =========================================================
    def _collect_stable_models(self, track_history):
        """
        공통분석기(track_analyzer)에서 받은 track_history 중
        안정적으로 움직이는 차량만 골라 궤적 모델 저장
        """
        for tid, pts in track_history.items():
            stable = self._is_stable_moving_track(pts)
            self.track_stable_motion[tid] = stable

            if not stable:
                continue

            model, rmse = self._fit_trajectory_model(pts)
            if model is None:
                continue

            self.track_models[tid] = model
            self.track_fit_error[tid] = rmse
            self.collected_track_ids.add(tid)
            self.collected_track_points[tid] = list(pts)

    # =========================================================
    # 6) 비슷한 궤적끼리 군집화
    # =========================================================
    def _cluster_models(self, roi_y1, roi_y2, frame_height):
        """
        궤적 거리 < 임계값 이면 같은 군집으로 묶음
        """
        stable_models = [
            (tid, self.track_models[tid])
            for tid in sorted(self.collected_track_ids)
            if tid in self.track_models
        ]

        if not stable_models:
            return []

        clusters = []

        for tid, model in stable_models:
            assigned = False
            thr = self.QUAD_CLUSTER_THR if model["type"] == "quadratic" else self.LINEAR_CLUSTER_THR

            for cluster in clusters:
                dist = self._model_distance(model, cluster["rep_model"], roi_y1, roi_y2, frame_height)
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

        return clusters

    # =========================================================
    # 7) 군집 비중 기반 대표 차선 추출
    # =========================================================
    def _extract_template_from_clusters(self, clusters, roi_y1, roi_y2):
        """
        단순 모드:
    - 군집 비중 40% 조건 제거
    - 최대 차선 수 제한 제거
    - 군집화된 결과를 전부 대표 차선으로 사용

    목적:
    - 현재 차량 흐름이 실제로 몇 개 군집으로 나뉘는지 먼저 확인
    - 대표 차선(centerline)이 어떻게 형성되는지 육안으로 확인
        """
        if not clusters:
            return [], []

        y_mid = (roi_y1 + roi_y2) / 2.0
        total_tracks = sum(len(c["items"]) for c in clusters)

        cluster_info = []
        for c in clusters:
            agg_model = self._aggregate_models([m for _, m in c["items"]])
            x_mid = self._predict_x(agg_model, y_mid)

            cluster_info.append({
                "cluster_id": c["cluster_id"],
                "rep_model": agg_model,
                "count": len(c["items"]),
                "ratio": len(c["items"]) / max(total_tracks, 1),
                "x_mid": x_mid,
                "member_ids": [tid for tid, _ in c["items"]],
            })

        # 왼쪽 -> 오른쪽 순서로 정렬
        cluster_info.sort(key=lambda x: x["x_mid"])

        # 모든 군집을 그대로 대표 차선으로 사용
        template = []
        for idx, c in enumerate(cluster_info):
            template.append({
                "lane_id": idx,
                "rep_model": c["rep_model"],
                "x_mid": c["x_mid"],
                "count": c["count"],
                "ratio": c["ratio"],
                "member_ids": c["member_ids"],
            })

        return cluster_info, template

    # =========================================================
    # 8) 대표 차선 기반 차량 lane 할당
    # =========================================================
    def _assign_lane(self, point, template):
        """
        가장 가까운 대표 차선에 무조건 붙이는 방식
        """
        if not template:
            return None

        x, y = point
        best_lane = None
        best_dist = 1e9

        for lane in template:
            model = lane["rep_model"]
            cx = self._predict_x(model, y)
            dist = abs(x - cx)

            if dist < best_dist:
                best_dist = dist
                best_lane = lane["lane_id"]

        return best_lane

    # =========================================================
    # 9) 차량 궤적 그래프 저장
    # =========================================================
    def save_trajectory_plot(self, roi_y1, roi_y2, frame_height, filename="lane_bootstrap_plot.png"):
        """
        bootstrap 시점에 수집된 안정 차량 궤적과 대표 차선을 그래프로 저장

        그래프 해석:
        - 얇은 선: 차량별 궤적
        - 굵은 선: 대표 차선(centerline)
        """
        if not self.collected_track_points:
            return None

        save_path = os.path.join(self.output_dir, filename)

        plt.figure(figsize=(8, 6))

        # 1) 차량 궤적 먼저 그림
        for tid, pts in self.collected_track_points.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, linewidth=1, alpha=0.6)
            plt.text(xs[-1], ys[-1], str(tid), fontsize=8)

        # 2) 대표 차선(centerline) 그림
        sample_ys = np.linspace(roi_y1, roi_y2, 50)

        for lane in self.current_template:
            model = lane["rep_model"]
            lane_id = lane["lane_id"]

            xs = [self._predict_x(model, y) for y in sample_ys]
            plt.plot(xs, sample_ys, linewidth=3, label=f"LANE {lane_id}")

        # 영상 좌표계처럼 위가 작은 y, 아래가 큰 y니까 뒤집기
        plt.gca().invert_yaxis()

        plt.title("Vehicle Trajectories and Lane Templates")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        return save_path

    # =========================================================
    # 10) 외부 호출 메인 함수
    # =========================================================
    def update(self, frame_id, analysis):
        """
        analysis:
            track_analyzer 결과 dict
        """

        track_history = analysis["track_history"]
        track_points = analysis["track_points"]
        roi_y1 = analysis["roi_y1"]
        roi_y2 = analysis["roi_y2"]
        frame_height = analysis["frame_height"]

        # -----------------------------------------------------
        # A) bootstrap 단계
        # -----------------------------------------------------
        if not self.template_confirmed:
            self.phase = "BOOTSTRAP"

            # 초기 100프레임 동안 안정 차량 궤적 수집
            self._collect_stable_models(track_history)

            # 100프레임 되면 template 1회 생성
            if frame_id >= self.BOOTSTRAP_FRAMES:
                clusters = self._cluster_models(roi_y1, roi_y2, frame_height)
                clusters_debug, new_template = self._extract_template_from_clusters(clusters, roi_y1, roi_y2)

                self.current_template = new_template
                self.template_confirmed = True
                self.phase = "ASSIGN"

                # bootstrap 완료 시 차량 궤적 그래프 저장
                self.save_trajectory_plot(
                    roi_y1=roi_y1,
                    roi_y2=roi_y2,
                    frame_height=frame_height,
                    filename=f"lane_bootstrap_plot_f{frame_id}.png"
                )
            else:
                clusters_debug = []

        # -----------------------------------------------------
        # B) bootstrap 이후
        # -----------------------------------------------------
        else:
            self.phase = "ASSIGN"
            clusters_debug = []

        # -----------------------------------------------------
        # C) 현재 template 기준 lane 할당
        # -----------------------------------------------------
        lane_map = {}
        raw_lane_map = {}

        for tid, pt in track_points.items():
            lane_id = self._assign_lane(pt, self.current_template)
            lane_map[tid] = lane_id
            raw_lane_map[tid] = lane_id   # 현재 구조에서는 raw/stable 동일

        result = {
            "lane_map": lane_map,
            "raw_lane_map": raw_lane_map,
            "lane_count": len(self.current_template),
            "centerlines": self.current_template,
            "lane_debug": {
                tid: {
                    "raw_lane": raw_lane_map.get(tid),
                    "stable_lane": lane_map.get(tid),
                } for tid in lane_map
            },
            "template_phase": self.phase,
            "template_confirmed": self.template_confirmed,
            "clusters": clusters_debug,
        }

        self.last_debug = {
            "phase": self.phase,
            "template_confirmed": self.template_confirmed,
            "lane_count": len(self.current_template),
            "template": self.current_template,
            "clusters": clusters_debug,
            "lane_map": lane_map,
            "raw_lane_map": raw_lane_map,
        }

        return result

    def get_debug_info(self):
        return self.last_debug