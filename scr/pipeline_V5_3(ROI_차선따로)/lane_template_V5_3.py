# ==========================================
# 파일명: lane_template_V5_3.py
# 설명:
# V5_3 2차 군집화 기반 차선(군집) 추정 모듈
#
# [목적]
# - 1차 군집화만으로는 비슷한 흐름이 잘게 쪼개질 수 있음
# - 그래서 1차 군집 대표선끼리 다시 2차 군집화하여
#   최종 대표 집단을 만든다
#
# [현재 버전 규칙]
# 1) bootstrap 1회만 수행
#    - 초기 100프레임 동안 안정 차량 궤적 수집
# 2) 차량 궤적을 선형(linear) 모델로만 피팅
# 3) 1차 군집화:
#    - 차량 궤적끼리 거리 계산
#    - 거리 < 임계값이면 같은 군집
# 4) 2차 군집화:
#    - 1차 군집 대표선끼리 다시 거리 계산
#    - 가까운 대표선끼리 다시 합침
# 5) bootstrap 이후 재평가 없음
# 6) 현재 프레임 차량은 최종 대표 집단 중 가장 가까운 것에 임시 할당
#
# [주의]
# - lane_map은 아직 "최종 차선 확정"이 아니라
#   "최종 대표 집단 할당 결과"로 해석해야 함
# ==========================================

import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


class LaneTemplateEstimator:
    def __init__(self, output_dir=None):
        # -----------------------------
        # 상태
        # -----------------------------
        self.current_template = []          # 최종 대표 집단 목록
        self.template_confirmed = False
        self.phase = "BOOTSTRAP"

        # -----------------------------
        # 수집 메모리
        # -----------------------------
        self.track_models = {}              # tid -> fitted linear model
        self.track_fit_error = {}           # tid -> rmse
        self.track_stable_motion = {}       # tid -> 안정 이동 여부
        self.collected_track_ids = set()
        self.collected_track_points = {}    # tid -> [(x, y), ...]

        # -----------------------------
        # 디버그
        # -----------------------------
        self.last_debug = {
            "phase": "BOOTSTRAP",
            "template_confirmed": False,
            "lane_count": 0,
            "template": [],
            "clusters_stage1": [],
            "clusters_stage2": [],
            "lane_map": {},
            "raw_lane_map": {},
        }

        # -----------------------------
        # 파라미터
        # -----------------------------
        self.BOOTSTRAP_FRAMES = 100
        self.FIT_MIN_POINTS = 30
        self.MIN_TOTAL_MOTION = 35
        self.MAX_DY_JUMP = 40
        self.MAX_DX_JUMP = 60

        # 선형 피팅 품질 필터
        self.MAX_FIT_RMSE = 35.0

        # 1차 군집: 차량 궤적끼리
        self.LINEAR_CLUSTER_THR_STAGE1 = 0.12

        # 2차 군집: 1차 대표선끼리
        # 1차보다 조금 더 완화해서 비슷한 흐름을 다시 묶어줌
        self.LINEAR_CLUSTER_THR_STAGE2 = 0.22

        # 그래프 저장 경로
        self.output_dir = output_dir or os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)

    # =========================================================
    # 1) 안정 이동 차량 판정
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
    # 2) 차량 궤적 선형 피팅
    #    x = a*y + b
    # =========================================================
    def _fit_trajectory_model(self, pts):
        recent = pts[-self.FIT_MIN_POINTS:]
        ys = np.array([p[1] for p in recent], dtype=np.float32)
        xs = np.array([p[0] for p in recent], dtype=np.float32)

        if len(np.unique(ys)) < 6:
            return None, 1e9

        try:
            lin_coef = np.polyfit(ys, xs, 1)
            lin_pred = np.polyval(lin_coef, ys)
            lin_rmse = float(np.sqrt(np.mean((xs - lin_pred) ** 2)))
        except Exception:
            return None, 1e9

        return {
            "type": "linear",
            "coef": lin_coef.astype(float).tolist()
        }, lin_rmse

    def _predict_x(self, model, y):
        if model is None:
            return None

        a, b = model["coef"]
        return a * y + b

    # =========================================================
    # 3) 모델 거리 계산
    # =========================================================
    def _coef_vector(self, model, frame_height):
        a, b = model["coef"]
        return np.array([a, b / max(frame_height, 1)], dtype=np.float32)

    def _model_distance(self, model1, model2, lane_y1, lane_y2, frame_height):
        """
        두 선형 궤적이 얼마나 비슷한지 계산
        - 계수 벡터 거리
        - ROI 여러 지점에서 x 차이
        """
        v1 = self._coef_vector(model1, frame_height)
        v2 = self._coef_vector(model2, frame_height)
        coef_dist = float(np.linalg.norm(v1 - v2))

        sample_ys = np.linspace(lane_y1, lane_y2, 5)
        diffs = []

        for y in sample_ys:
            x1 = self._predict_x(model1, y)
            x2 = self._predict_x(model2, y)
            diffs.append(abs(x1 - x2))

        shape_dist = float(np.mean(diffs)) / max(frame_height, 1)

        return coef_dist * 0.5 + shape_dist * 0.5

    # =========================================================
    # 4) 대표 모델 계산
    # =========================================================
    def _aggregate_models(self, models):
        if not models:
            return None

        arr = np.array([m["coef"] for m in models], dtype=np.float32)
        coef = np.median(arr, axis=0).tolist()

        return {
            "type": "linear",
            "coef": coef
        }

    # =========================================================
    # 5) bootstrap 동안 안정 차량 궤적 수집
    # =========================================================
    def _collect_stable_models(self, track_history):
        for tid, pts in track_history.items():
            stable = self._is_stable_moving_track(pts)
            self.track_stable_motion[tid] = stable

            if not stable:
                continue

            model, rmse = self._fit_trajectory_model(pts)
            if model is None:
                continue

            if rmse > self.MAX_FIT_RMSE:
                continue

            self.track_models[tid] = model
            self.track_fit_error[tid] = rmse
            self.collected_track_ids.add(tid)
            self.collected_track_points[tid] = list(pts)

    # =========================================================
    # 6) 1차 군집화
    #    차량 궤적끼리 군집화
    # =========================================================
    def _cluster_track_models_stage1(self, lane_y1, lane_y2, frame_height):
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

            for cluster in clusters:
                dist = self._model_distance(
                    model,
                    cluster["rep_model"],
                    lane_y1,
                    lane_y2,
                    frame_height
                )

                if dist < self.LINEAR_CLUSTER_THR_STAGE1:
                    cluster["items"].append((tid, model))
                    cluster["rep_model"] = self._aggregate_models(
                        [m for _, m in cluster["items"]]
                    )
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
    # 7) 1차 군집 정보 정리
    # =========================================================
    def _extract_cluster_info_stage1(self, clusters, lane_y1, lane_y2):
        if not clusters:
            return []

        y_mid = (lane_y1 + lane_y2) / 2.0
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

        cluster_info.sort(key=lambda x: x["x_mid"])
        return cluster_info

    # =========================================================
    # 8) 2차 군집화
    #    1차 대표선끼리 다시 군집화
    # =========================================================
    def _cluster_representatives_stage2(self, cluster_info_stage1, lane_y1, lane_y2, frame_height):
        """
        1차 군집 대표선들끼리 다시 거리 계산해서
        비슷한 흐름은 한 번 더 합친다.
        """
        if not cluster_info_stage1:
            return []

        clusters2 = []

        for c in cluster_info_stage1:
            model = c["rep_model"]
            assigned = False

            for cluster in clusters2:
                dist = self._model_distance(
                    model,
                    cluster["rep_model"],
                    lane_y1,
                    lane_y2,
                    frame_height
                )

                if dist < self.LINEAR_CLUSTER_THR_STAGE2:
                    cluster["items"].append(c)
                    cluster["rep_model"] = self._aggregate_models(
                        [item["rep_model"] for item in cluster["items"]]
                    )
                    assigned = True
                    break

            if not assigned:
                clusters2.append({
                    "cluster_id": len(clusters2),
                    "rep_model": model,
                    "items": [c]
                })

        return clusters2

    # =========================================================
    # 9) 2차 군집 결과 정리
    # =========================================================
    def _extract_cluster_info_stage2(self, clusters2, lane_y1, lane_y2):
        """
        2차 군집 결과를 최종 대표 집단으로 정리
        """
        if not clusters2:
            return []

        y_mid = (lane_y1 + lane_y2) / 2.0
        total_count = 0
        for c in clusters2:
            total_count += sum(item["count"] for item in c["items"])

        final_info = []
        for c in clusters2:
            agg_model = self._aggregate_models([item["rep_model"] for item in c["items"]])
            x_mid = self._predict_x(agg_model, y_mid)

            merged_count = sum(item["count"] for item in c["items"])

            merged_ids = []
            for item in c["items"]:
                merged_ids.extend(item["member_ids"])

            final_info.append({
                "cluster_id": c["cluster_id"],
                "rep_model": agg_model,
                "count": merged_count,
                "ratio": merged_count / max(total_count, 1),
                "x_mid": x_mid,
                "member_ids": merged_ids,
                "source_cluster_ids": [item["cluster_id"] for item in c["items"]],
            })

        final_info.sort(key=lambda x: x["x_mid"])
        return final_info

    # =========================================================
    # 10) 현재 차량을 가장 가까운 최종 대표 집단에 임시 할당
    # =========================================================
    def _assign_lane(self, point, template):
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
    # 11) bootstrap 결과 그래프 저장
    # =========================================================
    def save_trajectory_plot(self, lane_y1, lane_y2, filename=None):
        if not self.collected_track_points:
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"graph_v5_3_{timestamp}.png"

        save_path = os.path.join(self.output_dir, filename)

        plt.figure(figsize=(8, 6))

        # 차량 궤적
        for tid, pts in self.collected_track_points.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, linewidth=1, alpha=0.6)
            plt.text(xs[-1], ys[-1], str(tid), fontsize=8)

        # 최종 대표 집단
        sample_ys = np.linspace(lane_y1, lane_y2, 50)

        for lane in self.current_template:
            model = lane["rep_model"]
            lane_id = lane["lane_id"]

            xs = [self._predict_x(model, y) for y in sample_ys]
            plt.plot(xs, sample_ys, linewidth=3, label=f"CLUSTER {lane_id}")

        plt.gca().invert_yaxis()
        plt.title("Vehicle Trajectories and Cluster Representatives")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        print("📊 그래프 저장:", save_path)
        return save_path
    
    def save_trajectory_plot_stage1(self, lane_y1, lane_y2, cluster_info_stage1, filename=None):
        """
    1차 군집화 결과 저장
    - 차량 궤적
    - 1차 군집 대표선
        """

        if not self.collected_track_points:
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"graph_stage1_v5_3_{timestamp}.png"

        save_path = os.path.join(self.output_dir, filename)

        plt.figure(figsize=(8, 6))

        # 차량 궤적
        for tid, pts in self.collected_track_points.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, linewidth=1, alpha=0.6)
            plt.text(xs[-1], ys[-1], str(tid), fontsize=8)

        # 1차 군집 대표선
        sample_ys = np.linspace(lane_y1, lane_y2, 50)

        for c in cluster_info_stage1:
            model = c["rep_model"]
            cid = c["cluster_id"]

            xs = [self._predict_x(model, y) for y in sample_ys]
            plt.plot(xs, sample_ys, linewidth=3, label=f"STAGE1 {cid}")

        plt.gca().invert_yaxis()
        plt.title("Vehicle Trajectories and Stage1 Cluster Representatives")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        print("📊 1차 군집 그래프 저장:", save_path)
        return save_path
    
    def save_trajectory_plot_stage2(self, lane_y1, lane_y2, filename=None):
        """
    2차 군집화 결과 저장
    - 차량 궤적
    - 최종 대표 집단(2차 군집 결과)
        """

        if not self.collected_track_points:
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"graph_stage2_v5_3_{timestamp}.png"

        save_path = os.path.join(self.output_dir, filename)

        plt.figure(figsize=(8, 6))

        # 차량 궤적
        for tid, pts in self.collected_track_points.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, linewidth=1, alpha=0.6)
            plt.text(xs[-1], ys[-1], str(tid), fontsize=8)

        # 2차 군집 대표선
        sample_ys = np.linspace(lane_y1, lane_y2, 50)

        for lane in self.current_template:
            model = lane["rep_model"]
            lane_id = lane["lane_id"]

            xs = [self._predict_x(model, y) for y in sample_ys]
            plt.plot(xs, sample_ys, linewidth=3, label=f"STAGE2 {lane_id}")

        plt.gca().invert_yaxis()
        plt.title("Vehicle Trajectories and Stage2 Cluster Representatives")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        print("📊 2차 군집 그래프 저장:", save_path)
        return save_path
    


    # =========================================================
    # 12) 외부 호출 메인 함수
    # =========================================================
    def update(self, frame_id, analysis):
        track_history = analysis["track_history"]
        track_points = analysis["track_points"]
        frame_height = analysis["frame_height"]

        # ROI는 속도/상태/사고 판단용으로 유지한다.
        # 차선 추정은 화면 전체 흐름을 보기 위해 별도 범위를 사용한다.
        lane_y1 = int(frame_height * 0.20)
        lane_y2 = int(frame_height * 0.95)

        # -----------------------------------------------------
        # A) bootstrap 단계
        # -----------------------------------------------------
        if not self.template_confirmed:
            self.phase = "BOOTSTRAP"

            self._collect_stable_models(track_history)

            if frame_id >= self.BOOTSTRAP_FRAMES:
                # 1차 군집화
                clusters_stage1 = self._cluster_track_models_stage1(
                    lane_y1,
                    lane_y2,
                    frame_height
                )
                cluster_info_stage1 = self._extract_cluster_info_stage1(
                    clusters_stage1,
                    lane_y1,
                    lane_y2
                )

                # 2차 군집화
                clusters_stage2 = self._cluster_representatives_stage2(
                    cluster_info_stage1,
                    lane_y1,
                    lane_y2,
                    frame_height
                )
                final_cluster_info = self._extract_cluster_info_stage2(
                    clusters_stage2,
                    lane_y1,
                    lane_y2
                )

                # 최종 대표 집단을 current_template로 저장
                self.current_template = []
                for idx, c in enumerate(final_cluster_info):
                    self.current_template.append({
                        "lane_id": idx,
                        "cluster_id": c["cluster_id"],
                        "rep_model": c["rep_model"],
                        "x_mid": c["x_mid"],
                        "count": c["count"],
                        "ratio": c["ratio"],
                        "member_ids": c["member_ids"],
                        "source_cluster_ids": c["source_cluster_ids"],
                    })

                self.template_confirmed = True
                self.phase = "CLUSTER_VIEW"

                # 1차 군집 그래프 저장
                self.save_trajectory_plot_stage1(
                    lane_y1=lane_y1,
                    lane_y2=lane_y2,
                    cluster_info_stage1=cluster_info_stage1
                )

                # 2차 군집 그래프 저장
                self.save_trajectory_plot_stage2(
                    lane_y1=lane_y1,
                    lane_y2=lane_y2
                )

                clusters_stage1_debug = cluster_info_stage1
                clusters_stage2_debug = final_cluster_info
            else:
                clusters_stage1_debug = []
                clusters_stage2_debug = []

        # -----------------------------------------------------
        # B) bootstrap 이후
        # -----------------------------------------------------
        else:
            self.phase = "CLUSTER_VIEW"
            clusters_stage1_debug = []
            clusters_stage2_debug = []

        # -----------------------------------------------------
        # C) 현재 최종 대표 집단 기준 임시 할당
        # -----------------------------------------------------
        lane_map = {}
        raw_lane_map = {}

        for tid, pt in track_points.items():
            lane_id = self._assign_lane(pt, self.current_template)
            lane_map[tid] = lane_id
            raw_lane_map[tid] = lane_id

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
            "clusters_stage1": clusters_stage1_debug,
            "clusters_stage2": clusters_stage2_debug,
            "clusters": clusters_stage2_debug,
        }

        self.last_debug = {
            "phase": self.phase,
            "template_confirmed": self.template_confirmed,
            "lane_count": len(self.current_template),
            "template": self.current_template,
            "clusters_stage1": clusters_stage1_debug,
            "clusters_stage2": clusters_stage2_debug,
            "lane_map": lane_map,
            "raw_lane_map": raw_lane_map,
        }

        return result

    def get_debug_info(self):
        return self.last_debug
