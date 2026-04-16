# -*- coding: utf-8 -*-
"""
파일명: lane_estimator_test2.py

설명:
- 공통 lane template 방식 테스트 코드
- 초기 100프레임 동안 차량 궤적을 모아 lane template 생성
- 비슷한 궤적끼리 cluster
- cluster 비중이 40% 이상인 것만 대표 차선으로 채택
- template 확정 후 100프레임 freeze
- freeze 종료 후 다시 100프레임 수집
- 새 template가 기존 대비 50% 이상 바뀌면 갱신

핵심:
- "차량별 lane 확정"이 아니라
- "영상 전체 흐름 기반 공통 lane template" 테스트

실행:
python lane_estimator_test2.py
"""

import math
from collections import defaultdict


class LaneTemplateEstimator:
    def __init__(
        self,
        bootstrap_frames=100,         # 초기 template 생성용 프레임 수
        freeze_frames=100,            # template 확정 후 유지 프레임 수
        min_points_per_track=8,       # 한 track이 궤적으로 인정되기 위한 최소 점 수
        cluster_dist_thr=18.0,        # 궤적 간 거리 threshold
        main_cluster_ratio=0.40,      # 대표 차선 채택 최소 비율
        template_change_ratio=0.50    # 새 template 변경 인정 최소 비율
    ):
        self.bootstrap_frames = bootstrap_frames
        self.freeze_frames = freeze_frames
        self.min_points_per_track = min_points_per_track
        self.cluster_dist_thr = cluster_dist_thr
        self.main_cluster_ratio = main_cluster_ratio
        self.template_change_ratio = template_change_ratio

        self.current_frame = 0

        # 공통 lane template 상태
        self.template_confirmed = False
        self.template_freeze_until = -1
        self.current_template = []   # [{"lane_id":0, "rep_x":..., "count":..., "ratio":...}, ...]

        # bootstrap / 재평가용 track 수집
        self.track_points = defaultdict(list)   # tid -> [(x, y), ...]
        self.phase_start_frame = 1

    # =========================================================
    # 1) 입력 track 누적
    # =========================================================
    def update_tracks(self, frame_idx, tracks):
        """
        tracks:
        [
            {"id": 1, "point": (x, y)},
            {"id": 2, "point": (x, y)},
            ...
        ]
        """
        self.current_frame = frame_idx

        for t in tracks:
            tid = t["id"]
            x, y = t["point"]
            self.track_points[tid].append((x, y))

    # =========================================================
    # 2) track -> trajectory feature
    # =========================================================
    def _track_to_feature(self, pts):
        """
        궤적을 간단한 feature로 바꿈
        여기서는 테스트용으로
        - 시작점 x
        - 끝점 x
        - 평균 x
        - 총 y 변화량
        사용

        실제 프로젝트에서는 polyfit 계수나 sample y에서 x값 사용 가능
        """
        if len(pts) < self.min_points_per_track:
            return None

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        y_span = max(ys) - min(ys)
        if y_span < 10:
            return None

        feature = {
            "start_x": xs[0],
            "end_x": xs[-1],
            "mean_x": sum(xs) / len(xs),
            "y_span": y_span,
            "count": len(pts),
        }
        return feature

    # =========================================================
    # 3) feature 거리 계산
    # =========================================================
    def _feature_distance(self, f1, f2):
        """
        두 궤적 feature 간 거리 계산
        테스트용 단순 거리:
        - mean_x 차이
        - start_x 차이
        - end_x 차이
        """
        d_mean = abs(f1["mean_x"] - f2["mean_x"])
        d_start = abs(f1["start_x"] - f2["start_x"])
        d_end = abs(f1["end_x"] - f2["end_x"])

        dist = 0.5 * d_mean + 0.25 * d_start + 0.25 * d_end
        return dist

    # =========================================================
    # 4) feature 군집화
    # =========================================================
    def _cluster_features(self, features):
        """
        비슷한 궤적 feature끼리 묶기
        매우 단순한 greedy clustering
        """
        clusters = []

        for item in features:
            tid = item["tid"]
            feat = item["feature"]

            assigned = False
            for c in clusters:
                rep_feat = c["rep_feature"]
                dist = self._feature_distance(feat, rep_feat)

                if dist < self.cluster_dist_thr:
                    c["items"].append(item)
                    c["rep_feature"] = self._aggregate_cluster(c["items"])
                    assigned = True
                    break

            if not assigned:
                clusters.append({
                    "items": [item],
                    "rep_feature": feat
                })

        return clusters

    def _aggregate_cluster(self, items):
        """
        cluster 대표 feature 생성
        평균 x 기반
        """
        start_xs = [it["feature"]["start_x"] for it in items]
        end_xs = [it["feature"]["end_x"] for it in items]
        mean_xs = [it["feature"]["mean_x"] for it in items]
        y_spans = [it["feature"]["y_span"] for it in items]
        counts = [it["feature"]["count"] for it in items]

        return {
            "start_x": sum(start_xs) / len(start_xs),
            "end_x": sum(end_xs) / len(end_xs),
            "mean_x": sum(mean_xs) / len(mean_xs),
            "y_span": sum(y_spans) / len(y_spans),
            "count": sum(counts) / len(counts),
        }

    # =========================================================
    # 5) 대표 차선(template) 추출
    # =========================================================
    def _extract_main_lanes(self):
        """
        현재까지 모은 track_points로부터:
        1) 각 차량 궤적 feature 생성
        2) feature 군집화
        3) cluster 비중 계산
        4) 40% 이상 cluster만 대표 차선으로 채택
        """
        features = []

        for tid, pts in self.track_points.items():
            feat = self._track_to_feature(pts)
            if feat is None:
                continue
            features.append({
                "tid": tid,
                "feature": feat
            })

        if len(features) == 0:
            return [], []

        clusters = self._cluster_features(features)

        total_tracks = len(features)
        cluster_info = []

        for idx, c in enumerate(clusters):
            cluster_count = len(c["items"])
            ratio = cluster_count / total_tracks if total_tracks > 0 else 0.0
            rep = c["rep_feature"]

            cluster_info.append({
                "cluster_id": idx,
                "count": cluster_count,
                "ratio": ratio,
                "rep_x": rep["mean_x"],
                "member_ids": [it["tid"] for it in c["items"]],
            })

        # 화면 왼쪽 -> 오른쪽 순으로 정렬
        cluster_info.sort(key=lambda x: x["rep_x"])

        # 비중 40% 이상인 군집만 대표 차선으로 채택
        main_lanes = []
        lane_id = 0
        for c in cluster_info:
            if c["ratio"] >= self.main_cluster_ratio:
                main_lanes.append({
                    "lane_id": lane_id,
                    "rep_x": round(c["rep_x"], 2),
                    "count": c["count"],
                    "ratio": round(c["ratio"], 3),
                    "member_ids": c["member_ids"]
                })
                lane_id += 1

        return cluster_info, main_lanes

    # =========================================================
    # 6) template 변경률 계산
    # =========================================================
    def _template_change_score(self, old_template, new_template):
        """
        기존 template와 새 template 차이 계산
        단순 테스트 기준:
        - lane 개수 차이
        - 각 lane rep_x 차이

        0.0 ~ 1.0 사이 정도로 반환
        """
        if not old_template and not new_template:
            return 0.0

        if not old_template or not new_template:
            return 1.0

        old_n = len(old_template)
        new_n = len(new_template)

        if old_n == 0 and new_n == 0:
            return 0.0

        lane_count_change = abs(old_n - new_n) / max(old_n, new_n, 1)

        # 같은 index끼리 rep_x 비교
        compare_n = min(old_n, new_n)
        if compare_n == 0:
            position_change = 1.0
        else:
            diffs = []
            for i in range(compare_n):
                diffs.append(abs(old_template[i]["rep_x"] - new_template[i]["rep_x"]))
            avg_diff = sum(diffs) / len(diffs) if diffs else 0.0

            # 테스트용 정규화
            position_change = min(avg_diff / 50.0, 1.0)

        score = 0.5 * lane_count_change + 0.5 * position_change
        return round(score, 3)

    # =========================================================
    # 7) phase 처리
    # =========================================================
    def process_phase(self):
        """
        현재 프레임 기준으로
        - bootstrap 완료 여부
        - freeze 유지 여부
        - freeze 종료 후 재평가 여부
        처리
        """
        elapsed = self.current_frame - self.phase_start_frame + 1

        # -----------------------------------------------------
        # A) 아직 template가 없는 상태 -> bootstrap
        # -----------------------------------------------------
        if not self.template_confirmed:
            if elapsed < self.bootstrap_frames:
                return {
                    "phase": "BOOTSTRAP_COLLECTING",
                    "elapsed": elapsed,
                    "template_confirmed": False,
                    "template": self.current_template,
                }

            cluster_info, main_lanes = self._extract_main_lanes()

            self.current_template = main_lanes
            self.template_confirmed = True
            self.template_freeze_until = self.current_frame + self.freeze_frames

            return {
                "phase": "BOOTSTRAP_CONFIRMED",
                "elapsed": elapsed,
                "template_confirmed": True,
                "template": self.current_template,
                "clusters": cluster_info,
                "freeze_until": self.template_freeze_until,
            }

        # -----------------------------------------------------
        # B) freeze 중
        # -----------------------------------------------------
        if self.current_frame <= self.template_freeze_until:
            return {
                "phase": "FREEZE_KEEP",
                "elapsed": elapsed,
                "template_confirmed": True,
                "template": self.current_template,
                "freeze_until": self.template_freeze_until,
            }

        # -----------------------------------------------------
        # C) freeze 끝 -> 다시 100프레임 수집 후 재평가
        # -----------------------------------------------------
        reeval_elapsed = self.current_frame - (self.template_freeze_until + 1) + 1

        if reeval_elapsed < self.bootstrap_frames:
            return {
                "phase": "REEVAL_COLLECTING",
                "elapsed": reeval_elapsed,
                "template_confirmed": True,
                "template": self.current_template,
                "freeze_until": self.template_freeze_until,
            }

        cluster_info, new_template = self._extract_main_lanes()
        change_score = self._template_change_score(self.current_template, new_template)

        updated = False
        if change_score >= self.template_change_ratio:
            self.current_template = new_template
            self.template_freeze_until = self.current_frame + self.freeze_frames
            updated = True

        return {
            "phase": "REEVAL_DONE",
            "elapsed": reeval_elapsed,
            "template_confirmed": True,
            "template": self.current_template,
            "new_template": new_template,
            "clusters": cluster_info,
            "change_score": change_score,
            "updated": updated,
            "freeze_until": self.template_freeze_until,
        }

    # =========================================================
    # 8) template에 차량 할당
    # =========================================================
    def assign_lane_to_vehicle(self, point):
        """
        확정된 공통 template 기준으로 차량에 lane_id 할당
        point = (x, y)
        """
        if not self.current_template:
            return None

        x, y = point
        best_lane = None
        best_dist = 1e9

        for lane in self.current_template:
            dist = abs(x - lane["rep_x"])
            if dist < best_dist:
                best_dist = dist
                best_lane = lane["lane_id"]

        return best_lane


# =============================================================
# 테스트용 출력 함수
# =============================================================
def print_phase_result(result):
    print(f"\n[PHASE] {result['phase']}")
    print(f"elapsed={result.get('elapsed')}")

    if "freeze_until" in result:
        print(f"freeze_until={result['freeze_until']}")

    if "change_score" in result:
        print(f"change_score={result['change_score']}")
        print(f"updated={result['updated']}")

    if "clusters" in result:
        print("clusters:")
        for c in result["clusters"]:
            print(
                f"  cluster_id={c['cluster_id']} | "
                f"count={c['count']} | "
                f"ratio={c['ratio']:.3f} | "
                f"rep_x={c['rep_x']:.2f} | "
                f"members={c['member_ids']}"
            )

    print("template:")
    for lane in result.get("template", []):
        print(
            f"  lane_id={lane['lane_id']} | "
            f"rep_x={lane['rep_x']:.2f} | "
            f"count={lane['count']} | "
            f"ratio={lane['ratio']:.3f}"
        )


# =============================================================
# 테스트 시나리오
# =============================================================
def build_fake_tracks_for_bootstrap():
    """
    20개 차량 궤적 생성
    - 왼쪽 차선 흐름 9개
    - 오른쪽 차선 흐름 8개
    - 잡음 흐름 3개

    => 40% 기준이면
       9/20=45%, 8/20=40%, 3/20=15%
       따라서 2개 차선만 template 채택
    """
    estimator = LaneTemplateEstimator(
        bootstrap_frames=100,
        freeze_frames=100,
        min_points_per_track=8,
        cluster_dist_thr=18.0,
        main_cluster_ratio=0.40,
        template_change_ratio=0.50
    )

    # 100프레임 동안 동일한 흐름 생성
    for frame in range(1, 101):
        tracks = []

        # Cluster A: 9개 궤적 (왼쪽 lane)
        for tid in range(1, 10):
            x = 150 + (tid % 3) * 3 + frame * 0.02
            y = 100 + frame * 2
            tracks.append({"id": tid, "point": (x, y)})

        # Cluster B: 8개 궤적 (오른쪽 lane)
        for tid in range(10, 18):
            x = 300 + (tid % 3) * 4 + frame * 0.02
            y = 110 + frame * 2
            tracks.append({"id": tid, "point": (x, y)})

        # Cluster C: 3개 궤적 (잡음)
        for tid in range(18, 21):
            x = 430 + (tid % 2) * 7 + frame * 0.01
            y = 90 + frame * 2
            tracks.append({"id": tid, "point": (x, y)})

        estimator.update_tracks(frame, tracks)

    return estimator


def run_test():
    print("\n===== lane_estimator_test2 START =====")

    estimator = build_fake_tracks_for_bootstrap()

    # bootstrap 완료 시점 처리
    result = estimator.process_phase()
    print_phase_result(result)

    # template 기준 차량 할당 예시
    test_points = [
        (152, 320),
        (304, 320),
        (435, 320),
    ]

    print("\n[ASSIGN TEST]")
    for p in test_points:
        lane = estimator.assign_lane_to_vehicle(p)
        print(f"point={p} -> assigned_lane={lane}")

    print("\n===== lane_estimator_test2 END =====")


if __name__ == "__main__":
    run_test()