# ==========================================
# 파일명: lane_template_V5_5.py
# 설명:
# V5_5 차선(군집) 추정 모듈
#
# [개선 목적]
# 1) 기존 "초기 100프레임 1회 bootstrap" 방식 제거
# 2) 중앙영역 차량 2대 이상 + 연속 N프레임 유지 시 bootstrap 시작
# 3) bootstrap 실패 시 cooldown 후 재시도 가능
# 4) bootstrap 성공 시 최종 대표 집단(current_template) 확정
# 5) 현재 프레임 차량은 최종 대표 집단 중 가장 가까운 차선에 임시 할당
# 6) 동일 CCTV(터널)면 과거 저장된 lane memory를 먼저 불러오기
# 7) lane memory는 즉시 로드하되, 화면 표시만 50프레임 뒤에 시작
# 8) [추가] 관제사 요청 시점부터 50프레임을 따로 수집해 수동 재추정 가능
#
# [핵심 아이디어]
# - 실시간 CCTV에서는 초반에 차량이 적으면 차선 추정이 실패할 수 있음
# - 그래서 "시작 후 100프레임"이 아니라
#   "중앙영역 차량 수 조건이 만족된 뒤" bootstrap을 시작해야 안정적임
# - 또한 같은 터널은 구조가 거의 같으므로, 한 번 저장한 차선 template를
#   다음 진입 때 재사용하면 더 안정적으로 동작함
# - 다만 memory 차선을 영상 시작 직후 바로 그리면 어색할 수 있으므로
#   내부적으로는 즉시 로드하되, 화면 표시만 일정 프레임 뒤에 시작한다
# - 실험 단계에서는 관제사가 "지금 차량이 충분하다"고 판단한 시점에
#   버튼을 눌러 50프레임만 따로 모아 다시 차선을 추정할 수 있게 한다
# ==========================================

import os
import re
import json
import glob
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class LaneTemplateEstimator:
    def __init__(self, output_dir=None):
        # -----------------------------
        # 상태
        # -----------------------------
        self.current_template = []          # 최종 대표 집단 목록
        self.template_confirmed = False
        self.phase = "WAITING"              # WAITING / BOOTSTRAP / CLUSTER_VIEW / MEMORY_LOAD / MEMORY_WAIT / REESTIMATING

        # -----------------------------
        # 수집 메모리
        # -----------------------------
        self.track_models = {}              # tid -> fitted linear model
        self.track_fit_error = {}           # tid -> rmse
        self.track_stable_motion = {}       # tid -> 안정 이동 여부
        self.collected_track_ids = set()
        self.collected_track_points = {}    # tid -> [(x, y), ...]

        # -----------------------------
        # [추가] lane 재추정용 상태 변수
        # -----------------------------
        # 관제사가 "차선 재추정" 버튼을 누르면 True
        self.reestimate_requested = False

        # 실제 50프레임 수집 중인지 여부
        self.reestimate_collecting = False

        # 재추정 시작 프레임
        self.reestimate_start_frame = None

        # 재추정은 요청 시점부터 50프레임만 수집
        self.REESTIMATE_FRAME_WINDOW = 50

        # 현재 몇 프레임 수집했는지
        self.reestimate_frame_count = 0

        # 재추정용 track 포인트 버퍼
        # { track_id: [(x, y), (x, y), ...] }
        self.reestimate_track_points = defaultdict(list)

        # 프론트/로그용 상태 문자열
        # idle / reestimating / reestimated / confirmed
        self.reestimate_status = "idle"

        # -----------------------------
        # 디버그
        # -----------------------------
        self.last_debug = {
            "phase": "WAITING",
            "template_confirmed": False,
            "lane_count": 0,
            "template": [],
            "clusters_stage1": [],
            "clusters_stage2": [],
            "lane_map": {},
            "raw_lane_map": {},
            "memory_loaded": False,
            "memory_key": None,
            "memory_pending_display": False,
            "memory_delay_remaining": 0,
            "lane_reestimate_status": "idle",
            "lane_reestimate_frame_count": 0,
            "lane_reestimate_window": 50,
        }

        # -----------------------------
        # 파라미터
        # -----------------------------
        self.FIT_MIN_POINTS = 30

        # 안정 이동 여부 판단 기준
        self.MIN_TOTAL_MOTION = 35
        self.MAX_DY_JUMP = 40
        self.MAX_DX_JUMP = 60

        # 선형 피팅 품질 필터
        self.MAX_FIT_RMSE = 35.0

        # bootstrap 시작 조건
        self.CENTER_VEHICLE_COUNT_THR = 2
        self.BOOTSTRAP_READY_FRAMES = 20

        # bootstrap 수집/실패 제어
        self.BOOTSTRAP_MAX_COLLECT_FRAMES = 120
        self.MIN_BOOTSTRAP_TRACKS = 2
        self.BOOTSTRAP_COOLDOWN_FRAMES = 60

        # 현재 bootstrap 상태
        self.bootstrap_ready_count = 0
        self.bootstrap_collect_frames = 0
        self.bootstrap_cooldown = 0

        # 중앙영역 비율
        self.CENTER_X1_RATIO = 0.25
        self.CENTER_X2_RATIO = 0.75
        self.CENTER_Y1_RATIO = 0.20
        self.CENTER_Y2_RATIO = 0.85

        # 1차 군집: 차량 궤적끼리
        self.LINEAR_CLUSTER_THR_STAGE1 = 0.12

        # 2차 군집: 1차 대표선끼리
        self.LINEAR_CLUSTER_THR_STAGE2 = 0.22

        # 그래프 저장 경로
        self.output_dir = output_dir or os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)

        # -----------------------------
        # lane memory 저장 경로
        # -----------------------------
        self.memory_dir = os.path.join(self.output_dir, "lane_memory")
        os.makedirs(self.memory_dir, exist_ok=True)

        # -----------------------------
        # 현재 CCTV 이름 관련 상태
        # -----------------------------
        self.current_cctv_name = None
        self.current_cctv_key = None

        # 같은 CCTV에서 memory를 매 프레임 재조회하지 않기 위한 플래그
        self.memory_checked = False
        self.memory_loaded = False

        # -----------------------------
        # memory 표시 지연 관련 상태
        # -----------------------------
        self.MEMORY_DISPLAY_DELAY_FRAMES = 50
        self.memory_loaded_frame = None
        self.memory_pending_display = False

    # =========================================================
    # 0) bootstrap 관련 상태 초기화
    # =========================================================
    def _reset_bootstrap_memory(self):
        """
        bootstrap 재시도 전 메모리 초기화
        - template_confirmed는 건드리지 않음
        - bootstrap 수집용 메모리만 비움
        """
        self.track_models = {}
        self.track_fit_error = {}
        self.track_stable_motion = {}
        self.collected_track_ids = set()
        self.collected_track_points = {}
        self.bootstrap_collect_frames = 0

    # =========================================================
    # 0-1) 전체 상태 초기화 (새 CCTV 진입 시 사용)
    # =========================================================
    def _reset_for_new_cctv(self):
        """
        새로운 CCTV 이름이 들어온 경우
        lane estimator 전체 상태를 새로 시작하기 위한 초기화
        """
        self.current_template = []
        self.template_confirmed = False
        self.phase = "WAITING"

        self.track_models = {}
        self.track_fit_error = {}
        self.track_stable_motion = {}
        self.collected_track_ids = set()
        self.collected_track_points = {}

        self.bootstrap_ready_count = 0
        self.bootstrap_collect_frames = 0
        self.bootstrap_cooldown = 0

        self.memory_checked = False
        self.memory_loaded = False

        self.memory_loaded_frame = None
        self.memory_pending_display = False

        # 수동 재추정 상태도 함께 초기화
        self.reestimate_requested = False
        self.reestimate_collecting = False
        self.reestimate_start_frame = None
        self.reestimate_frame_count = 0
        self.reestimate_track_points.clear()
        self.reestimate_status = "idle"

    # =========================================================
    # 0-2) 그래프 파일 정리
    # =========================================================
    def _cleanup_debug_graphs(self):
        """
        새 bootstrap 결과를 저장하기 전에
        이전 그래프 png를 자동 삭제
        """
        patterns = [
            "graph_v5_5_*.png",
            "graph_stage1_v5_5_*.png",
            "graph_stage2_v5_5_*.png",
        ]

        for pattern in patterns:
            for path in glob.glob(os.path.join(self.output_dir, pattern)):
                try:
                    os.remove(path)
                except Exception:
                    pass

    # =========================================================
    # 0-3) CCTV 이름 정규화
    # =========================================================
    def _normalize_cctv_name(self, cctv_name):
        """
        실시간 CCTV 이름을 파일 저장/검색용 키로 변환

        예:
        "[수원광명선] 광명 구봉산터널"
        -> "수원광명선_광명_구봉산터널"
        """
        if not cctv_name:
            return None

        name = str(cctv_name).strip()
        name = name.replace("[", " ")
        name = name.replace("]", " ")
        name = name.replace("(", " ")
        name = name.replace(")", " ")
        name = re.sub(r'[\\/:*?"<>|]+', " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        name = name.replace(" ", "_")

        return name if name else None

    # =========================================================
    # 0-4) memory 파일 경로 생성
    # =========================================================
    def _get_memory_path(self, cctv_name):
        memory_key = self._normalize_cctv_name(cctv_name)
        if not memory_key:
            return None, None

        path = os.path.join(self.memory_dir, f"{memory_key}.json")
        return memory_key, path

    # =========================================================
    # 0-5) lane memory 저장
    # =========================================================
    def save_lane_memory(self, cctv_name=None):
        """
        현재 확정된 current_template를 json으로 저장
        """
        target_name = cctv_name or self.current_cctv_name
        if not target_name:
            print("⚠️ lane memory 저장 실패: CCTV 이름 없음")
            return None

        if not self.current_template:
            print("⚠️ lane memory 저장 실패: current_template 비어 있음")
            return None

        memory_key, save_path = self._get_memory_path(target_name)
        if not save_path:
            print("⚠️ lane memory 저장 실패: memory path 생성 실패")
            return None

        payload = {
            "cctv_name": target_name,
            "memory_key": memory_key,
            "lane_count": len(self.current_template),
            "centerlines": self.current_template,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"💾 lane memory 저장 완료: {save_path}")
        return save_path

    # =========================================================
    # 0-6) lane memory 로드
    # =========================================================
    def load_lane_memory(self, cctv_name=None, frame_id=None):
        """
        동일한 CCTV 이름의 저장된 차선 template가 있으면 불러오기

        [중요]
        - 내부적으로는 즉시 current_template에 로드
        - 하지만 화면 표시는 MEMORY_DISPLAY_DELAY_FRAMES 뒤에 시작
        """
        target_name = cctv_name or self.current_cctv_name
        if not target_name:
            return False

        memory_key, load_path = self._get_memory_path(target_name)
        if not load_path or not os.path.exists(load_path):
            return False

        try:
            with open(load_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            centerlines = payload.get("centerlines", [])
            if not isinstance(centerlines, list) or len(centerlines) == 0:
                return False

            self.current_template = centerlines
            self.template_confirmed = True

            self.phase = "MEMORY_WAIT"
            self.memory_loaded = True
            self.memory_checked = True
            self.memory_loaded_frame = frame_id
            self.memory_pending_display = True

            print(f"📂 저장된 lane memory 로드: {load_path}")
            return True

        except Exception as e:
            print(f"❌ lane memory 로드 실패: {e}")
            return False

    # =========================================================
    # 0-7) memory 표시 지연 남은 프레임 계산
    # =========================================================
    def _get_memory_delay_remaining(self, frame_id):
        """
        memory를 로드한 뒤 화면에 표시하기까지 남은 프레임 수 계산
        """
        if self.memory_loaded_frame is None:
            return self.MEMORY_DISPLAY_DELAY_FRAMES

        elapsed = max(0, frame_id - self.memory_loaded_frame)
        remaining = max(0, self.MEMORY_DISPLAY_DELAY_FRAMES - elapsed)
        return remaining

    # =========================================================
    # 0-8) 수동 보정용 lane 제거 후 저장
    # =========================================================
    def remove_lane_and_save(self, lane_id_to_remove, cctv_name=None):
        """
        사람이 "lane 1은 차선 아님" 같은 수동 보정을 할 때 쓰는 함수
        """
        if not self.current_template:
            print("⚠️ lane 제거 실패: current_template 비어 있음")
            return False

        filtered = [lane for lane in self.current_template if lane.get("lane_id") != lane_id_to_remove]

        if len(filtered) == len(self.current_template):
            print(f"⚠️ lane 제거 실패: lane_id={lane_id_to_remove} 없음")
            return False

        reindexed = []
        for new_idx, lane in enumerate(filtered):
            lane_copy = dict(lane)
            lane_copy["lane_id"] = new_idx
            reindexed.append(lane_copy)

        self.current_template = reindexed
        self.template_confirmed = len(self.current_template) > 0
        self.phase = "CLUSTER_VIEW" if self.template_confirmed else "WAITING"
        self.memory_pending_display = False
        self.memory_loaded = True

        self.save_lane_memory(cctv_name=cctv_name)
        print(f"🛠️ lane {lane_id_to_remove} 제거 후 memory 재저장 완료")
        return True

    # =========================================================
    # 1) 중앙영역 차량 수 계산
    # =========================================================
    def _count_center_tracks(self, track_points, frame_width, frame_height):
        """
        현재 프레임의 대표점(track_points)이 중앙영역에 몇 대 있는지 계산
        """
        cx1 = int(frame_width * self.CENTER_X1_RATIO)
        cx2 = int(frame_width * self.CENTER_X2_RATIO)
        cy1 = int(frame_height * self.CENTER_Y1_RATIO)
        cy2 = int(frame_height * self.CENTER_Y2_RATIO)

        count = 0

        for _, pt in track_points.items():
            x, y = pt
            if cx1 <= x <= cx2 and cy1 <= y <= cy2:
                count += 1

        return count

    # =========================================================
    # 2) 안정 이동 차량 판정
    # =========================================================
    def _is_stable_moving_track(self, pts):
        """
        최근 궤적이 충분히 길고, 급점프가 적고, 실제로 이동량이 있는지 검사
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
    # 3) 차량 궤적 선형 피팅
    # =========================================================
    def _fit_trajectory_model(self, pts):
        """
        최근 궤적을 선형으로 피팅
        x = a*y + b
        """
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
        """
        주어진 y에서 선형 모델의 x 위치 계산
        """
        if model is None:
            return None

        a, b = model["coef"]
        return a * y + b

    # =========================================================
    # 4) 모델 거리 계산
    # =========================================================
    def _coef_vector(self, model, frame_height):
        """
        선형 계수 벡터를 정규화된 형태로 변환
        """
        a, b = model["coef"]
        return np.array([a, b / max(frame_height, 1)], dtype=np.float32)

    def _model_distance(self, model1, model2, lane_y1, lane_y2, frame_height):
        """
        두 선형 궤적이 얼마나 비슷한지 계산
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
    # 5) 대표 모델 계산
    # =========================================================
    def _aggregate_models(self, models):
        """
        여러 선형 모델을 하나의 대표 모델로 통합
        중앙값(median) 사용
        """
        if not models:
            return None

        arr = np.array([m["coef"] for m in models], dtype=np.float32)
        coef = np.median(arr, axis=0).tolist()

        return {
            "type": "linear",
            "coef": coef
        }

    # =========================================================
    # 6) bootstrap 동안 안정 차량 궤적 수집
    # =========================================================
    def _collect_stable_models(self, track_history):
        """
        현재까지의 track_history에서 안정적으로 움직이는 차량만 선형 모델로 수집
        """
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
    # 6-1) [추가] 재추정 50프레임 버퍼를 기존 군집 입력 형태로 변환
    # =========================================================
    def _build_reestimate_template(self, lane_y1, lane_y2, frame_height):
        """
        관제사 요청 후 50프레임 동안 모은 reestimate_track_points를 사용해서
        기존 bootstrap과 동일한 군집화 흐름으로 current_template를 다시 만든다.

        핵심:
        - 완전히 새로운 로직을 만들지 않고
        - 기존 _cluster_track_models_stage1 / stage2 를 재사용한다
        - 대신 50프레임 버퍼를 임시로 self.track_models / self.collected_track_ids
          형식에 맞게 구성해서 넣는다
        """
        # -----------------------------
        # 1) 기존 bootstrap 메모리를 백업
        #    (현재 확정 template는 유지)
        # -----------------------------
        backup_track_models = self.track_models
        backup_track_fit_error = self.track_fit_error
        backup_track_stable_motion = self.track_stable_motion
        backup_collected_track_ids = self.collected_track_ids
        backup_collected_track_points = self.collected_track_points

        # 임시 재구성용 메모리
        temp_track_models = {}
        temp_track_fit_error = {}
        temp_track_stable_motion = {}
        temp_collected_track_ids = set()
        temp_collected_track_points = {}

        # -----------------------------
        # 2) 50프레임 버퍼에서 안정 track만 선형 모델 생성
        # -----------------------------
        for tid, pts in self.reestimate_track_points.items():
            pts = list(pts)

            # 재추정은 "버튼 누른 이후 50프레임"이라 포인트 수가 많지 않을 수 있으므로
            # 너무 엄격하면 아무 것도 안 잡힐 수 있다.
            # 그래서 여기서는 최소 10점 이상만 있으면 한 번 시도하고,
            # 기존 안정성 조건도 함께 참고한다.
            if len(pts) < 10:
                continue

            # 최근 30점보다 적으면 전부 사용, 많으면 최근 30점 사용
            fit_pts = pts[-self.FIT_MIN_POINTS:] if len(pts) >= self.FIT_MIN_POINTS else pts

            stable = True
            if len(pts) >= self.FIT_MIN_POINTS:
                stable = self._is_stable_moving_track(pts)

            if not stable:
                continue

            model, rmse = self._fit_trajectory_model(fit_pts if len(fit_pts) >= self.FIT_MIN_POINTS else (fit_pts + fit_pts[-1:] * max(0, self.FIT_MIN_POINTS - len(fit_pts))))
            if model is None:
                continue

            if rmse > self.MAX_FIT_RMSE:
                continue

            temp_track_models[tid] = model
            temp_track_fit_error[tid] = rmse
            temp_track_stable_motion[tid] = True
            temp_collected_track_ids.add(tid)
            temp_collected_track_points[tid] = pts

        # 유효 track 수 부족
        if len(temp_collected_track_ids) < self.MIN_BOOTSTRAP_TRACKS:
            self.track_models = backup_track_models
            self.track_fit_error = backup_track_fit_error
            self.track_stable_motion = backup_track_stable_motion
            self.collected_track_ids = backup_collected_track_ids
            self.collected_track_points = backup_collected_track_points
            print("⚠️ 재추정 실패: 유효 track 수 부족")
            return False, [], []

        # -----------------------------
        # 3) 기존 군집 로직 재사용
        # -----------------------------
        self.track_models = temp_track_models
        self.track_fit_error = temp_track_fit_error
        self.track_stable_motion = temp_track_stable_motion
        self.collected_track_ids = temp_collected_track_ids
        self.collected_track_points = temp_collected_track_points

        clusters_stage1 = self._cluster_track_models_stage1(
            lane_y1, lane_y2, frame_height
        )
        cluster_info_stage1 = self._extract_cluster_info_stage1(
            clusters_stage1, lane_y1, lane_y2
        )

        clusters_stage2 = self._cluster_representatives_stage2(
            cluster_info_stage1, lane_y1, lane_y2, frame_height
        )
        final_cluster_info = self._extract_cluster_info_stage2(
            clusters_stage2, lane_y1, lane_y2
        )

        success = self._is_valid_bootstrap_result(final_cluster_info)

        if success:
            new_template = []
            for idx, c in enumerate(final_cluster_info):
                new_template.append({
                    "lane_id": idx,
                    "cluster_id": c["cluster_id"],
                    "rep_model": c["rep_model"],
                    "x_mid": c["x_mid"],
                    "count": c["count"],
                    "ratio": c["ratio"],
                    "member_ids": c["member_ids"],
                    "source_cluster_ids": c["source_cluster_ids"],
                })

            self.current_template = new_template
            self.template_confirmed = True
            self.phase = "CLUSTER_VIEW"
            self.memory_loaded = False
            self.memory_pending_display = False
            self.memory_loaded_frame = None

            # 재추정 성공 시 같은 CCTV 이름으로 자동 저장
            # self.save_lane_memory(self.current_cctv_name) # 수동저장 추가로 .. 주석처리

            print(f"✅ 차선 재추정 완료: {len(self.current_template)}개 차선")
        else:
            print("⚠️ 차선 재추정 실패: 군집 결과가 유효하지 않음")

        # -----------------------------
        # 4) 원래 bootstrap 메모리 복구
        #    (현재 template는 새 걸로 유지)
        # -----------------------------
        self.track_models = backup_track_models
        self.track_fit_error = backup_track_fit_error
        self.track_stable_motion = backup_track_stable_motion
        self.collected_track_ids = backup_collected_track_ids
        self.collected_track_points = backup_collected_track_points

        return success, cluster_info_stage1, final_cluster_info

    # =========================================================
    # 7) 1차 군집화
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
    # 8) 1차 군집 정보 정리
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
    # 9) 2차 군집화
    # =========================================================
    def _cluster_representatives_stage2(self, cluster_info_stage1, lane_y1, lane_y2, frame_height):
        """
        1차 군집 대표선들끼리 다시 거리 계산해서 비슷한 흐름을 한 번 더 합침
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
    # 10) 2차 군집 결과 정리
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
    # 11) bootstrap 결과 유효성 검사
    # =========================================================
    def _is_valid_bootstrap_result(self, final_cluster_info):
        """
        bootstrap 결과가 실제로 쓸 만한지 검사
        """
        if len(final_cluster_info) < 1:
            return False

        if len(self.collected_track_ids) < self.MIN_BOOTSTRAP_TRACKS:
            return False

        return True

    # =========================================================
    # 12) memory 표시 대기 중인지 검사
    # =========================================================
    def _is_memory_display_ready(self, frame_id):
        """
        memory가 로드된 뒤 50프레임이 지나야 화면 표시 허용
        """
        if not self.memory_pending_display:
            return True

        remaining = self._get_memory_delay_remaining(frame_id)
        if remaining <= 0:
            self.memory_pending_display = False
            self.phase = "MEMORY_LOAD"
            return True

        self.phase = "MEMORY_WAIT"
        return False

    # =========================================================
    # 13) 현재 차량을 가장 가까운 최종 대표 집단에 임시 할당
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
    # 14) 그래프 저장
    # =========================================================
    def save_trajectory_plot(self, lane_y1, lane_y2, filename=None):
        """
        최종 대표 집단 그래프 저장
        """
        if not self.collected_track_points:
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"graph_v5_5_{timestamp}.png"

        save_path = os.path.join(self.output_dir, filename)

        plt.figure(figsize=(8, 6))

        for tid, pts in self.collected_track_points.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, linewidth=1, alpha=0.6)
            plt.text(xs[-1], ys[-1], str(tid), fontsize=8)

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
        """
        if not self.collected_track_points:
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"graph_stage1_v5_5_{timestamp}.png"

        save_path = os.path.join(self.output_dir, filename)

        plt.figure(figsize=(8, 6))

        for tid, pts in self.collected_track_points.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, linewidth=1, alpha=0.6)
            plt.text(xs[-1], ys[-1], str(tid), fontsize=8)

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
        """
        if not self.collected_track_points:
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"graph_stage2_v5_5_{timestamp}.png"

        save_path = os.path.join(self.output_dir, filename)

        plt.figure(figsize=(8, 6))

        for tid, pts in self.collected_track_points.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, linewidth=1, alpha=0.6)
            plt.text(xs[-1], ys[-1], str(tid), fontsize=8)

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
    # 14-1) [추가] 관제사 수동 차선 재추정 요청
    # =========================================================
    def request_reestimate(self, frame_id=None):
        """
        관제사가 버튼을 눌렀을 때 호출한다.

        동작:
        1) 요청 시점 기록
        2) 그 이후 50프레임 동안만 reestimate_track_points에 따로 수집
        3) 50프레임이 차면 기존 군집 로직을 재사용해 current_template 교체

        주의:
        - 실험 단계라 별도 조건 검사 없이 바로 시작한다
        - 현재 차선(template)은 재추정이 끝날 때까지 그대로 유지한다
        """
        self.reestimate_requested = True
        self.reestimate_collecting = True
        self.reestimate_start_frame = frame_id
        self.reestimate_frame_count = 0
        self.reestimate_track_points.clear()
        self.reestimate_status = "reestimating"
        self.phase = "REESTIMATING"

        print(f"🔄 차선 재추정 요청 접수: start_frame={frame_id}")

    # =========================================================
    # 14-2) [추가] track_history에서 재추정용 포인트 수집
    # =========================================================
    def _collect_reestimate_points(self, track_history):
        """
        현재 프레임의 track_history에서 마지막 점을 추출해
        reestimate용 버퍼에 누적한다.

        전제:
        - 현재 프로젝트의 track_history는 tid -> [(x, y), ...] 구조로 사용 중
        - 혹시 dict 형태가 들어와도 최소 대응 가능하도록 같이 처리
        """
        if not track_history:
            return

        for track_id, history in track_history.items():
            if not history:
                continue

            last_pt = history[-1]
            x, y = None, None

            # case 1) dict 형태
            if isinstance(last_pt, dict):
                if "x" in last_pt and "y" in last_pt:
                    x, y = last_pt["x"], last_pt["y"]
                elif "cx" in last_pt and "cy" in last_pt:
                    x, y = last_pt["cx"], last_pt["cy"]

            # case 2) tuple/list 형태
            elif isinstance(last_pt, (tuple, list)) and len(last_pt) >= 2:
                x, y = last_pt[0], last_pt[1]

            if x is None or y is None:
                continue

            self.reestimate_track_points[track_id].append((float(x), float(y)))

    # =========================================================
    # 14-3) [추가] 재추정 상태를 결과에 실어주기
    # =========================================================
    def _attach_reestimate_status_to_result(self, result):
        result["lane_reestimate_status"] = self.reestimate_status
        result["lane_reestimate_frame_count"] = self.reestimate_frame_count
        result["lane_reestimate_window"] = self.REESTIMATE_FRAME_WINDOW
        return result

    # =========================================================
    # 15) 외부 호출 메인 함수
    # =========================================================
    def update(self, frame_id, analysis):
        track_history = analysis["track_history"]
        track_points = analysis["track_points"]
        frame_height = analysis["frame_height"]

        # 현재 CCTV 이름
        incoming_cctv_name = analysis.get("cctv_name")

        # CCTV가 바뀌면 lane estimator 상태도 새로 시작
        if incoming_cctv_name and incoming_cctv_name != self.current_cctv_name:
            self.current_cctv_name = incoming_cctv_name
            self.current_cctv_key = self._normalize_cctv_name(incoming_cctv_name)
            self._reset_for_new_cctv()

        frame_width = analysis.get("frame_width", 1280)

        # 차선 추정은 ROI와 별도로 화면 전체 흐름을 보기 위해 더 넓은 세로 범위 사용
        lane_y1 = int(frame_height * 0.20)
        lane_y2 = int(frame_height * 0.95)

        clusters_stage1_debug = []
        clusters_stage2_debug = []

        # -----------------------------------------------------
        # [추가] R) 수동 재추정 50프레임 수집 로직
        # -----------------------------------------------------
        # 요청이 들어온 뒤에는 기존 자동 bootstrap과 별개로
        # 50프레임 동안 현재 track_history를 따로 모은다.
        if self.reestimate_collecting:
            self.phase = "REESTIMATING"
            self.reestimate_status = "reestimating"

            self._collect_reestimate_points(track_history)
            self.reestimate_frame_count += 1

            # 50프레임이 차면 즉시 새 template 생성 시도
            if self.reestimate_frame_count >= self.REESTIMATE_FRAME_WINDOW:
                ok, clusters_stage1_debug, clusters_stage2_debug = self._build_reestimate_template(
                    lane_y1=lane_y1,
                    lane_y2=lane_y2,
                    frame_height=frame_height
                )

                self.reestimate_collecting = False
                self.reestimate_requested = False
                self.reestimate_start_frame = None
                self.reestimate_track_points.clear()
                self.reestimate_frame_count = 0

                if ok:
                    self.reestimate_status = "reestimated"
                    self.phase = "CLUSTER_VIEW"
                else:
                    # 실패해도 기존 template는 그대로 유지
                    self.reestimate_status = "confirmed" if self.template_confirmed else "idle"
                    self.phase = "CLUSTER_VIEW" if self.template_confirmed else "WAITING"

        # -----------------------------------------------------
        # A) 저장된 memory가 있으면 bootstrap 전에 먼저 로드
        # 같은 CCTV에서는 한 번만 체크
        # 재추정 중에는 memory 자동 로드 로직을 건드리지 않는다.
        # -----------------------------------------------------
        if (not self.reestimate_collecting) and (not self.memory_checked) and (not self.template_confirmed) and self.current_cctv_name:
            self.memory_checked = True
            loaded = self.load_lane_memory(self.current_cctv_name, frame_id=frame_id)
            if loaded:
                self.phase = "MEMORY_WAIT"

        # -----------------------------------------------------
        # B) 이미 template가 확정된 경우
        # memory load 포함
        # 이후에는 bootstrap 안 하고 바로 차선 할당만 수행
        # 단, 재추정 중이면 bootstrap 분기는 건너뜀
        # -----------------------------------------------------
        if self.template_confirmed and (not self.reestimate_collecting):
            if self.memory_loaded:
                if self._is_memory_display_ready(frame_id):
                    self.phase = "MEMORY_LOAD"
                else:
                    self.phase = "MEMORY_WAIT"
            else:
                # 직전 재추정 완료 상태라면 한 번은 유지
                if self.reestimate_status != "reestimated":
                    self.reestimate_status = "confirmed"
                self.phase = "CLUSTER_VIEW"

        elif not self.template_confirmed and (not self.reestimate_collecting):
            # -------------------------------------------------
            # C) bootstrap cooldown 감소
            # -------------------------------------------------
            if self.bootstrap_cooldown > 0:
                self.bootstrap_cooldown -= 1

            # -------------------------------------------------
            # D) 중앙영역 차량 수 계산
            # -------------------------------------------------
            center_count = self._count_center_tracks(
                track_points,
                frame_width,
                frame_height
            )

            if self.bootstrap_cooldown == 0 and center_count >= self.CENTER_VEHICLE_COUNT_THR:
                self.bootstrap_ready_count += 1
            else:
                self.bootstrap_ready_count = 0

            # -------------------------------------------------
            # E) 아직 bootstrap 시작 전 대기 상태
            # -------------------------------------------------
            if self.bootstrap_collect_frames == 0 and self.bootstrap_ready_count < self.BOOTSTRAP_READY_FRAMES:
                self.phase = "WAITING"
                clusters_stage1_debug = []
                clusters_stage2_debug = []

            else:
                # ---------------------------------------------
                # F) bootstrap 시작 시점
                # ---------------------------------------------
                if self.bootstrap_collect_frames == 0 and self.bootstrap_ready_count >= self.BOOTSTRAP_READY_FRAMES:
                    self._reset_bootstrap_memory()
                    self.phase = "BOOTSTRAP"

                # ---------------------------------------------
                # G) bootstrap 수집 진행
                # ---------------------------------------------
                self.phase = "BOOTSTRAP"
                self._collect_stable_models(track_history)
                self.bootstrap_collect_frames += 1

                # ---------------------------------------------
                # H) 현재까지 수집된 모델로 군집화 시도
                # ---------------------------------------------
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

                clusters_stage1_debug = cluster_info_stage1
                clusters_stage2_debug = final_cluster_info

                # ---------------------------------------------
                # I) bootstrap 성공 조건
                # ---------------------------------------------
                success = self._is_valid_bootstrap_result(final_cluster_info)

                if success:
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
                    self.memory_loaded = False
                    self.memory_pending_display = False
                    self.memory_loaded_frame = None
                    self.reestimate_status = "confirmed"

                    # 새 결과 저장 전 이전 그래프 삭제
                    self._cleanup_debug_graphs()

                    # 1차/2차 그래프 저장
                    self.save_trajectory_plot_stage1(
                        lane_y1=lane_y1,
                        lane_y2=lane_y2,
                        cluster_info_stage1=cluster_info_stage1
                    )
                    self.save_trajectory_plot_stage2(
                        lane_y1=lane_y1,
                        lane_y2=lane_y2
                    )

                    # 같은 CCTV 이름으로 lane memory 자동 저장
                    self.save_lane_memory(self.current_cctv_name)

                # ---------------------------------------------
                # J) bootstrap 실패 처리
                # ---------------------------------------------
                elif self.bootstrap_collect_frames >= self.BOOTSTRAP_MAX_COLLECT_FRAMES:
                    self.phase = "WAITING"
                    self.bootstrap_cooldown = self.BOOTSTRAP_COOLDOWN_FRAMES
                    self.bootstrap_ready_count = 0
                    self._reset_bootstrap_memory()
                    clusters_stage1_debug = []
                    clusters_stage2_debug = []

        # -----------------------------------------------------
        # K) memory 표시 대기 중이면 화면에 차선을 숨긴다
        # 내부적으로는 current_template를 갖고 있지만,
        # 아직 50프레임이 지나지 않았으므로 프론트/오버레이에는 안 보이게 함
        # -----------------------------------------------------
        hide_memory_display = self.memory_loaded and self.memory_pending_display
        memory_delay_remaining = self._get_memory_delay_remaining(frame_id) if hide_memory_display else 0

        # -----------------------------------------------------
        # L) 현재 최종 대표 집단 기준 임시 할당
        # 단, memory 표시 대기 중이면 lane_map도 비워서
        # 화면에 차선/차선번호가 안 보이게 만든다
        # -----------------------------------------------------
        lane_map = {}
        raw_lane_map = {}

        if not hide_memory_display:
            for tid, pt in track_points.items():
                lane_id = self._assign_lane(pt, self.current_template)
                lane_map[tid] = lane_id
                raw_lane_map[tid] = lane_id

            visible_centerlines = self.current_template
            visible_lane_count = len(self.current_template)
        else:
            visible_centerlines = []
            visible_lane_count = 0

        result = {
            "lane_map": lane_map,
            "raw_lane_map": raw_lane_map,
            "lane_count": visible_lane_count,
            "centerlines": visible_centerlines,
            "lane_debug": {
                tid: {
                    "raw_lane": raw_lane_map.get(tid),
                    "stable_lane": lane_map.get(tid),
                } for tid in lane_map
            },
            "template_phase": self.phase,
            "template_confirmed": (self.template_confirmed and not hide_memory_display),
            "clusters_stage1": clusters_stage1_debug,
            "clusters_stage2": clusters_stage2_debug,
            "clusters": clusters_stage2_debug,
            "memory_loaded": self.memory_loaded,
            "memory_key": self.current_cctv_key,
            "memory_pending_display": self.memory_pending_display,
            "memory_delay_remaining": memory_delay_remaining,
        }

        # 재추정 상태값 추가
        result = self._attach_reestimate_status_to_result(result)

        self.last_debug = {
            "phase": self.phase,
            "template_confirmed": (self.template_confirmed and not hide_memory_display),
            "lane_count": visible_lane_count,
            "template": visible_centerlines,
            "clusters_stage1": clusters_stage1_debug,
            "clusters_stage2": clusters_stage2_debug,
            "lane_map": lane_map,
            "raw_lane_map": raw_lane_map,
            "memory_loaded": self.memory_loaded,
            "memory_key": self.current_cctv_key,
            "memory_pending_display": self.memory_pending_display,
            "memory_delay_remaining": memory_delay_remaining,
            "lane_reestimate_status": self.reestimate_status,
            "lane_reestimate_frame_count": self.reestimate_frame_count,
            "lane_reestimate_window": self.REESTIMATE_FRAME_WINDOW,
        }

        return result

    def get_debug_info(self):
        return self.last_debug