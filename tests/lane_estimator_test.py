# -*- coding: utf-8 -*-
"""
lane_estimator_test.py

목적
1) raw lane 안정화 테스트
2) stable lane 확정 조건:
   - 최근 최대 100프레임 기준 majority lane
   - 8프레임 연속 lane
   - 둘이 같을 때만 stable 확정
3) stable lane 확정 후 freeze 유지
4) freeze 종료 후 재배정 조건:
   - 최근 최대 100프레임 기준 새 majority lane
   - 12프레임 연속 lane
   - 둘이 같을 때만 재배정

실행
python lane_estimator_test.py
"""

from collections import defaultdict, deque, Counter


class LaneStabilizer:
    def __init__(
        self,
        history_size=50,         # 최근 raw lane 보관 길이
        min_samples=20,           # stable 판정 최소 표본 수
        confirm_streak=8,         # stable 확정용 연속 프레임 수
        reassign_streak=12,       # 재배정용 연속 프레임 수
        freeze_frames=100,        # stable 확정 후 freeze 길이
        majority_ratio=0.40       # 다수결 인정 최소 비율
    ):
        self.history_size = history_size
        self.min_samples = min_samples
        self.confirm_streak = confirm_streak
        self.reassign_streak = reassign_streak
        self.freeze_frames = freeze_frames
        self.majority_ratio = majority_ratio

        self.state = defaultdict(self._make_default_state)

    def _make_default_state(self):
        return {
            "raw_history": deque(maxlen=self.history_size),
            "stable_lane": None,
            "stable_since": None,
            "freeze_until": -1,

            "last_raw_lane": None,
            "same_raw_streak": 0,

            # 현재 streak lane 정보
            "current_streak_lane": None,
            "current_streak_len": 0,
        }

    def _get_majority_info(self, raw_history):
        """
        최근 history에서 가장 많이 나온 lane과 비율 계산
        """
        if not raw_history:
            return None, 0, 0.0

        counter = Counter(raw_history)
        majority_lane, majority_count = counter.most_common(1)[0]
        total = len(raw_history)
        majority_ratio = majority_count / total if total > 0 else 0.0

        return majority_lane, majority_count, majority_ratio

    def update(self, frame_idx, track_id, raw_lane):
        s = self.state[track_id]
        event = None

        # 1) raw history 저장
        s["raw_history"].append(raw_lane)

        # 2) 연속 streak 계산
        if raw_lane == s["last_raw_lane"]:
            s["same_raw_streak"] += 1
        else:
            s["same_raw_streak"] = 1
            s["last_raw_lane"] = raw_lane

        s["current_streak_lane"] = raw_lane
        s["current_streak_len"] = s["same_raw_streak"]

        # 3) majority 계산
        majority_lane, majority_count, majority_ratio = self._get_majority_info(s["raw_history"])
        total_samples = len(s["raw_history"])

        # =========================================================
        # A. stable lane이 아직 없을 때 -> stable 확정 시도
        # =========================================================
        if s["stable_lane"] is None:
            can_confirm = (
                total_samples >= self.min_samples and
                majority_lane is not None and
                majority_ratio >= self.majority_ratio and
                s["current_streak_len"] >= self.confirm_streak and
                majority_lane == s["current_streak_lane"]
            )

            if can_confirm:
                s["stable_lane"] = majority_lane
                s["stable_since"] = frame_idx
                s["freeze_until"] = frame_idx + self.freeze_frames
                event = f"[CONFIRM] stable_lane={majority_lane}"

        # =========================================================
        # B. stable lane이 이미 있을 때
        # =========================================================
        else:
            current_stable = s["stable_lane"]

            # B-1) freeze 중이면 유지
            if frame_idx <= s["freeze_until"]:
                pass

            # B-2) freeze 종료 후 재배정 시도
            else:
                new_candidate_lane = s["current_streak_lane"]

                can_reassign = (
                    total_samples >= self.min_samples and
                    majority_lane is not None and
                    majority_ratio >= self.majority_ratio and
                    new_candidate_lane != current_stable and
                    s["current_streak_len"] >= self.reassign_streak and
                    majority_lane == new_candidate_lane
                )

                if can_reassign:
                    old_lane = s["stable_lane"]
                    s["stable_lane"] = new_candidate_lane
                    s["stable_since"] = frame_idx
                    s["freeze_until"] = frame_idx + self.freeze_frames
                    event = f"[REASSIGN] {old_lane} -> {new_candidate_lane}"

        return {
            "frame": frame_idx,
            "track_id": track_id,
            "raw_lane": raw_lane,

            "samples": total_samples,
            "majority_lane": majority_lane,
            "majority_count": majority_count,
            "majority_ratio": round(majority_ratio, 3),

            "streak_lane": s["current_streak_lane"],
            "streak_len": s["current_streak_len"],

            "stable_lane": s["stable_lane"],
            "stable_since": s["stable_since"],
            "freeze_until": s["freeze_until"],
            "event": event
        }


def print_result(result):
    msg = (
        f"frame={result['frame']:03d} | "
        f"id={result['track_id']} | "
        f"raw={result['raw_lane']} | "
        f"samples={result['samples']:03d} | "
        f"maj={result['majority_lane']}({result['majority_ratio']:.2f}) | "
        f"streak={result['streak_lane']}:{result['streak_len']:02d} | "
        f"stable={result['stable_lane']} | "
        f"freeze_until={result['freeze_until']}"
    )

    if result["event"]:
        msg += f"   {result['event']}"

    print(msg)


def run_single_track_test():
    """
    시나리오
    1) 초반 흔들림
    2) lane 1이 전체적으로 가장 많고, 8연속도 lane 1 -> stable 확정 기대
    3) freeze 중 lane 2가 섞여도 stable 유지
    4) freeze 종료 후 lane 2가 충분히 우세 + 12연속이면 재배정 기대
    """

    stabilizer = LaneStabilizer(
        history_size=100,
        min_samples=20,         # 테스트 빨리 보려고 20
        confirm_streak=8,
        reassign_streak=12,
        freeze_frames=20,       # 테스트 빨리 보려고 20
        majority_ratio=0.40
    )

    track_id = 101

    raw_lane_sequence = [
        # 초기 흔들림
        1, 1, 2, 1, 1, 0, 1, 2, 1, 1,

        # lane 1 우세 + 8연속 성립
        1, 1, 1, 1, 1, 1, 1, 1,

        # freeze 중 흔들림
        2, 2, 1, 2, 1, 1, 2, 1, 2, 1,

        # freeze 끝난 뒤에도 아직 애매
        2, 1, 2, 2, 1, 2,

        # 이후 lane 2가 우세해지고 12연속 성립
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
    ]

    print("\n===== SINGLE TRACK TEST START =====\n")
    for frame_idx, raw_lane in enumerate(raw_lane_sequence, start=1):
        result = stabilizer.update(frame_idx, track_id, raw_lane)
        print_result(result)
    print("\n===== SINGLE TRACK TEST END =====\n")


def run_multi_track_test():
    """
    여러 차량 동시 테스트
    """

    stabilizer = LaneStabilizer(
        history_size=50,
        min_samples=20,         # 테스트용
        confirm_streak=8,
        reassign_streak=12,
        freeze_frames=15,
        majority_ratio=0.40
    )

    frame_data = {
        1:  {1: 0, 2: 2},
        2:  {1: 0, 2: 2},
        3:  {1: 1, 2: 2},
        4:  {1: 0, 2: 2},
        5:  {1: 0, 2: 2},
        6:  {1: 0, 2: 2},
        7:  {1: 0, 2: 2},
        8:  {1: 0, 2: 2},
        9:  {1: 0, 2: 2},
        10: {1: 0, 2: 2},
        11: {1: 1, 2: 2},
        12: {1: 1, 2: 2},
        13: {1: 1, 2: 2},
        14: {1: 1, 2: 2},
        15: {1: 1, 2: 2},
        16: {1: 1, 2: 2},
        17: {1: 1, 2: 1},
        18: {1: 1, 2: 1},
        19: {1: 1, 2: 1},
        20: {1: 1, 2: 1},
        21: {1: 1, 2: 1},
        22: {1: 1, 2: 1},
        23: {1: 1, 2: 1},
        24: {1: 1, 2: 1},
        25: {1: 1, 2: 1},
        26: {1: 1, 2: 1},
        27: {1: 1, 2: 1},
        28: {1: 1, 2: 1},
    }

    print("\n===== MULTI TRACK TEST START =====\n")
    for frame_idx in sorted(frame_data.keys()):
        for track_id, raw_lane in frame_data[frame_idx].items():
            result = stabilizer.update(frame_idx, track_id, raw_lane)
            print_result(result)
        print("-" * 110)
    print("\n===== MULTI TRACK TEST END =====\n")


if __name__ == "__main__":
    run_single_track_test()
    run_multi_track_test()