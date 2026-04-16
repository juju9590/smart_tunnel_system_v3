import os
import sys
import unittest
import inspect

# =========================
# 경로 설정
# tests 폴더에서 scr/pipeline_V5_3(상태로직) 안의
# traffic_state_V5_3.py를 import 하기 위한 설정
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "scr", "pipeline_V5_3(state_logic)")


sys.path.append(PIPELINE_DIR)

from traffic_state_V5_3 import TrafficState

def make_track(tid, x, y, w=40, h=20):
    """
    테스트용 bbox 생성 함수
    y2가 bottom point가 되도록 생성
    """
    x1 = x
    y1 = y - h
    x2 = x + w
    y2 = y
    return {"id": tid, "bbox": (x1, y1, x2, y2)}


class TestTrafficStateV53(unittest.TestCase):
    def setUp(self):
        self.state_model = TrafficState()

        # 테스트 빠르게 보기 위해 버퍼를 축소
        # 실제 코드에서는 300이지만,
        # 테스트에서는 10 정도로 줄여야 상태 전환 확인이 쉬움
        self.state_model.STATE_BUFFER_SIZE = 10
        self.state_model.STATE_HOLD_FRAMES = 3

        # 임계값 명시
        self.state_model.JAM_SPEED_THR = 2.0
        self.state_model.CONGESTION_SPEED_THR = 5.0

    def test_initial_state_is_normal(self):
        """
        시작 직후 tracks가 없어도 초기 상태는 NORMAL 이어야 함
        """
        result = self.state_model.update(frame_id=0, tracks=[], analysis={})
        print("RESULT TYPE:", type(result))
        print("RESULT VALUE:", result)
        self.assertEqual(result["state"], "NORMAL")
        self.assertTrue(result["debug"]["empty_frame"])

    def test_normal_state_with_fast_motion(self):
        """
        프레임당 이동량이 충분히 크면 NORMAL 유지
        """
        final_state = None

        for f in range(10):
            tracks = [
                make_track(1, 100, 100 + f * 6),
                make_track(2, 200, 120 + f * 7),
            ]
            result = self.state_model.update(frame_id=f, tracks=tracks, analysis={})
            final_state = result["state"]

        self.assertEqual(final_state, "NORMAL")
        self.assertGreaterEqual(result["debug"]["buffer_avg_speed"], 5.0)

    def test_congestion_state_with_medium_motion(self):
        """
        프레임당 이동량이 3~4 정도면 CONGESTION 예상
        """
        final_state = None

        for f in range(12):
            tracks = [
                make_track(1, 100, 100 + f * 3),
                make_track(2, 200, 120 + f * 4),
            ]
            result = self.state_model.update(frame_id=f, tracks=tracks, analysis={})
            final_state = result["state"]

        self.assertEqual(final_state, "CONGESTION")
        self.assertGreaterEqual(result["debug"]["buffer_avg_speed"], 2.0)
        self.assertLess(result["debug"]["buffer_avg_speed"], 5.0)

    def test_jam_state_with_slow_motion(self):
        """
        프레임당 이동량이 1 정도면 JAM 예상
        """
        final_state = None

        for f in range(12):
            tracks = [
                make_track(1, 100, 100 + f * 1),
                make_track(2, 200, 120 + f * 1),
            ]
            result = self.state_model.update(frame_id=f, tracks=tracks, analysis={})
            final_state = result["state"]

        self.assertEqual(final_state, "JAM")
        self.assertLess(result["debug"]["buffer_avg_speed"], 2.0)

    def test_hold_logic_blocks_immediate_change(self):
        """
        NORMAL 상태에서 갑자기 느려져도
        hold 때문에 바로 JAM으로 안 바뀌는지 확인
        """
        # 먼저 빠른 이동으로 NORMAL 상태 형성
        for f in range(8):
            tracks = [
                make_track(1, 100, 100 + f * 6),
                make_track(2, 200, 120 + f * 6),
            ]
            self.state_model.update(frame_id=f, tracks=tracks, analysis={})

        # 갑자기 매우 느린 이동
        result1 = self.state_model.update(
            frame_id=100,
            tracks=[
                make_track(1, 100, 101),
                make_track(2, 200, 121),
            ],
            analysis={}
        )

        result2 = self.state_model.update(
            frame_id=101,
            tracks=[
                make_track(1, 100, 102),
                make_track(2, 200, 122),
            ],
            analysis={}
        )

        # hold 중이라 바로 JAM이 아닐 수 있음
        self.assertIn(result1["state"], ["NORMAL", "CONGESTION"])
        self.assertIn(result2["state"], ["NORMAL", "CONGESTION", "JAM"])

    def test_empty_frame_keeps_buffer(self):
        """
        빈 프레임이어도 0을 넣지 않고
        기존 buffer_avg_speed가 유지되는지 확인
        """
        # 먼저 CONGESTION 근처 상태 형성
        for f in range(10):
            tracks = [
                make_track(1, 100, 100 + f * 3),
                make_track(2, 200, 120 + f * 4),
            ]
            self.state_model.update(frame_id=f, tracks=tracks, analysis={})

        before = self.state_model.get_debug_info()["buffer_avg_speed"]

        # 빈 프레임
        result = self.state_model.update(frame_id=999, tracks=[], analysis={})
        print("RESULT TYPE:", type(result))
        print("RESULT VALUE:", result)

        after = result["debug"]["buffer_avg_speed"]

        self.assertAlmostEqual(before, after, places=3)
        self.assertTrue(result["debug"]["empty_frame"])

        print("STATE:", result["state"])
        print("DEBUG:", result["debug"])


if __name__ == "__main__":
    unittest.main()