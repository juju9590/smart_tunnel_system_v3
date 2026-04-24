# 서비스 고정 테스트(로직결과 아니라 형태/제어/기본 계약만 봄)
# 터미널 실행 : python -m unittest modules.tunnel.tests.test_tunnel_service_fixed -v
# 패키지 경로를 포함해서 모듈로 실행

import unittest

from backend_flask.modules.tunnel.service import TunnelV55Service


class TestTunnelServiceFixed(unittest.TestCase):
    def setUp(self):
        self.service = TunnelV55Service()

    def test_service_initial_status_shape(self):
        status = self.service.get_status()

        required_keys = [
            "state",
            "avg_speed",
            "vehicles",
            "dwell_times",
            "vehicle_count",
            "events",
            "accident",
            "accident_label",
            "lane_count_estimated",
            "frame_id",
            "source_name",
            "source_url",
            "running",
            "connected",
            "error",
            "cctv_count",
        ]

        for key in required_keys:
            self.assertIn(key, status)

    def test_service_initial_types(self):
        status = self.service.get_status()

        self.assertIsInstance(status["state"], str)
        self.assertIsInstance(status["avg_speed"], (int, float))
        self.assertIsInstance(status["vehicles"], list)
        self.assertIsInstance(status["dwell_times"], dict)
        self.assertIsInstance(status["vehicle_count"], int)
        self.assertIsInstance(status["events"], list)
        self.assertIsInstance(status["accident"], bool)
        self.assertIsInstance(status["accident_label"], str)
        self.assertIsInstance(status["lane_count_estimated"], int)
        self.assertIsInstance(status["frame_id"], int)
        self.assertIsInstance(status["source_name"], str)
        self.assertIsInstance(status["running"], bool)
        self.assertIsInstance(status["connected"], bool)

    def test_add_event_keeps_max_length(self):
        for i in range(20):
            self.service.add_event(f"event {i}")

        status = self.service.get_status()
        self.assertLessEqual(len(status["events"]), self.service.max_event_count)

    def test_get_jpeg_frame_returns_none_or_bytes(self):
        result = self.service.get_jpeg_frame()
        self.assertTrue(result is None or isinstance(result, (bytes, bytearray)))

    def test_start_sets_running_field_safely(self):
        self.service.start()
        status = self.service.get_status()

        self.assertIn("running", status)
        self.assertIsInstance(status["running"], bool)
        self.assertIn("state", status)
        self.assertIsInstance(status["state"], str)

    def test_stop_sets_running_false(self):
        self.service.stop()
        status = self.service.get_status()

        self.assertFalse(status["running"])
        self.assertIsInstance(status["state"], str)

    def test_pick_random_stream_returns_bool(self):
        result = self.service.pick_random_stream()
        self.assertIsInstance(result, bool)

    def test_switch_to_another_stream_returns_bool(self):
        result = self.service.switch_to_another_stream()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()