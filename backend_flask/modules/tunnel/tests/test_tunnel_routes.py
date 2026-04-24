# 라우트/API 고정 테스트
# 터미널 실행 :  python -m unittest modules.tunnel.tests.test_tunnel_routes -v
# 패키지 경로를 포함해서 모듈로 실행

import unittest
from flask import Flask

from modules.tunnel.routes import tunnel_bp

import modules.tunnel.routes
print("ROUTES FILE =", modules.tunnel.routes.__file__)


class TunnelRoutesTestCase(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(tunnel_bp, url_prefix="/api/tunnel")
        self.client = app.test_client()

    def test_health_route(self):
        res = self.client.get("/api/tunnel/health")
        self.assertEqual(res.status_code, 200)

        data = res.get_json()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["module"], "tunnel")

    def test_status_route_returns_json(self):
        res = self.client.get("/api/tunnel/status")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.is_json)

        data = res.get_json()
        self.assertIsInstance(data, dict)

    def test_status_route_has_required_keys(self):
        res = self.client.get("/api/tunnel/status")
        data = res.get_json()

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
            self.assertIn(key, data)

    def test_start_route(self):
        res = self.client.post("/api/tunnel/start")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.is_json)

        data = res.get_json()
        self.assertIn("message", data)
        self.assertIn("running", data)
        self.assertIsInstance(data["running"], bool)

    def test_stop_route(self):
        res = self.client.post("/api/tunnel/stop")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.is_json)

        data = res.get_json()
        self.assertIn("message", data)
        self.assertIn("running", data)
        self.assertIsInstance(data["running"], bool)

    def test_video_feed_route(self):
        res = self.client.get("/api/tunnel/video_feed")
        self.assertEqual(res.status_code, 200)
        self.assertIn("multipart/x-mixed-replace", res.content_type)


if __name__ == "__main__":
    unittest.main()