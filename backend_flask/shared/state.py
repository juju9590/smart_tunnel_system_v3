import os

# 공유 상태 변수
current_broadcast_type = None

# ✅ 실시간 ITS CCTV 주소를 보관할 딕셔너리 추가
CCTV_URLS = {}

# ✅ 수정: 특정 키를 미리 정의하지 않고 빈 딕셔너리로 변경
# 이제 alert_sent_session["cctv1_reverse"] = True 처럼 개별 관리가 가능해집니다.
alert_sent_session = {} 

sim_coords = {"lat": 37.5665, "lng": 126.9780}
current_video_file = {"fire": None, "reverse": None}
latest_frames = {} # 이 딕셔너리도 video_origin 키값에 따라 분리됩니다.

# 경로 설정
CAPTURE_DIR = os.path.join(os.getcwd(), "static", "captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)

ANOMALY_DATA = {
    "fire": {"type": "화재 발생"},
    "reverse": {"type": "역주행"},
    "webcam": {"type": "실시간 현장"}
}

simulation_active = False  # 시뮬레이션 실행 여부 플래그