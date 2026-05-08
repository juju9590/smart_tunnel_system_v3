# ==========================================
# 파일명: export_lane_eval_csv.py
# 위치: backend_flask/tunnel/evaluation/export_lane_eval_csv.py
# 역할:
# - tunnel status API를 주기적으로 호출
# - 차선추정/차선관리 평가용 CSV만 저장
# - 기존 export_tunnel_status_csv.py와 분리된 평가 전용 exporter
#
# 실행:
#   python export_lane_eval_csv.py
# 종료:
#   Ctrl + C
# ==========================================

import csv
import os
import time
from datetime import datetime

import requests


STATUS_API_URL = "http://localhost:5000/api/tunnel/status"
POLL_INTERVAL_SEC = 1.0


CSV_COLUMNS = [
    "timestamp",
    "frame_id",
    "source_type",
    "source_name",
    "demo_video_filename",
    "cctv_name",
    "vehicle_count",
    "lane_count",
    "target_lane_count",
    "template_phase",
    "template_confirmed",
    "lane_reestimate_status",
    "lane_memory_loaded",
    "lane_memory_saved",
    "roi_y1",
    "roi_y2",
    "state",
    "accident",
    "note",
]


def safe_get(data, key, default=""):
    value = data.get(key, default)
    if value is None:
        return default
    return value


def build_row(data):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cctv_name = safe_get(data, "cctv_name", "")
    source_name = safe_get(data, "source_name", cctv_name)

    return {
        "timestamp": now_str,
        "frame_id": safe_get(data, "frame_id", 0),
        "source_type": safe_get(data, "source_type", ""),
        "source_name": source_name,
        "demo_video_filename": safe_get(data, "demo_video_filename", ""),
        "cctv_name": cctv_name,
        "vehicle_count": safe_get(data, "vehicle_count", 0),
        "lane_count": safe_get(data, "lane_count", 0),
        "target_lane_count": safe_get(data, "target_lane_count", ""),
        "template_phase": safe_get(data, "template_phase", ""),
        "template_confirmed": safe_get(data, "template_confirmed", ""),
        "lane_reestimate_status": safe_get(data, "lane_reestimate_status", ""),
        "lane_memory_loaded": safe_get(data, "lane_memory_loaded", ""),
        "lane_memory_saved": safe_get(data, "lane_memory_saved", ""),
        "roi_y1": safe_get(data, "roi_y1", ""),
        "roi_y2": safe_get(data, "roi_y2", ""),
        "state": safe_get(data, "state", ""),
        "accident": safe_get(data, "accident", False),
        "note": "",
    }


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "runtime_data", "csv_logs")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"lane_eval_{timestamp}.csv")

    print("==========================================")
    print("🚀 Lane Eval CSV Exporter 시작")
    print(f"STATUS API : {STATUS_API_URL}")
    print(f"저장 경로  : {output_path}")
    print("종료하려면 Ctrl + C")
    print("==========================================")

    csv_file = open(output_path, "w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    csv_file.flush()

    try:
        while True:
            try:
                response = requests.get(STATUS_API_URL, timeout=5)
                response.raise_for_status()
                data = response.json()
            except Exception as exc:
                print(f"❌ status API 호출 실패: {exc}")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            row = build_row(data)
            writer.writerow(row)
            csv_file.flush()

            print(
                f"✅ frame={row['frame_id']} | source={row['source_name']} | "
                f"lane={row['lane_count']} | target={row['target_lane_count']} | "
                f"template={row['template_confirmed']}"
            )

            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n🛑 Lane eval CSV 추출 종료")

    finally:
        csv_file.close()
        print(f"💾 CSV 파일 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
