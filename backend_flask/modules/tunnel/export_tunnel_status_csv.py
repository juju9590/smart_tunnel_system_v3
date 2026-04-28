# ==========================================
# 파일명: export_tunnel_status_csv.py
# 위치: backend_flask/modules/tunnel/export_tunnel_status_csv.py
# 역할:
# - tunnel status API를 주기적으로 호출
# - 상태 / 차량 / 이벤트 로그를 CSV로 저장
# - 기존 서비스 코드 수정 없이 별도 추출용으로 사용
# 실행:
#   python export_tunnel_status_csv.py
# 종료:
#   Ctrl + C
# ==========================================

import os
import csv
import time
import requests
from datetime import datetime


STATUS_API_URL = "http://localhost:5000/api/tunnel/status"
POLL_INTERVAL_SEC = 1.0


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "runtime_data", "csv_logs")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    status_csv_path = os.path.join(output_dir, f"status_log_{timestamp}.csv")
    vehicle_csv_path = os.path.join(output_dir, f"vehicle_log_{timestamp}.csv")
    event_csv_path = os.path.join(output_dir, f"event_log_{timestamp}.csv")

    status_file = open(status_csv_path, "w", newline="", encoding="utf-8-sig")
    vehicle_file = open(vehicle_csv_path, "w", newline="", encoding="utf-8-sig")
    event_file = open(event_csv_path, "w", newline="", encoding="utf-8-sig")

    status_writer = csv.writer(status_file)
    vehicle_writer = csv.writer(vehicle_file)
    event_writer = csv.writer(event_file)

    # ------------------------------------------
    # CSV 헤더
    # ------------------------------------------
    status_writer.writerow([
        "timestamp",
        "frame_id",
        "cctv_name",
        "state",
        "avg_speed",
        "vehicle_count",
        "lane_count",
        "accident",
        "weak_suspect",
        "strong_suspect",
        "confirm_candidate",
        "has_real_accident_evidence",
        "accident_score",
        "recent_prediction_count",
        "vehicle_drop",
        "defense_large_vehicle_occlusion",
        "defense_jam_stationary_cell",
        "defense_bbox_occlusion_false_stop",
        "pair_evidence_raw",
        "pair_collision_valid",
        "congestion_pair_only_false",
        "reasons",
    ])

    vehicle_writer.writerow([
        "timestamp",
        "frame_id",
        "cctv_name",
        "vehicle_id",
        "speed",
        "lane",
        "raw_lane",
        "dwell_time",
        "bbox",
    ])

    event_writer.writerow([
        "timestamp",
        "frame_id",
        "cctv_name",
        "event_message",
    ])

    # 이벤트 중복 저장 방지
    seen_event_keys = set()

    print("==========================================")
    print("🚀 Tunnel CSV Exporter 시작")
    print(f"STATUS API : {STATUS_API_URL}")
    print(f"상태 로그  : {status_csv_path}")
    print(f"차량 로그  : {vehicle_csv_path}")
    print(f"이벤트 로그: {event_csv_path}")
    print("종료하려면 Ctrl + C")
    print("==========================================")

    try:
        while True:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            try:
                response = requests.get(STATUS_API_URL, timeout=5)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                print(f"❌ status API 호출 실패: {e}")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            frame_id = safe_int(data.get("frame_id", 0))
            cctv_name = str(data.get("cctv_name", "-"))
            state = str(data.get("state", "READY"))
            avg_speed = safe_float(data.get("avg_speed", 0.0))
            vehicle_count = safe_int(data.get("vehicle_count", 0))
            lane_count = safe_int(data.get("lane_count", 0))
            accident = bool(data.get("accident", False))
            accident_locked = bool(data.get("accident_locked", False))
            final_accident_event = bool(accident or accident_locked)
            weak_suspect = bool(data.get("weak_suspect", False))
            strong_suspect = bool(data.get("strong_suspect", False))
            confirm_candidate = bool(data.get("confirm_candidate", False))
            has_real_accident_evidence = bool(data.get("has_real_accident_evidence", False))
            accident_score = safe_float(data.get("accident_score", 0.0))
            recent_prediction_count = safe_int(data.get("recent_prediction_count", 0))
            vehicle_drop = safe_int(data.get("vehicle_drop", 0))
            defense_large_vehicle_occlusion = bool(data.get("defense_large_vehicle_occlusion", False))
            defense_jam_stationary_cell = bool(data.get("defense_jam_stationary_cell", False))
            defense_bbox_occlusion_false_stop = bool(data.get("defense_bbox_occlusion_false_stop", False))
            pair_evidence_raw = bool(data.get("pair_evidence_raw", False))
            pair_collision_valid = bool(data.get("pair_collision_valid", False))
            congestion_pair_only_false = bool(data.get("congestion_pair_only_false", False))
            reasons = str(data.get("reasons", ""))

            # ------------------------------------------
            # 1) 상태 로그 저장 (프레임당 1줄)
            # ------------------------------------------
            status_writer.writerow([
                now_str,
                frame_id,
                cctv_name,
                state,
                round(avg_speed, 2),
                vehicle_count,
                lane_count,
                accident,
                weak_suspect,
                strong_suspect,
                confirm_candidate,
                has_real_accident_evidence,
                round(accident_score, 2),
                recent_prediction_count,
                vehicle_drop,
                defense_large_vehicle_occlusion,
                defense_jam_stationary_cell,
                defense_bbox_occlusion_false_stop,
                pair_evidence_raw,
                pair_collision_valid,
                congestion_pair_only_false,
                reasons,
            ])
            status_file.flush()

            # ------------------------------------------
            # 2) 차량 로그 저장
            # status API에 vehicles가 있으면 저장
            # ------------------------------------------
            vehicles = data.get("vehicles", [])
            if isinstance(vehicles, list):
                for v in vehicles:
                    vehicle_writer.writerow([
                        now_str,
                        frame_id,
                        cctv_name,
                        v.get("id", ""),
                        safe_float(v.get("speed", 0.0)),
                        v.get("lane", ""),
                        v.get("raw_lane", ""),
                        safe_float(v.get("dwell_time", 0.0)),
                        v.get("bbox", ""),
                    ])
                vehicle_file.flush()

            # ------------------------------------------
            # 3) 이벤트 로그 저장
            # event_log_entries가 있으면 event_id 기준으로 새 이벤트만 저장
            # legacy event_logs/events는 같은 실행 중 같은 메시지를 중복 저장하지 않는다.
            # ------------------------------------------
            event_log_entries = data.get("event_log_entries", None)
            event_logs = data.get("event_logs", None)
            wrote_event = False

            if isinstance(event_log_entries, list) and len(event_log_entries) > 0:
                for event_item in event_log_entries:
                    if isinstance(event_item, dict):
                        event_id = event_item.get("event_id", "")
                        event_message = event_item.get("text") or event_item.get("message", "")
                    else:
                        event_id = ""
                        event_message = str(event_item)

                    event_message = str(event_message)
                    if "사고 감지" in event_message and not final_accident_event:
                        continue

                    event_key = ("event_id", cctv_name, event_id) if event_id != "" else ("event", cctv_name, event_message)
                    if event_key in seen_event_keys:
                        continue

                    seen_event_keys.add(event_key)

                    event_writer.writerow([
                        now_str,
                        frame_id,
                        cctv_name,
                        event_message,
                    ])
                    wrote_event = True

                if wrote_event:
                    event_file.flush()

            elif isinstance(event_logs, list) and len(event_logs) > 0:
                for event_message in event_logs:
                    event_message = str(event_message)
                    if "사고 감지" in event_message and not final_accident_event:
                        continue

                    event_key = ("event_log", cctv_name, event_message)
                    if event_key in seen_event_keys:
                        continue

                    seen_event_keys.add(event_key)

                    event_writer.writerow([
                        now_str,
                        frame_id,
                        cctv_name,
                        event_message,
                    ])
                    wrote_event = True

                if wrote_event:
                    event_file.flush()

            else:
                events = data.get("events", [])
                if isinstance(events, list) and len(events) > 0:
                    for event_message in events:
                        event_message = str(event_message)
                        if "사고 감지" in event_message and not final_accident_event:
                            continue

                        event_key = ("event", cctv_name, event_message)
                        if event_key in seen_event_keys:
                            continue

                        seen_event_keys.add(event_key)

                        event_writer.writerow([
                            now_str,
                            frame_id,
                            cctv_name,
                            event_message,
                        ])
                        wrote_event = True

                    if wrote_event:
                        event_file.flush()

            print(
                f"✅ frame={frame_id} | cctv={cctv_name} | "
                f"state={state} | speed={avg_speed:.2f} | "
                f"vehicles={vehicle_count} | accident={accident}"
            )

            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n🛑 CSV 추출 종료")

    finally:
        status_file.close()
        vehicle_file.close()
        event_file.close()
        print("💾 CSV 파일 저장 완료")


if __name__ == "__main__":
    main()
