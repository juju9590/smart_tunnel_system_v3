# ==========================================
# 파일명: tunnel_main_V5_1.py
# 설명:
# V5_1 최종 실행 파일
# - 테스트 영상 저장
# - 로그 CSV 저장
# - outputs/pipeline_v5_1 에만 저장
# ==========================================

import cv2
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from scr.pipeline_V5_1.pipeline_core_V5_1 import PipelineCore

print("🚀 프로그램 시작")

# ==========================
# 현재 파일 기준 절대경로
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# ==========================
# 입력 영상 경로
# ==========================
# VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "raw_video", "test_video", "test_normal_1.mp4")
# VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "raw_video", "test_video", "test_normal_2.mp4")
# VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "raw_video", "test_video", "test_congestion_2-1.mp4")
# VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "raw_video", "test_video", "test_congestion_2-2.mp4")
VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "raw_video", "test_video", "test_accident_1-1.mp4")
# VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "raw_video", "test_video", "test_accident_2.mp4")
# VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "raw_video", "test_video", "test_accident_3.mp4")

# ==========================
# 출력 경로
# ==========================
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "pipeline_v5_1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, f"result_{timestamp}.mp4")
LOG_PATH = os.path.join(OUTPUT_DIR, f"v5_1_{timestamp}.csv")

# ==========================
# 모델 경로
# ==========================
MODEL_PATH = os.path.join( PROJECT_ROOT,
    "scripts", "runs", "train","tunnel_final", "weights", "best.pt" )

print("📁 현재 파일 위치:", BASE_DIR)
print("📁 프로젝트 루트:", PROJECT_ROOT)
print("🎥 VIDEO_PATH:", VIDEO_PATH)
print("🎥 영상 존재 여부:", os.path.exists(VIDEO_PATH))
print("🤖 MODEL_PATH:", MODEL_PATH)
print("🤖 모델 존재 여부:", os.path.exists(MODEL_PATH))
print("💾 VIDEO_OUT_PATH:", VIDEO_OUT_PATH)
print("📝 LOG_PATH:", LOG_PATH)

# ==========================
# 차선 간단 추정
# ==========================
def get_lane(cx, frame_width):
    return 1 if cx < frame_width // 2 else 2


def main():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ 영상 열기 실패")
        return

    # -----------------------------
    # 영상 저장 설정
    # -----------------------------
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        VIDEO_OUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not out.isOpened():
        print("❌ VideoWriter 생성 실패")
        cap.release()
        return

    # -----------------------------
    # 모델 로드
    # -----------------------------
    if not os.path.exists(MODEL_PATH):
        print("❌ 모델 파일이 없음:", MODEL_PATH)
        cap.release()
        out.release()
        return

    model = YOLO(MODEL_PATH)
    pipeline = PipelineCore()

    frame_id = 0

    # -----------------------------
    # 로그 파일 생성
    # -----------------------------
    log_file = open(LOG_PATH, "w", newline="", encoding="utf-8-sig")
    writer = csv.writer(log_file)

    writer.writerow([
        "frame",
        "vehicle_count",
        "avg_speed",
        "state",
        "accident",
        "acc_ratio"
    ])

    print("✅ 저장 시작")
    print("   └ 영상:", VIDEO_OUT_PATH)
    print("   └ 로그:", LOG_PATH)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ℹ️ 영상 끝 또는 프레임 읽기 종료")
                break

            frame_id += 1
            h, w, _ = frame.shape

            # -----------------------------
            # YOLO + ByteTrack
            # -----------------------------
            results = model.track(
                frame,
                persist=True,
                conf=0.25,
                iou=0.5,
                tracker="bytetrack.yaml",
                verbose=False
            )

            tracks = []

            # 안전 처리 (중요)
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()

                for box, tid in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    tid = int(tid)

                    tracks.append({
                        "id": tid,
                        "bbox": (x1, y1, x2, y2)
                    })

            # -----------------------------
            # 파이프라인 처리
            # -----------------------------
            result = pipeline.process(frame_id, tracks)

            state = result.get("state", "NORMAL")
            accident = result.get("accident", False)
            avg_speed = float(result.get("avg_speed", 0.0))
            acc_ratio = result.get("acc_ratio", 0)

            # -----------------------------
            # 객체 시각화
            # -----------------------------
            for t in tracks:
                tid = t["id"]
                x1, y1, x2, y2 = t["bbox"]

                cx = int((x1 + x2) / 2)
                lane = get_lane(cx, w)

                # 상태 우선순위
                if accident:
                    color = (0, 0, 255)       # 빨강
                elif state == "JAM":
                    color = (255, 0, 0)       # 파랑 (주황으로 바꾸자)
                elif state == "CONGESTION":
                    color = (0, 255, 255)     # 노랑
                else:
                    color = (0, 255, 0)       # 초록

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    f"ID:{tid} L:{lane}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )

            # -----------------------------
            # 상단 상태 표시
            # -----------------------------
            cv2.putText(
                frame, f"STATE: {state}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3
            )

            cv2.putText(
                frame, f"AVG: {avg_speed:.2f}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )

            cv2.putText(
                frame, f"ACC_RATIO: {acc_ratio}", (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2
            )

            cv2.putText(
                frame, f"COUNT: {len(tracks)}", (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            if accident:
                cv2.putText(
                    frame,
                    "!!! ACCIDENT !!!",
                    (w // 3, h // 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    4
                )

            # -----------------------------
            # 영상 저장
            # -----------------------------
            out.write(frame)

            # -----------------------------
            # 로그 저장
            # -----------------------------
            writer.writerow([
                frame_id,
                len(tracks),
                round(avg_speed, 2),
                state,
                int(accident),
                acc_ratio
            ])

            # -----------------------------
            # 화면 출력
            # -----------------------------
            cv2.imshow("SMART TUNNEL V5_1", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                print("🛑 ESC 종료")
                break

    except Exception as e:
        print("❌ 실행 중 오류:", e)

    finally:
        log_file.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print("✅ 저장 완료")
        print("   └ 영상:", VIDEO_OUT_PATH)
        print("   └ 로그:", LOG_PATH)
        print("   └ 폴더:", OUTPUT_DIR)


if __name__ == "__main__":
    main()