# ==========================================
# 파일명: routes.py
# 위치: backend_flask/modules/tunnel/routes.py
# 역할:
# - health
# - 캐시된 CCTV 목록 조회
# - 프론트에서 CCTV 후보 리스트 저장
# - 랜덤 CCTV 선택
# - 이름으로 CCTV 선택
# - 상태 조회
# - 실시간 영상 스트리밍
# - 차선 재추정 요청
# - 차선 메모리 저장
#
# 참고:
# - 환기 대응(ventilation) 정보는 routes.py에서 계산하지 않음
# - service.py 내부에서 status에 ventilation 값을 포함하면
#   /status 응답에서 그대로 프론트로 전달됨
# ==========================================

from flask import Blueprint, jsonify, Response, request
from .service import TunnelLiveService

# ---------------------------------------------------------
# Blueprint 생성
# url_prefix="/api/tunnel" 이므로
# 실제 주소 예:
#   /api/tunnel/health
#   /api/tunnel/status
#   /api/tunnel/lane/reestimate
# ---------------------------------------------------------
tunnel_bp = Blueprint("tunnel", __name__, url_prefix="/api/tunnel")

# ---------------------------------------------------------
# 서비스 객체 1회 생성
# routes에서는 이 service를 통해 상태 조회 / 선택 / 스트리밍 / 재추정 요청 수행
# ---------------------------------------------------------
service = TunnelLiveService()


@tunnel_bp.route("/health", methods=["GET"])
def health():
    """
    간단한 모듈 헬스 체크
    """
    return jsonify({
        "ok": True,
        "module": "tunnel"
    })


@tunnel_bp.route("/cctv-list", methods=["GET"])
def cctv_list():
    """
    현재 캐시된 CCTV 리스트 조회
    """
    data = service.get_cctv_list()
    return jsonify({
        "ok": True,
        "count": len(data),
        "items": data
    })


@tunnel_bp.route("/set-cctv-list", methods=["POST"])
def set_cctv_list():
    """
    프론트에서 전달한 CCTV 후보 리스트를 서비스에 저장
    body 예시:
    {
        "items": [
            {"name": "...", "url": "..."},
            ...
        ]
    }
    """
    data = request.get_json(silent=True) or {}
    items = data.get("items", [])

    ok = service.set_cctv_list(items)
    if not ok:
        return jsonify({
            "ok": False,
            "message": "유효한 CCTV 리스트가 아닙니다."
        }), 400

    return jsonify({
        "ok": True,
        "message": "CCTV 리스트 저장 완료",
        "count": len(service.get_cctv_list())
    })


@tunnel_bp.route("/select-random", methods=["GET"])
def select_random():
    """
    캐시된 CCTV 중 하나를 랜덤 선택
    """
    cctv = service.select_random_cctv(user_random=True)

    if not cctv:
        return jsonify({
            "ok": False,
            "message": "캐시된 터널 CCTV 없음"
        }), 404

    return jsonify({
        "ok": True,
        "message": "랜덤 CCTV 선택 완료",
        "cctv": cctv
    })


@tunnel_bp.route("/select-cctv", methods=["GET"])
def select_cctv():
    """
    이름으로 CCTV 선택
    예:
    /api/tunnel/select-cctv?name=광암터널2
    """
    name = request.args.get("name", "").strip()

    if not name:
        return jsonify({
            "ok": False,
            "message": "name 파라미터가 필요합니다."
        }), 400

    cctv = service.select_cctv_by_name(name)

    if not cctv:
        return jsonify({
            "ok": False,
            "message": f"'{name}' 에 해당하는 CCTV를 찾지 못했습니다."
        }), 404

    return jsonify({
        "ok": True,
        "message": "CCTV 선택 완료",
        "cctv": cctv
    })


@tunnel_bp.route("/status", methods=["GET"])
def status():
    """
    현재 터널 분석 상태 조회

    주의:
    - service.get_status() 안에서 상태 dict를 만들어 반환한다.
    - ventilation 정보가 포함되어 있으면 그대로 프론트로 전달된다.
    - 예:
        {
            "ok": True,
            "state": "CONGESTION",
            "vehicle_count": 8,
            "avg_speed": 3.2,
            "lane_count": 2,
            "accident": False,
            "ventilation": {
                "risk_score_final": 0.58,
                "risk_level": "WARNING",
                "alarm": True,
                "message": "체류시간 증가 및 혼잡 발생, 환기 상태 점검 권고"
            }
        }
    """
    try:
        result = service.get_status()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "ok": False,
            "message": f"상태 조회 중 오류: {str(e)}"
        }), 500


@tunnel_bp.route("/video-feed", methods=["GET"])
def video_feed():
    """
    MJPEG 스트리밍 응답
    """
    return Response(
        service.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# =========================================================
# 차선 재추정 요청
# =========================================================
@tunnel_bp.route("/lane/reestimate", methods=["POST"])
def lane_reestimate():
    """
    관제사가 버튼을 누르면 호출하는 API

    동작 목적:
    - 지금 시점부터 50프레임 동안 차선 재추정용 데이터를 수집하도록 요청
    - 실제 50프레임 수집과 재군집화는 service / lane_template 내부에서 진행

    service.py 안에 아래 메서드가 반드시 있어야 함:
        service.request_lane_reestimate()

    반환 예시:
    {
        "ok": True,
        "message": "차선 재추정 요청 접수",
        "frame_id": 1234,
        "window": 50
    }
    """
    try:
        result = service.request_lane_reestimate()
    except AttributeError:
        return jsonify({
            "ok": False,
            "message": "service.py에 request_lane_reestimate() 메서드가 없습니다."
        }), 500
    except Exception as e:
        return jsonify({
            "ok": False,
            "message": f"차선 재추정 요청 중 오류: {str(e)}"
        }), 500

    status_code = 200 if result.get("ok", False) else 400
    return jsonify(result), status_code


# =========================================================
# 차선 메모리 저장
# =========================================================
@tunnel_bp.route("/lane/save", methods=["POST"])
def lane_save():
    """
    현재 차선 메모리를 저장하는 API
    service.py 안에 아래 메서드가 있어야 함:
        service.save_current_lane_memory()
    """
    try:
        result = service.save_current_lane_memory()
    except AttributeError:
        return jsonify({
            "ok": False,
            "message": "service.py에 save_current_lane_memory() 메서드가 없습니다."
        }), 500
    except Exception as e:
        return jsonify({
            "ok": False,
            "message": f"차선 저장 요청 중 오류: {str(e)}"
        }), 500

    status_code = 200 if result.get("ok") else 400
    return jsonify(result), status_code


# =========================================================
# 목표 차선 수 설정
# =========================================================
@tunnel_bp.route("/lane/target-count", methods=["POST"])
def lane_target_count():
    data = request.get_json(silent=True) or {}

    try:
        lane_count = int(data.get("lane_count"))
    except Exception:
        return jsonify({
            "ok": False,
            "message": "lane_count는 2, 3, 4 중 하나여야 합니다."
        }), 400

    if lane_count not in (2, 3, 4):
        return jsonify({
            "ok": False,
            "message": "lane_count는 2, 3, 4 중 하나여야 합니다."
        }), 400

    try:
        result = service.set_target_lane_count(lane_count)
    except AttributeError:
        return jsonify({
            "ok": False,
            "message": "service.py에 set_target_lane_count() 메서드가 없습니다."
        }), 500
    except Exception as e:
        return jsonify({
            "ok": False,
            "message": f"목표 차선 수 설정 중 오류: {str(e)}"
        }), 500

    status_code = 200 if result.get("ok", False) else 400
    return jsonify(result), status_code


# =========================================================
# 사고 이벤트 처리
# =========================================================
@tunnel_bp.route("/event/resolve", methods=["POST"])
def event_resolve():
    data = request.get_json(silent=True) or {}
    event_id = str(data.get("event_id", "")).strip()
    action = str(data.get("action", "")).strip()

    if not event_id or not action:
        return jsonify({
            "ok": False,
            "message": "event_id와 action이 필요합니다."
        }), 400

    try:
        result = service.resolve_accident_event(event_id, action)
    except AttributeError:
        return jsonify({
            "ok": False,
            "message": "service.py에 resolve_accident_event() 메서드가 없습니다."
        }), 500
    except Exception as e:
        return jsonify({
            "ok": False,
            "message": f"사고 이벤트 처리 중 오류: {str(e)}"
        }), 500

    status_code = 200 if result.get("ok", False) else 400
    return jsonify(result), status_code


# =========================================================
# 사고 이벤트 통계 조회
# =========================================================
@tunnel_bp.route("/event/stats", methods=["GET"])
def event_stats():
    date_text = request.args.get("date", "").strip() or None

    try:
        result = service.get_event_stats(date_text)
    except AttributeError:
        return jsonify({
            "ok": False,
            "message": "service.py에 get_event_stats() 메서드가 없습니다."
        }), 500
    except Exception as e:
        return jsonify({
            "ok": False,
            "message": f"사고 이벤트 통계 조회 중 오류: {str(e)}"
        }), 500

    return jsonify(result)

# =========================================================
# 스트림 명시 종료
# =========================================================
@tunnel_bp.route("/stream/stop", methods=["POST"])
def stream_stop():
    try:
        result = service.stop_stream()
    except AttributeError:
        return jsonify({
            "ok": False,
            "message": "service.py에 stop_stream() 메서드가 없습니다."
        }), 500
    except Exception as e:
        return jsonify({
            "ok": False,
            "message": f"스트림 종료 중 오류: {str(e)}"
        }), 500

    status_code = 200 if result.get("ok", False) else 400
    return jsonify(result), status_code


# =========================================================
# 새 랜덤 CCTV 선택 + 다음 video-feed용 준비
# =========================================================
@tunnel_bp.route("/stream/restart-random", methods=["POST"])
def stream_restart_random():
    try:
        result = service.restart_with_random_cctv()
    except AttributeError:
        return jsonify({
            "ok": False,
            "message": "service.py에 restart_with_random_cctv() 메서드가 없습니다."
        }), 500
    except Exception as e:
        return jsonify({
            "ok": False,
            "message": f"스트림 재시작 중 오류: {str(e)}"
        }), 500

    status_code = 200 if result.get("ok", False) else 400
    return jsonify(result), status_code
