# ==========================================
# 파일명: tunnel.py
# 위치: backend_flask/modules/tunnel/tunnel.py
# 역할:
# - tunnel blueprint 외부 노출
# - app.py 등에서 import 해서 등록할 때 사용
# ==========================================

from .routes import tunnel_bp

# ---------------------------------------------------------
# 사용 예시 (app.py 쪽):
#
#   from modules.tunnel.tunnel import tunnel_bp
#   app.register_blueprint(tunnel_bp)
#
# 현재 tunnel 관련 라우트는 모두 routes.py 안에 작성한다.
# 추가 API가 필요하면 routes.py에 계속 확장하면 된다.
# ---------------------------------------------------------