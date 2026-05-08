import os
os.environ["OPENCV_FFMPEG_THREADS"] = "1"

import warnings
import logging
from dotenv import load_dotenv
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS

load_dotenv()

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`torch.distributed.reduce_op` is deprecated"
)

app = Flask(__name__)

CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "ngrok-skip-browser-warning"]
}})

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "fallback-secret-key")

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_timeout=60,
    ping_interval=25
)
app.extensions["socketio"] = socketio

# Smart Tunnel API 등록
from tunnel.tunnel import tunnel_bp

app.register_blueprint(tunnel_bp, url_prefix="/api/tunnel")


@socketio.on("resolve_emergency")
def handle_resolve(data):
    print(f"📡 조치 신호 전파: {data.get('alertId')}")
    emit("emergency_resolved", data, broadcast=True)


@app.route("/")
def index():
    return "Smart Tunnel V3 Personal Web Backend is Running"


log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=False,
        allow_unsafe_werkzeug=True
    )
