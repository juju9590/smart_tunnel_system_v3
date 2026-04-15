# routes.py

from flask import Blueprint, jsonify

tunnel_bp = Blueprint("tunnel", __name__)

GLOBAL_STATE = "NORMAL"
GLOBAL_EVENT = "NONE"

@tunnel_bp.route("/status")
def status():
    return jsonify({
        "state": GLOBAL_STATE,
        "event": GLOBAL_EVENT
    })