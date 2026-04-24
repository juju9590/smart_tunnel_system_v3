import requests
import json
import os
from datetime import datetime

def send_discord_notification(url, event_type, location, image_path):
    # 1. 서버 주소 설정
    MY_SERVER_URL = "http://220.76.197.149:5173"
    
    # 2. 타입 분기 (화재/역주행 등)
    is_fire = "fire" in event_type.lower() or "화재" in event_type
    
    # 테마 색상 및 이름만 심플하게 설정
    color = 15158332 if is_fire else 3447003 # 화재: 빨강, 그 외: 파랑
    system_name = "AI 모니터링 시스템"
    severity = "심각" if is_fire else "주의"

    # 3. 군더더기 없는 깔끔한 페이로드 (이미지 크기 고정 느낌)
    payload = {
        "username": system_name,
        "embeds": [{
            "title": f"[{event_type}] 탐지 알림",
            "description": f"{location} 구역에서 이상 징후가 감지되었습니다.\n\n**[관제 시스템으로 이동하여 확인하기]({MY_SERVER_URL})**",
            "color": color,
            "fields": [
                {
                    "name": "탐지 위치",
                    "value": location,
                    "inline": True
                },
                {
                    "name": "발생 시각",
                    "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "inline": True
                },
                {
                    "name": "위험 등급",
                    "value": severity,
                    "inline": True
                }
            ],
            "image": {"url": "attachment://file.jpg"} # 업로드한 이미지 파일을 임베드에 표시 (비율 유지하며 너비 고정)
        }]
    }

    try:
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                # 💡 핵심: str().replace() 대신 json.dumps()를 사용해야 안전합니다.
                response = requests.post(
                    url, 
                    data={'payload_json': json.dumps(payload)}, 
                    files={'file': ('file.jpg', f, 'image/jpeg')}
                )
        else:
            response = requests.post(url, json=payload)
            
        # 디스코드 응답 결과 확인 (디버깅용)
        if response.status_code in [200, 204]:
            print(f"✅ [디스코드] {event_type} 알림 전송 성공")
        else:
            print(f"❌ [디스코드] 응답 오류: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ [디스코드] 연결 에러: {e}")

