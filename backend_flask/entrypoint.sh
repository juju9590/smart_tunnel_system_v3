#!/bin/bash

# 1. DB 서버가 준비될 때까지 대기
echo "Waiting for DB ($DB_HOST) to start..."
while ! nc -z $DB_HOST $DB_PORT; do
    sleep 1
done
echo "DB is up - executing command"

# 2. migrations 폴더 초기화 (없을 때만)
if [ ! -d "migrations" ]; then
    echo "Initializing migrations..."
    flask db init
fi

# 3. 마이그레이션 반영 로직 보강
echo "Running migrations..."

# 히스토리가 꼬였을 경우를 대비해 현재 DB 상태를 코드상의 최신 버전으로 강제 동기화 (핵심 추가)
flask db stamp head || echo "Stamp failed, moving on..."

# 변경사항 있으면 마이그레이션 생성
flask db migrate -m "Auto-migration" || echo "No changes in model"

# 최종 업그레이드
flask db upgrade || echo "Upgrade failed, check if table already exists"

# 4. Flask 앱 실행
echo "Starting Flask app..."
exec python app.py