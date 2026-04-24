FROM python:3.11-slim

WORKDIR /app

# netcat(nc) 추가 설치 (DB 대기용)
RUN apt-get update && apt-get install -y \
    ffmpeg gcc g++ python3-dev libgl1 libglib2.0-0 libgomp1 \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

ENV TZ=Asia/Seoul

COPY . .

# 실행 스크립트에 권한 부여
RUN chmod +x entrypoint.sh

EXPOSE 5000

# CMD 대신 entrypoint.sh 실행
ENTRYPOINT ["./entrypoint.sh"]