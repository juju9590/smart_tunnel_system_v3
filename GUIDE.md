## 💻 1. Frontend Setup (React / Vite)

프론트엔드 실행을 위해 아래 명령어를 순서대로 입력하세요.

```bash
cd frontend_js
npm install
npm run dev
```

---

## ⚙️ 2. Simulation & Environment (.env)

데이터 및 환경 변수 설정을 위해 아래 작업을 수행하세요.

### 📁 Assets 설정
- `assets` 폴더를 다운로드하여  
  `backend_flask/` 경로에 삽입합니다.

### 📁 test 설정
- `test` 폴더를 다운로드하여  
  `backend_flask/modules/plate/` 경로에 삽입합니다.

### 🔑 환경 변수 설정
- `env.txt` 파일을 다운로드 후  
- 프로젝트 최상단 폴더에 위치시키고  
- 파일명을 `.env`로 변경합니다.

---

## 🐍 3. Backend Setup (Flask)

백엔드 가상환경 설정 및 라이브러리 설치 가이드입니다.

⚠️ **주의:**  
작업 전 `backend_flask` 폴더 내 기존 `migrations` 폴더가 있다면 반드시 삭제하세요.

```bash
cd backend_flask

# 1. Conda 가상환경 생성 및 활성화
conda create -n tads python=3.11 -y
conda activate tads

# 2. 필수 라이브러리 설치
pip install -r requirements.txt
```

---

## 🗄️ 4. Database Setup (MySQL)

데이터베이스 초기화 및 테이블 생성을 위한 단계입니다.

### 🛠️ DB 생성

MySQL Workbench 또는 VSCode Database 확장에서 아래 쿼리를 실행하세요.

```sql
CREATE DATABASE tads;
```

### 🔄 Migration 실행

```bash
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

### ▶️ 서버 실행

```bash
python app.py
```

---

## 🤝 5. Collaboration Guide

협업 규칙(Fork, PR, 브랜치 전략)은 아래 문서를 참고하세요.

👉 [CONTRIBUTING.md 보러가기](./CONTRIBUTING.md)

---

## 🚀 Quick Start

빠르게 실행하고 싶다면 아래 순서대로 실행하세요.

```bash
# frontend
cd frontend_js
npm install
npm run dev

# backend
cd backend_flask
conda activate tads
python app.py
```

---

© 2026 TADS Team. All Rights Reserved.