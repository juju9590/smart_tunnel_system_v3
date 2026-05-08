## 2. Backend 설정

### 2-1. 백엔드 폴더 이동

프로젝트 루트에서 백엔드 폴더로 이동합니다.

```bash
cd backend_flask
```

---

### 2-2. Python 패키지 설치

백엔드 실행에 필요한 Python 패키지를 설치합니다.

```bash
python -m pip install -r requirements.txt
```

---

## 3. 환경변수 설정

실시간 CCTV 조회를 위해 ITS API 설정이 필요합니다.

`backend_flask/.env.example` 파일을 참고하여 `backend_flask/.env` 파일을 생성합니다.

```bash
copy .env.example .env
```

생성된 `.env` 파일을 열고 실제 값을 입력합니다.

```env
ITS_API_KEY=your_its_api_key_here
ITS_CCTV_API_URL=your_its_cctv_api_url_here
```

실제 입력 예시는 아래와 같습니다.

```env
ITS_API_KEY=실제_ITS_API_KEY
ITS_CCTV_API_URL=https://openapi.its.go.kr:9443/cctvInfo
```

> `.env` 파일은 API Key가 포함되므로 GitHub에 올리지 않습니다.  
> GitHub에는 `.env.example` 파일만 포함합니다.

---

## 4. Backend 실행

`backend_flask` 폴더에서 Flask 서버를 실행합니다.

```bash
python app.py
```

정상 실행 시 아래 주소에서 백엔드 서버가 실행됩니다.

```text
http://127.0.0.1:5000
```

---

## 5. Backend 동작 확인

백엔드 서버가 실행 중인 상태에서 브라우저 또는 API 도구로 아래 주소를 확인합니다.

```text
http://127.0.0.1:5000/api/tunnel/cctv-list
```

정상 동작 시 CCTV 목록 JSON 응답이 반환됩니다.

만약 CCTV 연결이 실패하면 다음 항목을 확인합니다.

- `backend_flask/.env` 파일이 존재하는지
- `ITS_API_KEY` 값이 실제 키인지
- `ITS_CCTV_API_URL` 값이 올바른지
- 백엔드 서버가 실행 중인지
- 브라우저 Network에서 `/api/tunnel/cctv-list` 요청이 정상 응답하는지