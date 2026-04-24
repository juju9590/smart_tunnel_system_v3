# 🤝 TADS 프로젝트 협업 가이드 (Contributing Guidelines)

TADS(Traffic Monitoring and Anomaly Detection) 프로젝트에 참여해 주셔서 감사합니다! 우리 팀의 일관된 코드 관리와 안정적인 배포를 위해 아래 규칙을 반드시 준수해 주세요.

---

## 📍 1. 저장소 구조 (Repository Strategy)

우리 프로젝트는 Fork 방식이 아닌 **단일 공유 저장소(Collaborator)** 모델을 사용합니다. **절대 Fork 하지 마세요!**

* **Origin (원격 저장소):** 팀장이 관리하는 공식 본진 (우리 모두의 기준점)
* **Local (내 컴퓨터):** 실제로 코딩을 진행하는 각자의 작업 공간

---

## 🚀 2. 개발 워크플로우 (Step-by-Step)

### Step 0. 프로젝트 최초 세팅 (처음 1회만)
1. 깃허브 알림이나 이메일로 온 **저장소 초대(Invitation)를 수락**합니다.
2. 터미널을 열고 팀장 저장소를 내 컴퓨터로 직접 **Clone** 합니다.

```bash
git clone [https://github.com/qksdkqksdk5/TADS_Project.git](https://github.com/qksdkqksdk5/TADS_Project.git)
cd TADS_Project

Step 1. 작업 시작 전 (최신화 동기화)
매일 작업을 시작하기 전, 또는 새로운 기능을 개발하기 전에는 반드시 최신 코드를 받아와야 충돌을 방지할 수 있습니다.
Bash
git checkout main
git pull origin main

Step 2. 브랜치 생성 (Branching)
절대로 main 브랜치에서 직접 코딩하지 마세요. 반드시 기능별 브랜치를 생성하고 이동합니다.
Bash
# 브랜치 이름 규칙: 분류/기능명 (예: feature/login-ui)
git checkout -b feature/기능명

Step 3. 작업 및 커밋 (Commit)
작업 단위별로 자주 커밋해 주세요. 커밋 메시지는 다른 사람이 봐도 알 수 있게 명확히 적습니다.
Bash
git add .
git commit -m "feat: ○○ 기능 추가 및 로직 수정"

Step 4. 제출 (Push & Pull Request)
작업이 완료되면 본인의 브랜치를 원격 저장소(origin)에 올리고 PR을 날립니다.
Bash
# origin(공유 저장소)에 내 브랜치를 푸시
git push origin feature/기능명
GitHub 웹페이지 접속 후 초록색 [Compare & pull request] 버튼 클릭

변경 사항 요약 작성 후 [Create pull request] 클릭

⚠️ 3. 협업 골든 룰 (Golden Rules)
🚨 main 브랜치 직접 Push 절대 금지: main 브랜치는 오직 PR을 통해서만 코드가 합쳐집니다. 실수로 직접 푸시할 경우 시스템 보호를 위해 즉시 강제 롤백(Revert) 처리됩니다.

공유 코어 로직 수정 주의: .env, requirements.txt, manager.py 등 공통 설정 파일은 수정 전 팀장(이동훈)과 사전 협의가 필요합니다.

PR 전 Pull 필수: 내 코드를 올리기 직전에 반드시 git pull origin main을 실행하여 최신 상태인지 확인하세요.

코드 리뷰: 팀장이 PR을 검토하고 승인(Merge)해야만 최종 반영됩니다. 수정 요청이 오면 해당 브랜치에서 다시 수정 후 push 하세요.

망했을 땐 SOS: 깃(Git)이 심하게 꼬였거나 에러가 도저히 안 잡힌다면 혼자 끙끙대지 말고 주저 없이 팀장에게 헬프를 요청해 주세요!

🛠️ 4. 브랜치 네이밍 컨벤션
feature/ : 새로운 기능 추가

fix/ : 버그 수정

docs/ : 문서 수정 (README 등)

refactor/ : 코드 리팩토링 (기능 변화 없이 구조 개선)

5. 배포 성공 시 백업

git checkout main
git pull origin main

git tag v1.0.1 : 태그 생성 (버전 이름은 자유롭게)

git push origin v1.0.1 : 태그를 원격 저장소에 올리기

Happy Coding!