import os
import re
from datetime import datetime
from flask import Blueprint, request, jsonify
from openai import OpenAI
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
chat_bp = Blueprint('chat', __name__)

# ============================================================
# [설정 1] MySQL DB 연결 (커넥션 풀 포함)
# ============================================================
DB_URL = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(
    DB_URL,
    pool_size=5,        # 🟡 수정: 커넥션 풀 설정 (동시접속 안정성)
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800   # 30분마다 커넥션 갱신
)

# ============================================================
# [설정 2] MongoDB 연결 (실패해도 서버 다운 방지)
# ============================================================
history_col = None
mongo_client = None

# chat.py 내부 get_mongo_collection() 수정
def get_mongo_collection():
    global history_col, mongo_client
    if history_col is not None:
        return history_col
        
    try:
        MONGO_URI = os.getenv("MONGO_URI")
        # [수정] waitQueueMultiple 같은 Gevent 전용 옵션 제거 (표준 연결)
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        mongo_client.server_info()
        history_col = mongo_client["TADS_DB"]["chat_history"]
        print("MongoDB 연결 성공")
    except Exception as e:
        print(f"MongoDB 연결 실패: {e}")
        history_col = None
        
    return history_col

# ============================================================
# [설정 3] OpenAI 클라이언트
# ============================================================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# [설정 4] RAG 초기화 (실패해도 서버 다운 방지)
# ============================================================
def init_rag_system():
    knowledge_path = os.path.join(BASE_DIR, "TADS_knowledge.txt")
    persist_db_path = os.path.join(BASE_DIR, "chroma_db")

    if not os.path.exists(knowledge_path):
        print(f"경고: {knowledge_path} 파일이 없습니다.")
        return None

    embeddings = OpenAIEmbeddings()

    # 🟡 수정: 이미 persist된 DB가 있으면 재생성 없이 로드만 (임베딩 비용 절약)
    if os.path.exists(persist_db_path) and os.listdir(persist_db_path):
        print("기존 Chroma DB 로드")
        return Chroma(
            persist_directory=persist_db_path,
            embedding_function=embeddings
        )

    print("Chroma DB 새로 생성")
    loader = TextLoader(knowledge_path, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_db_path
    )

try:
    vector_db = init_rag_system()  # 🔴 수정: RAG 초기화 실패해도 서버 다운 방지
except Exception as e:
    print(f"RAG 초기화 실패 (RAG 비활성화): {e}")
    vector_db = None

# ============================================================
# [설정 5] 스키마 컨텍스트 로드
# ============================================================
with open(os.path.join(BASE_DIR, "schema_context.txt"), "r", encoding="utf-8") as f:
    SCHEMA_CONTEXT = f.read()

# ============================================================
# [설정 6] SQL 인젝션 방어 - 읽기 전용 체크
# ============================================================
FORBIDDEN_KEYWORDS = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'TRUNCATE']

def is_safe_query(query: str) -> bool:
    upper = query.upper()
    return not any(kw in upper for kw in FORBIDDEN_KEYWORDS)


# ============================================================
# [라우터] POST /ask
# ============================================================
@chat_bp.route('/ask', methods=['POST'])
def ask_tads():
    data = request.json
    user_question = data.get('question')
    user_name = data.get('userName', '익명 사용자')

    intent = "UNKNOWN"
    row_count = 0
    rows = []
    query = ""
    knowledge_context = ""

    if not user_question:
        return jsonify({"answer": "질문을 입력해 주세요."}), 400

    try:
        # --------------------------------------------------------
        # [Step 0] MongoDB에서 직전 대화 기록 조회
        # --------------------------------------------------------
        last_log = None
        last_intent_for_prompt = 'UNKNOWN'
        prev_info = ""

        current_history_col = get_mongo_collection()

        # 🔴 수정: history_col이 None이면 스킵 (MongoDB 연결 실패 시 크래시 방지)
        if current_history_col is not None:
            last_log = current_history_col.find_one(
                {"user_name": user_name},
                sort=[("timestamp", -1)]
            )
            if last_log:
                last_intent_for_prompt = last_log.get('intent', 'UNKNOWN')
                prev_info = f"사용자의 직전 질문: {last_log['user_question']}\nAI의 직전 답변: {last_log['ai_answer']}"

        # --------------------------------------------------------
        # [Step 1] 의도 파악 (SQL / RAG / BOTH)
        # --------------------------------------------------------
        intent_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""사용자 질문의 의도를 'SQL', 'RAG', 'BOTH' 중 하나로 분류하세요.
                    이전 질문의 의도는 {last_intent_for_prompt}였습니다. 이를 참고하여 현재 질문의 의도를 분류하세요.

                    만약 '화재는?', '역주행은?' 처럼 질문이 짧고 이전 의도가 SQL이었다면,
                    이번에도 해당 항목의 데이터를 조회하려는 SQL 의도로 간주하세요.

                    1. SQL: '결과', '목록', '기록', '데이터', '찾아줘', '보여줘', '~영상' 등 특정 데이터를 조회하려는 의도가 조금이라도 있는 경우.
                    2. RAG: 시스템의 정의, 작동 원리, 대처 방법, 기능 설명 등 일반적인 지식을 묻는 경우.
                    3. BOTH: 데이터를 보여주면서 동시에 설명이나 절차도 필요한 경우.

                    - '너는 뭐야?', '어떤 도움을 줄 수 있어?', '안녕' 등 시스템이나 AI 자체에 대한 질문,
                      또는 일상적인 인사말은 반드시 'RAG'로 분류하세요.
                    - 데이터나 DB와 전혀 관련 없는 질문은 절대 'SQL'로 분류하지 마세요.

                    - 질문에 '마지막', '이전', '아까', '질문', '뭐였어', '뭐라고' 등이 포함된 경우:
                      SQL 조회가 아닌 대화 맥락 확인이므로 반드시 'RAG'로 분류하세요.

                    - '답', '답변', '뭐라고 했어', '어떻게 답했어' 등 이전 대화 내용을 묻는 경우:
                      반드시 'RAG'로 분류하세요.
                    
                    - '아', '근데', '말고', '아니고' 같은 구어체 수정 표현이 포함되어도
                      데이터 조회 의도가 있으면 반드시 SQL로 분류하세요.
                      예: "아 이번주말고 저번주" → SQL

                    [이전 대화 정보]
                    {prev_info if prev_info else "이전 기록 없음"}
                    """
                },
                {"role": "user", "content": user_question}
            ],
            temperature=0,
            timeout=30  # 🟢 수정: OpenAI 타임아웃 설정
        )

        # intent 정제 - LLM이 "SQL입니다" 같이 반환하는 경우 대비
        raw_intent = intent_completion.choices[0].message.content.strip().upper()
        intent = next((i for i in ['BOTH', 'SQL', 'RAG'] if i in raw_intent), 'UNKNOWN')

        # --------------------------------------------------------
        # [Step 2] RAG 지식 검색
        # --------------------------------------------------------
        if intent in ['RAG', 'BOTH', 'UNKNOWN']:
            if vector_db:
                related_docs = vector_db.similarity_search(user_question, k=2)
                knowledge_context = "\n".join([doc.page_content for doc in related_docs])
                if not knowledge_context.strip():
                    knowledge_context = "관련 운영 지침 없음"

        # --------------------------------------------------------
        # [Step 3] SQL 생성 및 DB 실행
        # --------------------------------------------------------
        if intent in ['SQL', 'BOTH']:
            sql_completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SCHEMA_CONTEXT + "\n순수 SQL만 반환하세요."},
                    {"role": "user", "content": user_question}
                ],
                temperature=0,
                timeout=30  # 🟢 수정: OpenAI 타임아웃 설정
            )
            raw_query = sql_completion.choices[0].message.content.strip()
            match = re.search(r"(SELECT.*)", raw_query, re.IGNORECASE | re.DOTALL)
            query = match.group(1).split(';')[0] + ';' if match else raw_query

            if not is_safe_query(query):
                return jsonify({"answer": "데이터 조회만 가능합니다."}), 400

            if "SELECT" in query.upper():
                with engine.connect() as connection:
                    result = connection.execute(text(query))
                    rows = [dict(row._mapping) for row in result]
                    row_count = len(rows)

        # --------------------------------------------------------
        # [Step 4] 최종 답변 생성
        # --------------------------------------------------------
        answer_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""당신은 TADS 관제 시스템 비서입니다.

                [직전 대화 기록]
                - 직전 질문: {last_log['user_question'] if last_log else '없음'}
                - 직전 답변: {last_log['ai_answer'] if last_log else '없음'}

                [운영 지침]
                {knowledge_context if knowledge_context else '없음'}

                [답변 규칙 - 순서대로 판단]
                1. 질문에 '마지막/이전/아까/질문/답/답변/뭐였어/뭐라고' 포함 시 → [직전 대화 기록]만 참고
                - '질문' 언급 → "마지막 질문은 'OOO'였습니다."
                - '답/답변' 언급 → "마지막 답변은 'OOO'였습니다."
                - '답은?' 단독 → 직전 질문과 연결해서 답변
                - 둘 다 물으면 둘 다 알려주세요.
                2. '방법/절차/어떻게' 포함 시 → [운영 지침] 기반으로 설명, 데이터 건수 언급 금지
                3. intent가 'RAG'인 경우 → [운영 지침] 기반으로 자유롭게 답변, 데이터 건수 절대 언급 금지
                4. intent가 'SQL' 또는 'BOTH'인 경우에만 아래 적용:  ← 핵심 수정
                - 데이터 있음 → "총 {row_count}건이 확인되었습니다. 상세 내역은 표를 확인해 주세요."
                - 데이터 없음 → "조회 결과 해당 데이터가 존재하지 않습니다."
                """
                },
                {
                    "role": "user",
                    "content": f"질문: {user_question}\n판단된 의도: {intent}\n실제 데이터 건수: {row_count}건\n데이터 존재 여부: {'있음' if row_count > 0 else '없음'}"
                }
            ],
            temperature=0,
            timeout=30  # 🟢 수정: OpenAI 타임아웃 설정
        )

        final_answer = answer_response.choices[0].message.content

        # --------------------------------------------------------
        # [Step 5] MongoDB 대화 기록 저장
        # --------------------------------------------------------
        # 🔴 수정: history_col None 체크 후 저장
        if history_col is not None:
            try:
                # 🟢 수정: query 없으면 키 자체를 제외 (None 저장 대신)
                doc = {
                    "timestamp": datetime.now(),
                    "user_name": user_name,
                    "user_question": user_question,
                    "ai_answer": final_answer,
                    "intent": intent,
                    "row_count": row_count,
                    "success": True
                }
                if query:
                    doc["sql_query"] = query
                current_history_col.insert_one(doc)
            except Exception as mongo_err:
                print(f"MongoDB 저장 실패: {mongo_err}")

        # --------------------------------------------------------
        # [Step 6] 응답 반환
        # --------------------------------------------------------
        return jsonify({
            "answer": final_answer,
            "data": rows,
            "query": query
        })

    except Exception as e:
        print(f"에러 발생: {e}")
        return jsonify({"answer": f"분석 중 오류 발생: {str(e)}"}), 500