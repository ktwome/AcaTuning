import os
import numpy as np
import pymysql
import chromadb
import logging
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from dotenv import load_dotenv

load_dotenv()
# ✅ Flask 설정
app = Flask(__name__)

# ✅ 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ 벡터 임베딩 모델 로드
logging.info("🔄 Loading embedding model...")
embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
logging.info("✅ Embedding model loaded successfully.")

# ✅ ChromaDB 클라이언트 설정
logging.info("🔄 Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="chroma_db")
lecture_vector_db = chroma_client.get_or_create_collection("lecture_embeddings")
logging.info("✅ Connected to ChromaDB.")

# ✅ MySQL 연결 정보
MYSQL_CONFIG = {
    "host": str(os.getenv('MYSQL_HOST')),
    "port": int(os.getenv('MYSQL_PORT')),
    "user": str(os.getenv('MYSQL_USER')),
    "password": str(os.getenv('MYSQL_PASSWORD')),
    "database": str(os.getenv('MYSQL_DATABASE'))
}

def get_db_connection():
    """MySQL 연결 생성"""
    try:
        conn = pymysql.connect(**MYSQL_CONFIG, charset="utf8")
        logging.info("✅ Connected to MySQL successfully.")
        return conn
    except pymysql.MySQLError as e:
        logging.error(f"❌ MySQL connection failed: {e}")
        return None

def get_embedding(text: str) -> list:
    """입력된 텍스트를 벡터로 변환"""
    logging.info(f"🔍 Generating embedding for: {text}")
    return embedding_model.encode(text).tolist()

def get_total_lecture_count():
    """전체 강의 개수를 가져옴"""
    conn = get_db_connection()
    if not conn:
        return 0
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM `db`")
        total_count = cursor.fetchone()[0]
        logging.info(f"📊 Total lectures in database: {total_count}")
    except pymysql.MySQLError as e:
        logging.error(f"❌ Failed to fetch lecture count: {e}")
        total_count = 0
    finally:
        cursor.close()
        conn.close()
    return total_count

def get_lecture_info(lecture_name):
    """MySQL에서 강의 정보를 가져옴 (추가 필드 포함)"""
    conn = get_db_connection()
    if not conn:
        return None
    cursor = conn.cursor()
    try:
        query = """
          SELECT `학수번호`, `과목명`, `분반`, `이수구분`, `학점`, `강의 시간`, `강의 장소`, `교수명`
          FROM `db`
          WHERE `과목명` = %s
        """
        cursor.execute(query, (lecture_name,))
        result = cursor.fetchone()
        logging.info(f"📄 Lecture info retrieved for '{lecture_name}': {result}")
        if result:
            return {
                "course_number": result[0],
                "course_name": result[1],
                "section": result[2],
                "category": result[3],
                "credits": result[4],
                "lecture_time": result[5],
                "lecture_location": result[6],
                "professor_name": result[7]
            }
        else:
            return None
    except pymysql.MySQLError as e:
        logging.error(f"❌ Failed to fetch lecture info for '{lecture_name}': {e}")
        return None
    finally:
        cursor.close()
        conn.close()

@app.route("/", methods=["GET", "POST"])
def index():
    """관심사 입력 페이지"""
    if request.method == "POST":
        keyword = request.form["keyword"]
        logging.info(f"🔍 User input received: {keyword}")
        return search_lectures(keyword)
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_lectures(keyword=None):
    """관심사를 입력받아 강의 검색 후 TSNE 좌표를 추가한 결과를 반환하고, 이를 시각화 페이지로 전달"""
    if keyword is None:
        keyword = request.form["keyword"]

    if not keyword:
        logging.warning("⚠️ No keyword provided!")
        return render_template("index.html", error="키워드를 입력하세요.")

    logging.info(f"🔍 Searching lectures for keyword: {keyword}")

    # ✅ 입력 키워드 임베딩 변환
    query_vector = np.array(get_embedding(keyword))
    logging.info(f"Query vector (first 5 values): {query_vector[:5]}")

    # ✅ ChromaDB에서 유사한 강의 검색 (n_results=10)
    logging.info("🔎 Querying ChromaDB for similar lectures...")
    try:
        results = lecture_vector_db.query(
            query_embeddings=[query_vector.tolist()],
            n_results=100,
            include=["embeddings", "metadatas", "distances"]
        )
    except Exception as e:
        logging.error(f"❌ Error querying ChromaDB: {e}")
        return render_template("index.html", error="ChromaDB 검색 중 오류가 발생했습니다.")

    if not results or "metadatas" not in results or not results["metadatas"]:
        logging.warning("⚠️ No matching lectures found in ChromaDB.")
        return render_template("search.html", keyword=keyword, lectures=[])

    # 내부 리스트 추출 (각각 10개의 결과)
    embeddings_list = results["embeddings"][0] if isinstance(results["embeddings"], list) and len(results["embeddings"]) > 0 else []
    metadatas_list = results["metadatas"][0] if isinstance(results["metadatas"], list) and len(results["metadatas"]) > 0 else []

    # TSNE를 위해, 키워드 임베딩과 강의 임베딩들을 결합 (n+1개, 첫 번째는 키워드)
    all_vectors = np.vstack([query_vector, np.array(embeddings_list)])
    logging.info(f"TSNE input shape: {all_vectors.shape}")

    tsne = TSNE(n_components=2, perplexity=min(30, len(all_vectors)-1), random_state=42)
    tsne_results = tsne.fit_transform(all_vectors)
    logging.info("t-SNE transformation completed.")

    # t-SNE 결과: 첫번째는 키워드, 나머지는 강의
    keyword_coord = tsne_results[0]
    lecture_coords = tsne_results[1:]
    # 중심을 키워드로 옮기기 위해 전체 좌표를 shift
    shift = keyword_coord.copy()
    keyword_coord = np.array([0, 0])
    lecture_coords = lecture_coords - shift  # 모든 강의 좌표 이동

    total_lectures = get_total_lecture_count()

    lecture_list = []
    for i in range(len(embeddings_list)):
        emb_result = np.array(embeddings_list[i])
        norm_query = np.linalg.norm(query_vector)
        norm_emb = np.linalg.norm(emb_result)
        if norm_query == 0 or norm_emb == 0:
            cosine_similarity = 0
        else:
            cosine_similarity = np.dot(query_vector, emb_result) / (norm_query * norm_emb)
        logging.info(f"Result {i+1}: Raw norm: query={norm_query:.4f}, result={norm_emb:.4f}, Cosine similarity = {cosine_similarity:.4f}")

        metadata = metadatas_list[i] if i < len(metadatas_list) else {}
        if isinstance(metadata, dict):
            lecture_names = [metadata.get("lecture_name", "Unknown")]
            logging.info(f"Result {i+1}: Detected single lecture name: {lecture_names}")
        elif isinstance(metadata, list):
            lecture_names = [item.get("lecture_name", "Unknown") for item in metadata if isinstance(item, dict)]
            logging.info(f"Result {i+1}: Detected multiple lecture names: {lecture_names}")
        else:
            lecture_names = ["Unknown"]
            logging.warning(f"Result {i+1}: Metadata format unrecognized: {metadata}")

        # TSNE 좌표
        dim1 = float(lecture_coords[i][0])
        dim2 = float(lecture_coords[i][1])

        for lecture_name in lecture_names:
            if not isinstance(lecture_name, str):
                logging.warning(f"⚠️ Warning: lecture_name is not a string! Metadata: {metadata}")
                continue

            logging.info(f"Result {i+1}: Processing lecture '{lecture_name}' with cosine similarity {cosine_similarity:.4f} and TSNE coords ({dim1:.4f}, {dim2:.4f})")
            lecture_info = get_lecture_info(lecture_name)
            if lecture_info:
                # lecture_info dictionary에 TSNE 좌표와 임베딩 점수를 추가
                lecture_info.update({"dim1": dim1, "dim2": dim2, "score": round(cosine_similarity, 4)})
                lecture_list.append(lecture_info)
            else:
                logging.warning(f"⚠️ No lecture info found in MySQL for lecture_name: {lecture_name}")

    lecture_list.sort(key=lambda x: x["score"], reverse=True)
    logging.info(f"✅ Found {len(lecture_list)} relevant lectures.")

    # TSNE에서 계산된 키워드 좌표도 전달 (중심에 위치)
    return render_template("search.html", keyword=keyword, lectures=lecture_list,
                           keyword_x=keyword_coord[0], keyword_y=keyword_coord[1])

if __name__ == "__main__":
    logging.info("🚀 Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
