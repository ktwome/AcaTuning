import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
import chromadb

def visualize_noun_embeddings_tsne(chroma_db_path, collection_name="noun_embeddings", perplexity=30):
    """
    ChromaDB에서 명사 임베딩을 가져와 t-SNE를 이용해 시각화하는 함수.

    Args:
    - chroma_db_path (str): ChromaDB 저장 경로
    - collection_name (str): 사용할 컬렉션 이름 (기본값: "noun_embeddings")
    - perplexity (int): t-SNE의 perplexity 값 (기본값 30)

    Returns:
    - Plotly interactive scatter plot
    """

    # ✅ ChromaDB 클라이언트 초기화
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    noun_vector_db = chroma_client.get_or_create_collection(collection_name)

    # ✅ ChromaDB에서 명사 벡터 가져오기 (include 옵션 추가)
    results = noun_vector_db.get(include=["embeddings", "metadatas"])

    # ✅ 데이터 개수 확인
    num_ids = len(results.get("ids", []))
    print(f"📌 ChromaDB에서 가져온 데이터 개수: {num_ids}")

    if num_ids == 0:
        print("⚠️ 저장된 명사 벡터가 없습니다! 시각화를 중단합니다.")
        return None

    # ✅ 명사와 임베딩 데이터 추출
    nouns = [meta["noun"] for meta in results["metadatas"] if "noun" in meta]
    vectors = results.get("embeddings", [])

    print(f"📌 총 {len(nouns)}개의 명사 임베딩을 시각화합니다.")
    print(f"📌 벡터 데이터 개수: {len(vectors)}")

    # ✅ 벡터 데이터 검증 및 변환 (NumPy 배열 변환)
    if not isinstance(vectors, list):
        print("⚠️ 벡터 데이터가 리스트가 아님. 리스트로 변환합니다.")
        vectors = np.array(vectors).tolist()

    if not vectors or any(not isinstance(vec, list) or len(vec) != len(vectors[0]) for vec in vectors):
        print("⚠️ 오류: 벡터 데이터가 비어 있거나 형식이 올바르지 않음. 시각화를 중단합니다.")
        return None

    if len(vectors) < 2:
        print("⚠️ 벡터 개수가 부족하여 시각화가 불가능합니다.")
        return None

    # ✅ NumPy 배열로 변환 (TSNE에서 요구하는 2D 형태)
    vectors_np = np.array(vectors, dtype=np.float32)

    # ✅ t-SNE 차원 축소 수행
    print("🚀 t-SNE를 이용하여 2차원으로 변환 중...")
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(vectors) - 1), random_state=42)
    reduced_vectors = tsne.fit_transform(vectors_np)  # NumPy 배열 유지

    # ✅ DataFrame 생성
    df_vis = pd.DataFrame(reduced_vectors, columns=["dim_1", "dim_2"])
    df_vis["noun"] = nouns  # 명사 추가

    # ✅ Plotly를 이용한 시각화
    fig = px.scatter(
        df_vis, x="dim_1", y="dim_2", text="noun",
        title="명사 임베딩 시각화 (t-SNE)",
        labels={"dim_1": "축소된 차원 1", "dim_2": "축소된 차원 2"},
        width=1920, height=1080
    )

    fig.update_traces(textposition='top center')  # 라벨 위치 조정
    fig.show()

# ✅ 실행 예제
chroma_db_path = "D:/CDM/Workspace/AcaTuning/chroma_db"  # ✅ ChromaDB 경로 지정
visualize_noun_embeddings_tsne(chroma_db_path)
