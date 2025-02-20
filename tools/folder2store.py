import os
import numpy as np
import pandas as pd
import chromadb
import nltk
from collections import Counter
from soynlp.noun import LRNounExtractor_v2
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import traceback
from ollama import Client
import ast

def get_embedding(text: str) -> list:
    """텍스트를 SBERT를 이용해 임베딩 벡터로 변환"""
    embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")  # ✅ 함수 내부에서 모델 정의
    return embedding_model.encode(text).tolist()  # ChromaDB는 list 형태만 허용

def keywords_on_text(noun_list: list, lecture_name: str, client: Client, model_name: str = "llama3.1:8b") -> list:
    """
    Ollama LLM을 활용하여 복합 명사 중 독립적으로 강의를 설명할 수 있는 단어만 필터링하는 함수
    """
    SYS_PROMPT = (
        "당신은 주어진 강의명과 키워드 리스트를 분석하여,"
        "독립적으로 강의 내용을 명확히 설명할 수 있는 핵심 키워드만을 추출하는 역할을 맡고 있습니다.\n\n"
        "규칙:\n"
        "1) 키워드는 강의명과 관련이 있어야 합니다.\n"
        "2) 키워드는 독립적으로 보았을 때도 의미가 명확해야 합니다.\n"
        "   (예: '문제 풀이'는 모호하지만 '수학 문제 풀이'는 명확함)\n"
        "3) 반드시 파이썬 리스트 형태로 출력하세요,\n"
        "   반드시 응답에 추가 설명이나 문장은 포함하지 마세요.\n"
        "4) 중복된 키워드는 제거하고, 중요한 키워드만 단일 리스트로 제공합니다.\n\n"
        "출력 예시:\n"
        "['회귀 분석', '확률 통계', '데이터 마이닝']"
    )

    USER_PROMPT = f"강의명: {lecture_name}\n복합 명사 리스트: {noun_list}\n출력: "
    response = client.generate(
        model=model_name,
        system=SYS_PROMPT,
        prompt=USER_PROMPT
    )
    
    # 응답을 리스트로 변환
    raw_result = response.response
    try:
        keywords_list = ast.literal_eval(raw_result)
        return keywords_list
    except (SyntaxError, ValueError):
        print(f"⚠️ LLM 응답 변환 실패! 빈 리스트 반환함.")
        return []


def excel2df(file_path: str, client: Client) -> tuple: 
    '''
    엑셀 파일에서 강의명, 강의 개요, 요구 지식, 강의 계획을 추출하여 딕셔너리 형태로 반환
    '''
    df = pd.DataFrame(pd.read_excel(file_path, header=None))
    name, abstract, required_knowledge, target_knowledge = '', '', '', ''
    plan = []

    # 강의 개요 추출
    abstract_r_index = df.iloc[:, 0].str.contains('강의개요', na=False).idxmax()
    abstract = df.iloc[abstract_r_index + 1, 0]
    
    # 강의명 추출
    name_r_index = df.iloc[:, 18].str.contains('교과목명', na=False).idxmax()
    name = df.iloc[name_r_index, 24]

    # 선수 과목 추출
    required_knowledge_r_index = df.iloc[:, 0].str.contains('선수과목', na=False).idxmax()
    required_knowledge = df.iloc[required_knowledge_r_index + 1, 0]
    required_knowledge = '없음' if pd.isna(required_knowledge) else required_knowledge
    
    # 강의 목표 추출
    target_knowledge_r_index = df.iloc[:, 0].str.contains('강의목표', na=False).idxmax()
    target_knowledge = df.iloc[target_knowledge_r_index + 1, 0]

    # 강의 계획 추출
    plan_startpoint_r_index = df.iloc[:, 0].str.contains('주별', na=False).idxmax()
    for _ in range(plan_startpoint_r_index+1, plan_startpoint_r_index+58, 4):
        plan.append({
            'theme': df.iloc[_ , 10] if not pd.isna(df.iloc[_, 10]) else '없음',
            'goal': df.iloc[_ + 1, 10] if not pd.isna(df.iloc[_+1, 10]) else '없음',
            'keyword': df.iloc[_ + 2, 10] if not pd.isna(df.iloc[_+2, 10]) else '없음',
            'approach': df.iloc[_+3, 10] if not pd.isna(df.iloc[_+3, 10]) else '없음'
        })

    # DataFrame 생성
    plan_df = pd.DataFrame(plan)
    plan_df['name'] = name
    plan_df['abstract'] = abstract
    plan_df['required_knowledge'] = required_knowledge
    plan_df['target_knowledge'] = target_knowledge

    # 모든 텍스트를 하나의 리스트로 모으기
    text_data = []
    for col in plan_df.columns:
        for text in plan_df[col].dropna():  # NaN 값 제외
            if isinstance(text, str):  # 문자열인 경우만 처리
                sentences = nltk.sent_tokenize(text)  # 문장 분리
                text_data.extend(sentences)

    # 복합 명사 추출
    noun_extractor = LRNounExtractor_v2()
    nouns = noun_extractor.train_extract(text_data)
    compound_nouns = [noun for noun in nouns.keys() if len(noun) > 1]

    # ✅ LLM을 사용하여 필터링
    filtered_nouns = keywords_on_text(compound_nouns, name, client)

    # 명사 빈도수 계산
    noun_counter = Counter()
    for sentence in text_data:
        for noun in filtered_nouns:
            if noun in sentence:
                noun_counter[noun] += 1

    # 결과 데이터프레임 생성 (명사 및 빈도수)
    noun_freq_df = pd.DataFrame(noun_counter.items(), columns=['noun', 'frequency']).sort_values(by='frequency', ascending=False)

    return name, noun_freq_df

def noun_exists_in_db(noun: str) -> bool:
    """특정 명사가 이미 벡터 데이터베이스에 존재하는지 확인"""
    results = noun_vector_db.get(where={"noun": noun}, include=["embeddings", "metadatas"])
    return len(results["ids"]) > 0  # 명사가 존재하면 True 반환

def store_noun_embeddings(noun_df):
    """복합 명사 임베딩을 벡터 데이터베이스에 저장 (중복 방지)"""
    for _, row in noun_df.iterrows():
        noun = row["noun"]
        
        # ✅ 중복 확인
        existing_data = noun_vector_db.get(include=["embeddings"], where={"noun": noun})
        if existing_data["ids"]:
            print(f"⏩ 명사 '{noun}' 이미 존재하여 추가하지 않음.")
            continue

        # ✅ 벡터 생성
        vector = get_embedding(noun) # NumPy 배열로 변환

        # ✅ 데이터베이스 저장
        try:
            noun_vector_db.add(
                ids=[noun], embeddings=[vector],  # ✅ 리스트로 변환 후 저장
                metadatas=[{"noun": noun, "frequency": row["frequency"]}]
            )

            # ✅ 저장 후 확인
            check_data = noun_vector_db.get(include=["embeddings"], where={"noun": noun})
            if not check_data["ids"]:
                print(f"⚠️ 명사 '{noun}' 저장 실패! ChromaDB에 저장되지 않음.")
            else:
                print(f"✅ 명사 '{noun}' 저장 완료! 벡터 크기: {len(vector)}")

        except Exception as e:
            print(f"⚠️ 명사 '{noun}' 저장 중 오류 발생: {e}")

def lecture_exists_in_db(lecture_name: str) -> bool:
    """강의명이 이미 벡터 데이터베이스에 존재하는지 확인"""
    results = lecture_vector_db.get(where={"lecture_name": lecture_name}, include=["embeddings", "metadatas"])
    return len(results["ids"]) > 0  # 강의명이 존재하면 True 반환

def store_lecture_embeddings(lecture_name, noun_df):
    """강의 평균 임베딩을 벡터 데이터베이스에 저장"""
    
    # ✅ 모든 명사의 임베딩을 NumPy 배열로 변환
    embeddings = [get_embedding(noun) for noun in noun_df["noun"]]
    
    # ✅ 평균 벡터 계산
    lecture_vector = np.mean(embeddings, axis=0)  # ✅ NumPy 배열 형태로 유지

    try: 
        lecture_vector_db.add(
            ids=[lecture_name],
            embeddings=[lecture_vector],  # ✅ 리스트 변환 후 저장
            metadatas=[{"lecture_name": lecture_name}]
        )

        # ✅ 저장 후 확인 (include=["embeddings"] 추가)
        check_data = lecture_vector_db.get(where={"lecture_name": lecture_name}, include=["embeddings", "metadatas"])
        
        if not check_data["ids"]:
            print(f"⚠️ 강의 '{lecture_name}' 저장 실패! ChromaDB에 저장되지 않음.")
        else:
            print(f"✅ 강의 '{lecture_name}' 평균 벡터 저장 완료! 벡터 크기: {len(lecture_vector)}")

    except Exception as e:
        print(f"⚠️ 강의 '{lecture_name}' 저장 중 오류 발생: {e}")



def process_all_excel_files(folder_path, client: Client):
    """폴더 내 모든 엑셀 파일을 처리하고, 중복 강의명을 제외한 데이터를 벡터 DB에 저장"""
    nltk.download('punkt_tab')

    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            file_path = os.path.join(folder_path, file)
            
            try:
                lecture_name, noun_df = excel2df(file_path, client)
                
                if lecture_exists_in_db(lecture_name):
                    print(f"✅ 강의 '{lecture_name}'는 이미 존재함, 스킵합니다.")
                    continue

                print(f"📌 강의 '{lecture_name}' 처리 중...")

                # ✅ 중복 방지하여 명사 벡터 DB 저장
                store_noun_embeddings(noun_df)

                # ✅ 강의 평균 벡터 DB 저장
                store_lecture_embeddings(lecture_name, noun_df)

                print(f"✅ 강의 '{lecture_name}' 벡터 저장 완료!\n")

            except Exception as e:
                print(f"⚠️ 파일 {file} 처리 중 오류 발생: {e}")
                traceback.print_exc()  # ✅ 오류 원인 출력


if __name__ == '__main__':
    
    # ✅ 1. ChromaDB 클라이언트 초기화
    chroma_client = chromadb.PersistentClient(path="chroma_db")  

    # ✅ 2. 벡터 데이터베이스 컬렉션 생성
    noun_vector_db = chroma_client.get_or_create_collection("noun_embeddings")   # 복합 명사 벡터 DB
    lecture_vector_db = chroma_client.get_or_create_collection("lecture_embeddings")  # 강의 평균 벡터 DB

    folder_path = "D:/CDM/Workspace/AcaTuning/user_input/ver2"
    
    # ✅ Ollama 클라이언트 생성 후 전달
    ollama_client = Client()
    process_all_excel_files(folder_path, ollama_client)
    