import pandas as pd
import pymysql

# ✅ MySQL 연결 정보
MYSQL_CONFIG = {
    "host": "your_mysql_host",
    "user": "your_mysql_user",
    "password": "your_mysql_password",
    "database": "lecture_db"
}

# ✅ CSV 파일 로드
file_path = "/mnt/data/강의데이터.csv"
df = pd.read_csv(file_path, encoding="euc-kr")  # 인코딩 확인 필요

# ✅ MySQL에 연결
conn = pymysql.connect(**MYSQL_CONFIG)
cursor = conn.cursor()

# ✅ 데이터 삽입
for _, row in df.iterrows():
    sql = """
    INSERT INTO lectures (course_number, course_name, section, category, credits, lecture_hours, practice_hours, professor_name, lecture_time, lecture_location)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(sql, (
        row["학수번호"], row["과목명"], row["분반"], row["이수구분"], row["학점"],
        row["이론 강의 시수"], row["실습 강의 시수"], row["교수명"], row["강의 시간"], row["강의 장소"]
    ))

# ✅ 변경 사항 저장
conn.commit()
cursor.close()
conn.close()

print("✅ MySQL 데이터 삽입 완료!")
