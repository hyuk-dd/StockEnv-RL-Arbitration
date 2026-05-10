import pymysql
import pandas as pd

# ==========================================
# 0. DB 접속 정보 (비밀번호 확인 필수)
# ==========================================
DB_CONFIG = {
    # 개인 DB 접속 정보 입력
    # "host": "",
    # "user": "",
    # "password": "",  # <--- 본인 비밀번호 입력
    # "db": "",
    # "charset": ""
}

# 1. DB 접속
conn = pymysql.connect(**DB_CONFIG)

# 2. 커서 생성 (DB에 명령을 전달하는 트럭)
cursor = conn.cursor()

# 3. SQL 실행 (예시)
sql = "SELECT * FROM Ticker WHERE TickerName LIKE '%삼성전자%'"

# 4. 커서 실행
cursor.execute(sql)

# 5. 결과 확인
# 결과 하나만 가져오기 (fetchone)
# 결과 모두 가져오기 (fetchall)
# 파이썬에서 보기 좋게 출력하기 (pd.read_sql(sql, conn))
# print(cursor.fetchone())
# print(cursor.fetchall())
print(pd.read_sql(sql, conn))

# 6. 접속 종료
# conn.close()

# 005930 코드 정보 찾기
target_code = "005930" 
sql_find_id = "SELECT TickerID, TickerName, MarketID FROM Ticker WHERE TickerCode = %s"
cursor.execute(sql_find_id)
print(pd.read_sql(sql_find_id, conn, params=(target_code,)))

# Country 테이블 정보
sql = "select * from country"
cursor.execute(sql)
print(pd.read_sql(sql, conn))

# Market 테이블 정보
sql = "select * from Market"
cursor.execute(sql)
print(pd.read_sql(sql, conn))

# Ticker 테이블 정보
sql = "select * from Ticker"
cursor.execute(sql)
print(pd.read_sql(sql, conn))

# DailyPrice 테이블 정보
sql = "select * from DailyPrice limit 10"
cursor.execute(sql)
print(pd.read_sql(sql, conn))

# IndexInfo 테이블 정보
sql = "select * from IndexInfo"
cursor.execute(sql)
print(pd.read_sql(sql, conn))

# IndexPrice 테이블 정보
sql = "select * from IndexPrice"
cursor.execute(sql)
print(pd.read_sql(sql, conn))

######################
# 시장 정보와 티커 정보 join
sql = "select * from market m join ticker t on m.MarketID = t.MarketID"
cursor.execute(sql)
print(pd.read_sql(sql, conn))

# 
sql = "select * from indexinfo f join indexprice p on f.IndexID = p.IndexID"
cursor.execute(sql)
df = pd.read_sql(sql, conn)
print(df.loc[df.IndexName == 'NIKKEI 225', 'Close'].describe())

conn.close()