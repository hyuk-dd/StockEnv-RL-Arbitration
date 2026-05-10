-- ==============================================================================
-- 데이터베이스 스키마 정의
-- ==============================================================================
CREATE SCHEMA IF NOT EXISTS stock;
USE stock;

-- ------------------------------------------------------------------------------
-- 1. Country : 국가 코드와 이름 관리
-- ------------------------------------------------------------------------------
CREATE TABLE Country (
	CountryID 	INT AUTO_INCREMENT PRIMARY KEY,
	ISO2 		CHAR(2) UNIQUE,		-- 국가 코드 체계(ex. KR, US)
    ISO3 		CHAR(3) UNIQUE,		-- 국가 코드 체계(ex. KOR, USA)
	CountryName VARCHAR(100) UNIQUE NOT NULL
);

-- ------------------------------------------------------------------------------
-- 2. Market : 거래소/시장 (KOSPI, KOSDAQ, NYSE 등)
-- ------------------------------------------------------------------------------
CREATE TABLE Market (
	MarketID 	INT AUTO_INCREMENT PRIMARY KEY,
	MarketCode	VARCHAR(20) UNIQUE NOT NULL,	-- ex) KOSPI, NYSE
    CountryID	INT NOT NULL,
    FOREIGN KEY (CountryID) REFERENCES Country(CountryID) ON DELETE RESTRICT	-- 참조 중인 국가 행을 삭제하려고 하면 거부
);
-- 국가별 시장 목록 조회, 국가 -> 시장 조인 가속
CREATE INDEX ix_market_country ON Market (CountryID);

-- ------------------------------------------------------------------------------
-- 3. Ticker : 종목 코드(티커) 및 메타 정보
-- ------------------------------------------------------------------------------
CREATE TABLE Ticker (
	TickerID 	 INT AUTO_INCREMENT PRIMARY KEY,
	TickerCode 	 VARCHAR(32) NOT NULL,		-- 종목 코드 (ex. AAPL, 005930)
	TickerName 	 VARCHAR(200),				-- 종목명
	MarketID 	 INT NOT NULL,
    SecurityType ENUM('STOCK', 'ETF', 'INDEX') NOT NULL DEFAULT 'STOCK',
    -- 동일한 시장 내에서 코드 중복 불가
    UNIQUE KEY uk_market_ticker (MarketID, TickerCode),
	FOREIGN KEY (MarketID) REFERENCES Market(MarketID) ON DELETE RESTRICT	-- 참조 중인 시장 행을 삭제하려고 하면 거부
);
-- 시장별 종목 목록 조회, 시장 -> 종목 조인 가속
CREATE INDEX ix_ticker_market ON Ticker (MarketID);

-- ------------------------------------------------------------------------------
-- 4. IndexInfo : 대표 벤치마크 지수 메타 정보
-- ------------------------------------------------------------------------------
CREATE TABLE IndexInfo (
    IndexID   INT AUTO_INCREMENT PRIMARY KEY,
    IndexName VARCHAR(200) NOT NULL,	-- ex) KOSPI 200, S&P 500
    CountryID INT NOT NULL,
	FOREIGN KEY (CountryID) REFERENCES Country(CountryID) ON DELETE RESTRICT
);

-- ------------------------------------------------------------------------------
-- 5. DailyPrice : 종목별 일봉 가격 정보
-- ------------------------------------------------------------------------------
CREATE TABLE DailyPrice (
    TickerID INT NOT NULL,
    Date 	 DATE NOT NULL,
    Open 	 DECIMAL(19, 4),    	-- 부동 소수점
    High 	 DECIMAL(19, 4),
    Low  	 DECIMAL(19, 4),
    Close 	 DECIMAL(19, 4),
    AdjClose DECIMAL(19, 4),
    Volume 	 BIGINT,				-- 거래량은 매우 클 수 있으므로 BIGINT로 구성
    PRIMARY KEY (TickerID, Date), 	-- 복합 기본키로 두어 중복 원천 차단 / 조회 시 TikcerID -> Date 순으로 조회 가능)
	FOREIGN KEY (TickerID) REFERENCES Ticker(TickerID) ON DELETE CASCADE,	-- 같이 삭제 되도록
    -- 데이터 무결성 체크 (고가, 저가 논리 검증)
    CHECK (
        High IS NULL OR Low IS NULL OR
        (High >= COALESCE(GREATEST(Open, Close), High)		-- 시가/종가 중 큰 값이 NULL이면 High를 대신 써라
        AND Low  <= COALESCE(LEAST(Open, Close), Low))		-- 시가/종가 중 작은 값이 NULL이면 Low를 대신 써라
    )
);
-- 최근 N일 가격을 역정렬로 빠르게 가져오기 위한 복합 인덱스 (LIMIT/OFFSET 최적화)
CREATE INDEX ix_dailyprice_date_desc ON DailyPrice (TickerID, Date DESC);

-- ------------------------------------------------------------------------------
-- 6. IndexPrice : 벤치마크 지수의 일별 가격 정보
-- ------------------------------------------------------------------------------
CREATE TABLE IndexPrice (
    IndexID INT NOT NULL,
    Date 	DATE NOT NULL,
    Open 	DECIMAL(19, 4),
    High 	DECIMAL(19, 4),
    Low  	DECIMAL(19, 4),
    Close 	DECIMAL(19, 4),
    Volume 	BIGINT,
    PRIMARY KEY (IndexID, Date),
	FOREIGN KEY (IndexID) REFERENCES IndexInfo(IndexID) ON DELETE CASCADE
);
