import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# NIKKEI 225는 Yahoo Finance에서 수집
# tic = "^N225"
# start = "2019-01-01"
# end = "2025-07-01"

# df = yf.download(tic, start=start, end=end, interval="1d", 
#                  auto_adjust=False, progress=False)
# df.columns = df.columns.droplevel("Ticker")
# df = df.dropna(subset=["Close"]).copy()
# df.to_csv("NIKKEI225_daily.csv", encoding="utf-8-sig")


# 데이터 출처 : Investing.com
ks200 = pd.read_csv('../stock_data/benchmark_index/KOSPI 200_index.csv')
spx = pd.read_csv('../stock_data/benchmark_index/S&P 500_index.csv')
n225 = pd.read_csv('../stock_data/benchmark_index/NIKKEI 225_index.csv')
csi300 = pd.read_csv('../stock_data/benchmark_index/CSI 300_index.csv')


def clean_numeric(s):
    # %, B 같은 문자가 들어간 열은 사용 전 선택적으로 처리
    return (s.astype(str)
              .str.replace(',', '', regex=False)
              .str.replace('%', '', regex=False)
              .str.replace('B', '', regex=False)
              .str.strip())

def prep(df, col='종가'):
    # 공백 섞인 날짜 포맷 정리
    df['날짜'] = (df['날짜'].astype(str)
                  .str.replace(' ', '', regex=False))  # "2025- 06- 01" -> "2025-06-01" [attached_file:84]
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    # 숫자 변환
    df[col] = pd.to_numeric(clean_numeric(df[col]), errors='coerce')
    # 정렬 및 인덱스 설정
    df = df.sort_values('날짜').set_index('날짜')[[col]].rename(columns={col: 'close'})
    # 결측 제거
    return df.dropna()


ks200 = prep(ks200)
spx = prep(spx)
n225 = prep(n225)
csi300 = prep(csi300)

df = pd.concat([
    ks200.rename(columns={'close':'KOSPI 200'}),
    spx.rename(columns={'close':'S&P 500'}),
    n225.rename(columns={'close':'Nikkei 225'}),
    csi300.rename(columns={'close':'CSI 300'})
], axis=1).dropna()


# 시각화
plt.ion()

# x축과 y축 범위 동일
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()

series = {
    'KOSPI 200': df['KOSPI 200'],
    'S&P 500': df['S&P 500'],
    'Nikkei 225': df['Nikkei 225'],
    'CSI 300': df['CSI 300']
}

for ax, (name, s) in zip(axes, series.items()):
    ax.plot(s.index, s.values, color='tab:blue')
    # ax.set_title(name)
    # ax.grid(True, alpha=0.3)
    # ax.legend(loc='upper left')


labels = ["(a)", "(b)", "(c)", "(d)"]
for ax, lab in zip(axes, labels):
    # 축 좌표계(왼아래=0,0 ~ 오른위=1,1) 기준으로 좌하단에 배치
    ax.text(0.5, -0.13, lab, transform=ax.transAxes,
            ha="center", va="top", fontsize=20)

# 격자 제거 + 박스형 해제(위/오른쪽 스파인 숨기기)
for ax in axes:
    ax.grid(False)  # 배경 격자 off
    ax.spines[['top','right']].set_visible(False)  # 상자형 방지

# 공통 레이블과 폰트 크기
for ax in axes:
    ax.tick_params(labelsize=14)    # 눈금 라벨 크기

fig.tight_layout()
fig.subplots_adjust(hspace=0.3)    # 서브플롯 간격(상하)
fig.subplots_adjust(wspace=0.2)    # 서브플롯 간격(좌우)

plt.show()