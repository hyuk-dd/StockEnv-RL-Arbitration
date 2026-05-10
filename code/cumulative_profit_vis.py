import pandas as pd
import matplotlib.pyplot as plt


# Arbitration
def arb_country_cumulative_profit(country):
    df = pd.read_csv(f"../results/{country}/asset3.csv", 
                         header=None, names=["arb_account"])
    # 첫 번째 행 계좌 값이 0이면 첫 번째 행 삭제
    if not df.empty and pd.to_numeric(df.loc[0, "arb_account"], errors="coerce") == 0:
        df = df.drop(index=0).reset_index(drop=True)

    # 누적 수익률 (%)
    start_account = df["arb_account"].iloc[0]
    # df["cum_profit_abs"] = df["account"] - start_account 
    df["arb_cum_profit"] = (df["arb_account"] / start_account - 1) * 100

    # 전일 대비 수익률 (%)
    df["arb_ret"] = df["arb_account"].pct_change() * 100

    return df

# Base
def base_country_cumulative_profit(country):
    df = pd.read_csv(f"../results/{country}/asset_base.csv", 
                         header=None, names=["base_account"])
    # 첫 번째 행 계좌 값이 0이면 첫 번째 행 삭제
    if not df.empty and pd.to_numeric(df.loc[0, "base_account"], errors="coerce") == 0:
        df = df.drop(index=0).reset_index(drop=True)

    # 누적 수익률 (%)
    start_account = df["base_account"].iloc[0]
    df["base_cum_profit"] = (df["base_account"] / start_account - 1) * 100

    # 전일 대비 수익률 (%)
    df["base_ret"] = df["base_account"].pct_change() * 100

    return df

# Merged
def country_cumulative_profit(country):
    df_arb = arb_country_cumulative_profit(country)
    df_base = base_country_cumulative_profit(country)
    df = pd.concat([df_arb, df_base], axis=1)

    return df


korea_df = country_cumulative_profit("korea")
usa_df = country_cumulative_profit("usa")
japan_df = country_cumulative_profit("japan")
china_df = country_cumulative_profit("china")

# 벤치마크 지수
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
              .str.replace('M', '', regex=False)
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

ks200['거래량'] = pd.to_numeric(clean_numeric(ks200['거래량']), errors='coerce')
ks200['변동 %'] = pd.to_numeric(clean_numeric(ks200['변동 %']), errors='coerce')
ks200.describe()

spx['거래량'] = pd.to_numeric(clean_numeric(spx['거래량']), errors='coerce')
spx['변동 %'] = pd.to_numeric(clean_numeric(spx['변동 %']), errors='coerce')
spx.describe()

def trim_and_cum(cum_df, benchmark_df):
    if len(benchmark_df) > len(cum_df):
        drop_n = len(benchmark_df) - len(cum_df)
        benchmark_df = benchmark_df.iloc[drop_n:]
    
    benchmark_df['cumulative'] = (benchmark_df['close'] / benchmark_df['close'].iloc[0] - 1.0) * 100.0
    return benchmark_df

ks200 = trim_and_cum(korea_df, ks200)
spx = trim_and_cum(usa_df, spx)
n225 = trim_and_cum(japan_df, n225)
csi300 = trim_and_cum(china_df, csi300)



plt.ion()
# 누적 수익률 시각화
# x축과 y축 범위 동일
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

for ax in axes.ravel():
    ax.xaxis.set_tick_params(labelbottom=True)  # 위쪽 서브플롯의 x라벨 표시
    ax.yaxis.set_tick_params(labelleft=True)    # 오른쪽 서브플롯의 y라벨 표시

# 각 서브플롯에 데이터 그리기
axes[0,0].plot(korea_df.index, korea_df["base_cum_profit"], 
               color="#1f77b4", lw=1, label="Base")
axes[0,0].plot(korea_df.index, korea_df["arb_cum_profit"],  
               color="#ff7f0e", lw=1, label="Arbitration")
axes[0,0].plot(korea_df.index, ks200['cumulative'], color="#9e9e9e", lw=1.4, linestyle=(0, (9, 8)), alpha=0.8)
# axes[0,0].set_title("Korea", fontsize=22)

axes[0,1].plot(usa_df.index, usa_df["base_cum_profit"], 
               color="#1f77b4", lw=1, label="Base")
axes[0,1].plot(usa_df.index, usa_df["arb_cum_profit"],  
               color="#ff7f0e", lw=1, label="Arbitration")
axes[0,1].plot(usa_df.index, spx['cumulative'], color="#9e9e9e", lw=1.4, linestyle=(0, (9, 8)), alpha=0.8)
# axes[0,1].set_title("USA", fontsize=22)

axes[1,0].plot(japan_df.index, japan_df["base_cum_profit"], 
               color="#1f77b4", lw=1, label="Base")
axes[1,0].plot(japan_df.index, japan_df["arb_cum_profit"],  
               color="#ff7f0e", lw=1, label="Arbitration")
axes[1,0].plot(japan_df.index, n225['cumulative'], color="#9e9e9e", lw=1.4, linestyle=(0, (9, 8)), alpha=0.8)
# axes[1,0].set_title("Japan", fontsize=22)

axes[1,1].plot(china_df.index, china_df["base_cum_profit"], 
               color="#1f77b4", lw=1, label="Base")
axes[1,1].plot(china_df.index, china_df["arb_cum_profit"],  
               color="#ff7f0e", lw=1, label="Arbitration")
axes[1,1].plot(china_df.index, csi300['cumulative'], color="#9e9e9e", lw=1.4, linestyle=(0, (9, 8)), alpha=0.8)
# axes[1,1].set_title("China", fontsize=22)

labels = ["(a)", "(b)", "(c)", "(d)"]
for ax, lab in zip(axes.ravel(), labels):
    # 축 좌표계(왼아래=0,0 ~ 오른위=1,1) 기준으로 좌하단에 배치
    ax.text(0.5, -0.13, lab, transform=ax.transAxes,
            ha="center", va="top", fontsize=20)

# 격자 제거 + 박스형 해제(위/오른쪽 스파인 숨기기)
for ax in axes.ravel():
    ax.grid(False)  # 배경 격자 off
    ax.spines[['top','right']].set_visible(False)  # 상자형 방지

# 공통 레이블과 폰트 크기
for ax in axes.ravel():
    # ax.set_xlabel("Time-step", fontsize=20)
    # ax.set_ylabel("Cumulative-reward", fontsize=20)
    ax.tick_params(labelsize=14)    # 눈금 라벨 크기

# # 서브플롯별 개별 범례는 표시하지 않고, 핸들/라벨을 모아 하나의 범례 생성
# handles, labels = [], []
# for ax in axes.ravel():
#     h, l = ax.get_legend_handles_labels()
#     handles += h
#     labels  += l

# # 중복 라벨 제거(선택)
# uniq = dict(zip(labels, handles))
# # 통합 범례 [상단]
# fig.legend(uniq.values(), uniq.keys(), 
#            loc="upper right",
#            bbox_to_anchor=(1, 0.93),
#            ncol=2, fontsize=14,
#            borderaxespad=0.2
# )

# fig.suptitle("Cumulative Return (Base vs Arb)", fontsize=24)  # 전체 제목
# fig.tight_layout(rect=[0, 0, 1, 0.94])  # 서브플롯 간격 자동 조정하되 rect로 상단 범례/제목 자리 확보
fig.tight_layout()
fig.subplots_adjust(hspace=0.3)    # 서브플롯 간격(상하)
fig.subplots_adjust(wspace=0.2)    # 서브플롯 간격(좌우)

plt.show()




##### 

import pandas as pd
import matplotlib.pyplot as plt


# Arbitration
def arb_country_cumulative_profit(country):
    df = pd.read_csv(f"../results/{country}/asset3.csv", 
                         header=None, names=["arb_account"])
    # 첫 번째 행 계좌 값이 0이면 첫 번째 행 삭제
    if not df.empty and pd.to_numeric(df.loc[0, "arb_account"], errors="coerce") == 0:
        df = df.drop(index=0).reset_index(drop=True)

    # 누적 수익률 (%)
    start_account = df["arb_account"].iloc[0]
    # df["cum_profit_abs"] = df["account"] - start_account 
    df["arb_cum_profit"] = (df["arb_account"] / start_account - 1) * 100

    # 전일 대비 수익률 (%)
    df["arb_ret"] = df["arb_account"].pct_change() * 100

    return df

# Base
def base_country_cumulative_profit(country):
    df = pd.read_csv(f"../results/{country}/asset_base.csv", 
                         header=None, names=["base_account"])
    # 첫 번째 행 계좌 값이 0이면 첫 번째 행 삭제
    if not df.empty and pd.to_numeric(df.loc[0, "base_account"], errors="coerce") == 0:
        df = df.drop(index=0).reset_index(drop=True)

    # 누적 수익률 (%)
    start_account = df["base_account"].iloc[0]
    df["base_cum_profit"] = (df["base_account"] / start_account - 1) * 100

    # 전일 대비 수익률 (%)
    df["base_ret"] = df["base_account"].pct_change() * 100

    return df

# Merged
def country_cumulative_profit(country):
    df_arb = arb_country_cumulative_profit(country)
    df_base = base_country_cumulative_profit(country)
    df = pd.concat([df_arb, df_base], axis=1)

    return df


korea_df = country_cumulative_profit("korea")
usa_df = country_cumulative_profit("usa")
japan_df = country_cumulative_profit("japan")
china_df = country_cumulative_profit("china")

# 벤치마크 지수
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
              .str.replace('M', '', regex=False)
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


def trim_and_cum(cum_df, benchmark_df):
    if len(benchmark_df) > len(cum_df):
        drop_n = len(benchmark_df) - len(cum_df)
        benchmark_df = benchmark_df.iloc[drop_n:]
    
    benchmark_df['cumulative'] = (benchmark_df['close'] / benchmark_df['close'].iloc[0] - 1.0) * 100.0
    return benchmark_df

ks200 = trim_and_cum(korea_df, ks200)
spx = trim_and_cum(usa_df, spx)
n225 = trim_and_cum(japan_df, n225)
csi300 = trim_and_cum(china_df, csi300)



plt.ion()
# 누적 수익률 시각화
# x축과 y축 범위 동일
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

for ax in axes.ravel():
    ax.xaxis.set_tick_params(labelbottom=True)  # 위쪽 서브플롯의 x라벨 표시
    ax.yaxis.set_tick_params(labelleft=True)    # 오른쪽 서브플롯의 y라벨 표시

# 보조축 생성: 서브플롯마다 ax.twinx()
ax_kor_r = axes[0,0].twinx()
ax_usa_r = axes[0,1].twinx()    
ax_jpn_r = axes[1,0].twinx()
ax_chn_r = axes[1,1].twinx()


# 각 서브플롯에 데이터 그리기
axes[0,0].plot(korea_df.index, korea_df["base_cum_profit"], 
               color="#1f77b4", lw=1, label="Base")
axes[0,0].plot(korea_df.index, korea_df["arb_cum_profit"],  
               color="#ff7f0e", lw=1, label="Arbitration")
ax_kor_r.fill_between(korea_df.index, ks200["close"], ks200["close"].min(), color="#8f8f8f", alpha=0.14, zorder=0)
ax_kor_r.plot(korea_df.index, ks200["close"], color="#b0b0b0", lw=1.0, linestyle=(0,(10,7)), alpha=0.6, zorder=1)
ax_kor_r.tick_params(colors="#9a9a9a")


axes[0,1].plot(usa_df.index, usa_df["base_cum_profit"], 
               color="#1f77b4", lw=1, label="Base")
axes[0,1].plot(usa_df.index, usa_df["arb_cum_profit"],  
               color="#ff7f0e", lw=1, label="Arbitration")
ax_usa_r.fill_between(usa_df.index, spx["close"], spx["close"].min(), color="#8f8f8f", alpha=0.14, zorder=0)
ax_usa_r.plot(usa_df.index, spx["close"], color="#b0b0b0", lw=1.0, linestyle=(0,(10,7)), alpha=0.6, zorder=1)
ax_usa_r.tick_params(colors="#9a9a9a")

axes[1,0].plot(japan_df.index, japan_df["base_cum_profit"], 
               color="#1f77b4", lw=1, label="Base")
axes[1,0].plot(japan_df.index, japan_df["arb_cum_profit"],  
               color="#ff7f0e", lw=1, label="Arbitration")
ax_jpn_r.fill_between(japan_df.index, n225["close"], n225["close"].min(), color="#8f8f8f", alpha=0.14, zorder=0)
ax_jpn_r.plot(japan_df.index, n225["close"], color="#b0b0b0", lw=1.0, linestyle=(0,(10,7)), alpha=0.6, zorder=1)
ax_jpn_r.tick_params(colors="#9a9a9a")

axes[1,1].plot(china_df.index, china_df["base_cum_profit"], 
               color="#1f77b4", lw=1, label="Base")
axes[1,1].plot(china_df.index, china_df["arb_cum_profit"],  
               color="#ff7f0e", lw=1, label="Arbitration")
ax_chn_r.fill_between(china_df.index, csi300["close"], csi300["close"].min(), color="#8f8f8f", alpha=0.14, zorder=0)
ax_chn_r.plot(china_df.index, csi300["close"], color="#b0b0b0", lw=1.0, linestyle=(0,(10,7)), alpha=0.6, zorder=1)
ax_chn_r.tick_params(colors="#9a9a9a")

labels = ["(a)", "(b)", "(c)", "(d)"]
for ax, lab in zip(axes.ravel(), labels):
    # 축 좌표계(왼아래=0,0 ~ 오른위=1,1) 기준으로 좌하단에 배치
    ax.text(0.5, -0.13, lab, transform=ax.transAxes,
            ha="center", va="top", fontsize=20)

# 격자 제거 + 박스형 해제(위/오른쪽 스파인 숨기기)
for ax in axes.ravel():
    ax.grid(False)  # 배경 격자 off
    ax.spines[['top','right']].set_visible(False)  # 상자형 방지

# 공통 레이블과 폰트 크기
for ax in axes.ravel():
    # ax.set_xlabel("Time-step", fontsize=20)
    # ax.set_ylabel("Cumulative-reward", fontsize=20)
    ax.tick_params(labelsize=14)    # 눈금 라벨 크기

fig.tight_layout()
fig.subplots_adjust(hspace=0.3)    # 서브플롯 간격(상하)
fig.subplots_adjust(wspace=0.2)    # 서브플롯 간격(좌우)

plt.show()


### 벤치마크 지수 그래프
# 한: 20190708 20250710
# 미: 20190703 20250707
# 일: 20190712 20250709
# 중: 20190715 20250711


import pandas as pd
import matplotlib.pyplot as plt


# Arbitration
def arb_country_cumulative_profit(country):
    df = pd.read_csv(f"../results/{country}/asset3.csv", 
                         header=None, names=["arb_account"])
    # 첫 번째 행 계좌 값이 0이면 첫 번째 행 삭제
    if not df.empty and pd.to_numeric(df.loc[0, "arb_account"], errors="coerce") == 0:
        df = df.drop(index=0).reset_index(drop=True)

    # 누적 수익률 (%)
    start_account = df["arb_account"].iloc[0]
    # df["cum_profit_abs"] = df["account"] - start_account 
    df["arb_cum_profit"] = (df["arb_account"] / start_account - 1) * 100

    # 전일 대비 수익률 (%)
    df["arb_ret"] = df["arb_account"].pct_change() * 100

    return df

# Base
def base_country_cumulative_profit(country):
    df = pd.read_csv(f"../results/{country}/asset_base.csv", 
                         header=None, names=["base_account"])
    # 첫 번째 행 계좌 값이 0이면 첫 번째 행 삭제
    if not df.empty and pd.to_numeric(df.loc[0, "base_account"], errors="coerce") == 0:
        df = df.drop(index=0).reset_index(drop=True)

    # 누적 수익률 (%)
    start_account = df["base_account"].iloc[0]
    df["base_cum_profit"] = (df["base_account"] / start_account - 1) * 100

    # 전일 대비 수익률 (%)
    df["base_ret"] = df["base_account"].pct_change() * 100

    return df

# Merged
def country_cumulative_profit(country):
    df_arb = arb_country_cumulative_profit(country)
    df_base = base_country_cumulative_profit(country)
    df = pd.concat([df_arb, df_base], axis=1)

    return df


korea_df = country_cumulative_profit("korea")
usa_df = country_cumulative_profit("usa")
japan_df = country_cumulative_profit("japan")
china_df = country_cumulative_profit("china")

# 벤치마크 지수
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
              .str.replace('M', '', regex=False)
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


# def trim_and_cum(cum_df, benchmark_df):
#     if len(benchmark_df) > len(cum_df):
#         drop_n = len(benchmark_df) - len(cum_df)
#         benchmark_df = benchmark_df.iloc[drop_n:]
    
#     benchmark_df['cumulative'] = (benchmark_df['close'] / benchmark_df['close'].iloc[0] - 1.0) * 100.0
#     return benchmark_df

# ks200 = trim_and_cum(korea_df, ks200)
# spx = trim_and_cum(usa_df, spx)
# n225 = trim_and_cum(japan_df, n225)
# csi300 = trim_and_cum(china_df, csi300)



plt.ion()
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 각 서브플롯에 데이터 그리기
axes[0,0].plot(ks200.index, ks200["close"], color="#b0b0b0", lw=1.0, linestyle=(0,(10,7)), alpha=0.6, zorder=1)
axes[0,0].fill_between(ks200.index, ks200["close"], ks200["close"].min(), color="#8f8f8f", alpha=0.14, zorder=0)
# ks200.tick_params(colors="#9a9a9a")

axes[0,1].plot(spx.index, spx["close"], color="#b0b0b0", lw=1.0, linestyle=(0,(10,7)), alpha=0.6, zorder=1)
axes[0,1].fill_between(spx.index, spx["close"], spx["close"].min(), color="#8f8f8f", alpha=0.14, zorder=0)
# ax_usa_r.tick_params(colors="#9a9a9a")

axes[1,0].plot(n225.index, n225["close"], color="#b0b0b0", lw=1.0, linestyle=(0,(10,7)), alpha=0.6, zorder=1)
axes[1,0].fill_between(n225.index, n225["close"], n225["close"].min(), color="#8f8f8f", alpha=0.14, zorder=0)
# ax_jpn_r.tick_params(colors="#9a9a9a")

axes[1,1].plot(csi300.index, csi300["close"], color="#b0b0b0", lw=1.0, linestyle=(0,(10,7)), alpha=0.6, zorder=1)
axes[1,1].fill_between(csi300.index, csi300["close"], csi300["close"].min(), color="#8f8f8f", alpha=0.14, zorder=0)
# ax_chn_r.tick_params(colors="#9a9a9a")

labels = ["(a)", "(b)", "(c)", "(d)"]
for ax, lab in zip(axes.ravel(), labels):
    # 축 좌표계(왼아래=0,0 ~ 오른위=1,1) 기준으로 좌하단에 배치
    ax.text(0.5, -0.13, lab, transform=ax.transAxes,
            ha="center", va="top", fontsize=20)

# 격자 제거 + 박스형 해제(위/오른쪽 스파인 숨기기)
for ax in axes.ravel():
    ax.grid(False)  # 배경 격자 off
    ax.spines[['top','right']].set_visible(False)  # 상자형 방지
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# 공통 레이블과 폰트 크기
# for ax in axes.ravel():
#     # ax.set_xlabel("Time-step", fontsize=20)
#     # ax.set_ylabel("Cumulative-reward", fontsize=20)
#     ax.tick_params(labelsize=14)    # 눈금 라벨 크기

for ax in axes.ravel():
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

fig.tight_layout()
fig.subplots_adjust(hspace=0.3)    # 서브플롯 간격(상하)
fig.subplots_adjust(wspace=0.2)    # 서브플롯 간격(좌우)

plt.show()