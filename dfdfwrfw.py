import pandas as pd
import matplotlib.pyplot as plt

# ---- paths ----
korea_base_asset = "/home/hail/Desktop/results/korea/asset_base.csv"
korea_arbitration_asset = "/home/hail/Desktop/results/korea/asset3.csv"
usa_base_asset = "/home/hail/Desktop/results/usa/asset_base.csv"
usa_arbitration_asset = "/home/hail/Desktop/results/usa/asset3.csv"
japan_base_asset = "/home/hail/Desktop/results/japan/asset_base.csv"
japan_arbitration_asset = "/home/hail/Desktop/results/japan/asset3.csv"
china_base_asset = "/home/hail/Desktop/results/china/asset_base.csv"
china_arbitration_asset = "/home/hail/Desktop/results/china/asset3.csv"

countries = [
    ("Korea", korea_base_asset, korea_arbitration_asset),
    ("USA", usa_base_asset, usa_arbitration_asset),
    ("Japan", japan_base_asset, japan_arbitration_asset),
    ("China", china_base_asset, china_arbitration_asset),
]

# ---- µ¥ÀÌÅÍ ºÒ·¯¿Í¼­ ÀüÃ¼ YÃà ¹üÀ§ °è»ê ----
all_values = []
profits_summary = []  # ´©Àû ¼öÀÍ ÀúÀå¿ë

for country, base_path, arb_path in countries:
    base = pd.read_csv(base_path, header=None).iloc[1:, 0].astype(float) - 1_000_000
    arb = pd.read_csv(arb_path, header=None).iloc[1:, 0].astype(float) - 1_000_000

    all_values.extend(base)
    all_values.extend(arb)

    # ÃÖÁ¾ ´©Àû ¼öÀÍ(¸¶Áö¸· ½ÃÁ¡ ±âÁØ)
    final_base = base.iloc[-1]
    final_arb = arb.iloc[-1]
    profits_summary.append((country, final_base, final_arb))

ymin, ymax = min(all_values), max(all_values)

# ---- ±×·¡ÇÁ ----
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Cumulative Profit (¥Ä from 1,000,000)", fontsize=15)

for ax, (country, base_path, arb_path) in zip(axes.flatten(), countries):
    base = pd.read_csv(base_path, header=None).iloc[1:].reset_index(drop=True)
    arb = pd.read_csv(arb_path, header=None).iloc[1:].reset_index(drop=True)

    n = min(len(base), len(arb))
    base = base.iloc[:n, 0].astype(float) - 1_000_000
    arb = arb.iloc[:n, 0].astype(float) - 1_000_000

    ax.plot(base, label="Base", color="blue", linewidth=1.2)
    ax.plot(arb, label="Arbitration", color="orange", linewidth=1.2)

    ax.set_title(country, fontsize=13)
    ax.set_xlabel("Time-step", fontsize=10)
    ax.set_ylabel("Profit (float)", fontsize=10)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis='both', labelsize=9, length=6, width=1.2)
    ax.grid(False)

    # Ãà¸¸ ³²±â±â
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

# ---- ¹ü·Ê 1°³¸¸ ÇÏ´Ü ÅëÇÕ ----
lines, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(lines, labels, loc='lower center', ncol=2, fontsize=10, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

# ---- ´©Àû ¼öÀÍ Ãâ·Â ----
print("=== Final Cumulative Profit (¥Ä from 1,000,000) ===")
for country, base_profit, arb_profit in profits_summary:
    print(f"{country:<8} | Base: {base_profit:,.2f} | Arbitration: {arb_profit:,.2f}")
