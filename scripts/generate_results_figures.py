
"""
Generate results tables and figures from ablation sweep.
Produces:
  - results/ablation_summary_table.csv  (expanded with means/stds)
  - results/figures/ablation_*.png      (bar charts, comparison plots)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


CSV_PATH = Path("results/ablation_runs/summary.csv")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} experiments")
print(df.head(10))


stats = df.groupby(["weight", "temp"]).agg({
    "balanced_acc": ["mean", "std", "min", "max"],
    "auroc": ["mean", "std"]
}).round(4)

print("\n=== Aggregated Statistics (mean ± std across seeds) ===")
print(stats)


extended = []
for (w, t), group in df.groupby(["weight", "temp"]):
    extended.append({
        "weight": w,
        "temp": t,
        "bal_acc_mean": group["balanced_acc"].mean(),
        "bal_acc_std": group["balanced_acc"].std(),
        "auroc_mean": group["auroc"].mean() if group["auroc"].notna().any() else np.nan,
        "auroc_std": group["auroc"].std() if group["auroc"].notna().any() else np.nan,
        "num_seeds": len(group),
    })
extended_df = pd.DataFrame(extended)
extended_csv = Path("results/ablation_summary_table.csv")
extended_df.to_csv(extended_csv, index=False)
print(f"\n✓ Saved: {extended_csv}")


fig, ax = plt.subplots(figsize=(10, 6))
pivot_bal = extended_df.pivot(index="weight", columns="temp", values="bal_acc_mean")
sns.heatmap(pivot_bal, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, vmin=0, vmax=1, cbar_kws={"label": "Balanced Acc"})
ax.set_title("Ablation Results: Balanced Accuracy by Contrastive Weight & Temperature", fontsize=12, fontweight="bold")
ax.set_xlabel("Temperature")
ax.set_ylabel("Contrastive Weight")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_balanced_acc_heatmap.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved: {OUT_DIR}/01_balanced_acc_heatmap.png")
plt.close()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


baseline = df[df["weight"] == 0.0]
temps_baseline = baseline["temp"].unique()
baseline_means = [baseline[baseline["temp"] == t]["balanced_acc"].mean() for t in sorted(temps_baseline)]
ax1.bar(sorted(temps_baseline), baseline_means, color="steelblue", alpha=0.7, edgecolor="black")
ax1.set_ylabel("Balanced Accuracy", fontsize=11)
ax1.set_xlabel("Temperature (not applicable, w=0)", fontsize=11)
ax1.set_title("Baseline: No Contrastive Loss (w=0.0)", fontsize=12, fontweight="bold")
ax1.set_ylim([0.4, 0.6])
ax1.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random (0.5)")
ax1.legend()


contrastive = df[df["weight"] > 0.0]
configs = [f"w={w:.1f}" for w in sorted(contrastive["weight"].unique())]
config_means = [contrastive[contrastive["weight"] == w]["balanced_acc"].mean() for w in sorted(contrastive["weight"].unique())]
ax2.bar(configs, config_means, color="coral", alpha=0.7, edgecolor="black")
ax2.set_ylabel("Balanced Accuracy", fontsize=11)
ax2.set_xlabel("Contrastive Weight")
ax2.set_title("With Contrastive Loss: Aggregated (w > 0.0)", fontsize=12, fontweight="bold")
ax2.set_ylim([0.4, 0.6])
ax2.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random (0.5)")
ax2.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "02_baseline_vs_contrastive.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved: {OUT_DIR}/02_baseline_vs_contrastive.png")
plt.close()


fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("tight")
ax.axis("off")

table_data = extended_df[[
    "weight", "temp", "bal_acc_mean", "bal_acc_std", "num_seeds"
]].round(4).values.tolist()

table_cols = ["Weight", "Temp", "Bal-Acc (μ)", "Bal-Acc (σ)", "Seeds"]
tbl = ax.table(cellText=table_data, colLabels=table_cols, cellLoc="center", loc="center",
               colColours=["#40466e"]*5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2)


colors = {0.0: "#ffcccc", 0.1: "#ccffcc", 1.0: "#ccccff"}
for i, (w, _, _, _, _) in enumerate(table_data, start=1):
    for j in range(5):
        tbl[(i, j)].set_facecolor(colors.get(w, "white"))

plt.title("Ablation Results: Balanced Accuracy Summary\n(mean ± std across 3 random seeds)", 
          fontsize=12, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(OUT_DIR / "03_summary_table.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved: {OUT_DIR}/03_summary_table.png")
plt.close()


fig, ax = plt.subplots(figsize=(12, 6))
df["config"] = df.apply(lambda r: f"w={r['weight']:.1f}\nt={r['temp']:.2f}", axis=1)
sns.boxplot(data=df, x="weight", y="balanced_acc", hue="temp", ax=ax, palette="Set2")
ax.set_ylabel("Balanced Accuracy", fontsize=11)
ax.set_xlabel("Contrastive Weight", fontsize=11)
ax.set_title("Balanced Accuracy Distribution across Seeds & Temperatures", fontsize=12, fontweight="bold")
ax.axhline(0.5, color="red", linestyle="--", alpha=0.3, linewidth=1)
ax.legend(title="Temperature", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_distribution_boxplot.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved: {OUT_DIR}/04_distribution_boxplot.png")
plt.close()

print("\n" + "="*60)
print("✓ ALL FIGURES AND TABLES GENERATED SUCCESSFULLY")
print("="*60)
print(f"\nOutput directory: {OUT_DIR.resolve()}")
print(f"Files created:")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  - {f.name}")
print(f"\nExtended summary: {extended_csv.resolve()}")
