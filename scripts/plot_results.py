import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='results/eval.csv', help="Metrics CSV file")
ap.add_argument('--out_dir', default='results/figures', help="Output directory for plots")
ap.add_argument('--metric', default='balanced_acc', help="Metric to plot")
args = ap.parse_args()

csv_path = Path(args.csv)
if not csv_path.exists():
    print(f"ERROR: CSV not found: {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded data:\n{df}")

os.makedirs(args.out_dir, exist_ok=True)


if 'experiment' in df.columns and 'balanced_acc' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='experiment', y='balanced_acc', hue='split')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Balanced Accuracy')
    plt.title('Baseline vs Ablation: Balanced Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'balanced_acc_by_experiment.png'), dpi=150)
    print(f"✓ Saved: {args.out_dir}/balanced_acc_by_experiment.png")
    plt.close()


if 'experiment' in df.columns and 'auroc_ovr' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='experiment', y='auroc_ovr', hue='split')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('AUROC OVR')
    plt.title('Baseline vs Ablation: AUROC')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'auroc_by_experiment.png'), dpi=150)
    print(f"✓ Saved: {args.out_dir}/auroc_by_experiment.png")
    plt.close()


plt.figure(figsize=(12, 6))
plt.axis('off')
table_data = df.round(4).values.tolist()
table_cols = df.columns.tolist()
table = plt.table(cellText=table_data, colLabels=table_cols, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2)
plt.title('Metrics Summary', pad=20, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, 'metrics_table.png'), dpi=150, bbox_inches='tight')
print(f"✓ Saved: {args.out_dir}/metrics_table.png")
plt.close()

print(f"\n✓ All plots saved to: {args.out_dir}/")
