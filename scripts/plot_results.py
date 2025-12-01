import argparse, pandas as pd, matplotlib.pyplot as plt, os
ap = argparse.ArgumentParser()
ap.add_argument('--csvs', nargs='+', required=False)
ap.add_argument('--out', default='results/figures/ablation.png')
args = ap.parse_args()

dfs = []
if args.csvs:
    for c in args.csvs:
        dfs.append(pd.read_csv(c))
else:
    # default look in results/ for any eval.csv
    for root,_,files in os.walk('results'):
        for f in files:
            if f.endswith('.csv'):
                dfs.append(pd.read_csv(os.path.join(root,f)))
df = pd.concat(dfs, ignore_index=True)
plt.figure()
df.plot(x='model', y=['balanced_acc','auroc_ovr'], kind='bar')
os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.tight_layout(); plt.savefig(args.out)
print("saved", args.out)
