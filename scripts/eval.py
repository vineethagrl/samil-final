import argparse, yaml, os, json, torch, numpy as np, pandas as pd
from samil.datamodules import make_loaders
from samil.model_samil import SAMIL
from samil.model_abmil import ABMIL
from samil.metrics import balanced_acc, auroc_ovr

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', default='configs/colab_small.yaml')
ap.add_argument('--model', default='samil', choices=['abmil','samil'])
ap.add_argument('--out', default='results/eval.csv')
args = ap.parse_args()
cfg = yaml.safe_load(open(args.cfg))
tr, va, te = make_loaders(cfg['data_root'], cfg['img_size'], 1)
model = SAMIL() if args.model=='samil' else ABMIL()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device); model.eval()
ys, ps = [], []
with torch.no_grad():
    for bag, y, _ in te:
        bag = bag.squeeze(0).to(device)
        out = model(bag)
        prob = torch.softmax(out['logits'], dim=-1).cpu().numpy()
        ys.append(int(y)); ps.append(prob)
yhat = np.argmax(np.vstack(ps), axis=1)
bal = balanced_acc(ys, yhat)
auc = auroc_ovr(ys, np.vstack(ps))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
pd.DataFrame([{"model": args.model, "balanced_acc": bal, "auroc_ovr": auc}]).to_csv(args.out, index=False)
print("saved", args.out)
