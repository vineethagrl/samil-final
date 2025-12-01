import argparse, yaml, os, json, torch, numpy as np
from samil.datamodules import make_loaders
from samil.model_abmil import ABMIL
from samil.model_samil import SAMIL
from samil.metrics import balanced_acc, auroc_ovr

def load_cfg(path=None):
    if path and os.path.exists(path): return yaml.safe_load(open(path))
    return json.load(open("paths.json"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='configs/colab_small.yaml')
    ap.add_argument('--model', default='samil', choices=['abmil','samil'])
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)

    os.makedirs(cfg['save_dir'], exist_ok=True)
    tr, va, te = make_loaders(cfg['data_root'], cfg['img_size'], cfg['batch_size'])

    if args.model=='abmil':
        model = ABMIL()
    else:
        model = SAMIL(lambda_sa=cfg['lambda_sa'], use_bag_contrastive=cfg['use_bag_contrastive'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    for epoch in range(cfg['epochs']):
        model.train()
        for bag, y, _ in tr:
            bag, y = bag.squeeze(0).to(device), y.to(device)
            out = model(bag, target=y) if args.model=='samil' else model(bag)
            loss = out['loss'] if args.model=='samil' else torch.nn.functional.cross_entropy(out['logits'].unsqueeze(0), y.unsqueeze(0))
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for bag, y, _ in va:
                bag = bag.squeeze(0).to(device)
                out = model(bag) if args.model=='abmil' else model(bag)
                prob = torch.softmax(out['logits'], dim=-1).cpu().numpy()
                ys.append(int(y)); ps.append(prob)
        yhat = np.argmax(np.vstack(ps), axis=1)
        bal = balanced_acc(ys, yhat)
        auc = auroc_ovr(ys, np.vstack(ps))
        print(f"epoch {epoch}: val balanced-acc={bal:.3f} auroc-ovr={auc:.3f}")

if __name__ == "__main__":
    main()
