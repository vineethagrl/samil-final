import argparse, yaml, os, json, torch, numpy as np, sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from samil.datamodules import make_loaders
from samil.model_abmil import ABMIL
from samil.model_samil import SAMIL
from samil.metrics import balanced_acc, auroc_ovr
from samil.losses import bag_contrastive_ntxent

def load_cfg(path=None):
    if path and os.path.exists(path): return yaml.safe_load(open(path))
    return json.load(open("paths.json"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', '--cfg', default='configs/colab_small.yaml', dest='cfg')
    ap.add_argument('--model', default='samil', choices=['abmil','samil'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--checkpoint_dir', default=None)
    args = ap.parse_args()


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = load_cfg(args.cfg)

    os.makedirs(cfg['save_dir'], exist_ok=True)


    if args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        ckpt_dir = cfg['save_dir']

    tr, va, te = make_loaders(cfg['data_root'], cfg['img_size'], cfg['batch_size'])

    if args.model=='abmil':
        model = ABMIL()
    else:
        model = SAMIL(lambda_sa=cfg.get('lambda_sa', 0.5), use_bag_contrastive=cfg.get('use_bag_contrastive', False))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    contrastive_weight = cfg.get('contrastive_weight', 1.0)

    for epoch in range(cfg['epochs']):
        model.train()
        for batch in tr:
            bag, y, _ = batch

            if bag.dim() == 5:
                if bag.size(0) == 1:
                    bag = bag.squeeze(0)
                    B = 1
                else:
                    B = bag.size(0)
            else:
                B = 1

            total_loss = 0.0
            reps_list = []
            labels_list = []


            for i in range(B):
                bag_i = bag[i].to(device) if B>1 else bag.to(device)
                if B>1:
                    y_i = y[i].to(device)
                else:
                    y_i = y.to(device)
                    if hasattr(y_i, 'dim') and y_i.dim() > 0:
                        y_i = y_i.squeeze(0)
                out = model(bag_i, target=y_i) if args.model=='samil' else model(bag_i)
                if args.model=='samil':
                    loss_i = out['loss'] if out['loss'] is not None else torch.tensor(0.0, device=device)
                else:
                    loss_i = torch.nn.functional.cross_entropy(out['logits'].unsqueeze(0), y_i.unsqueeze(0))
                total_loss = total_loss + loss_i
                if args.model=='samil' and model.use_bag_contrastive and 'rep' in out:
                    reps_list.append(out['rep'].unsqueeze(0))
                    labels_list.append(y_i.unsqueeze(0))

            if len(reps_list) > 0:
                reps = torch.cat(reps_list, dim=0)
                labels = torch.cat(labels_list, dim=0).view(-1)
                c_loss = bag_contrastive_ntxent(reps, labels, temperature=cfg.get('contrastive_temp', 0.1))
                total_loss = total_loss + contrastive_weight * c_loss

            opt.zero_grad()
            total_loss.backward()
            opt.step()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for bag, y, _ in va:

                bag_i = bag.squeeze(0).to(device) if bag.dim()==5 and bag.size(0)==1 else bag.to(device)
                out = model(bag_i) if args.model=='abmil' else model(bag_i)
                prob = torch.softmax(out['logits'], dim=-1).cpu().numpy()
                ys.append(int(y)); ps.append(prob)
        yhat = np.argmax(np.vstack(ps), axis=1)
        bal = balanced_acc(ys, yhat)
        auc = auroc_ovr(ys, np.vstack(ps))
        print(f"epoch {epoch}: val balanced-acc={bal:.3f} auroc-ovr={auc:.3f}")

if __name__ == "__main__":
    main()
