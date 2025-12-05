import argparse, yaml, os, json, torch, numpy as np, pandas as pd, sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from samil.datamodules import make_loaders
from samil.model_samil import SAMIL
from samil.model_abmil import ABMIL
from samil.metrics import balanced_acc, auroc_ovr

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', default='configs/colab_small.yaml')
ap.add_argument('--model', default='samil', choices=['abmil','samil'])
ap.add_argument('--checkpoint_dir', default=None, help="Directory containing saved model checkpoints")
ap.add_argument('--results_dir', default='results', help="Results directory with experiments")
ap.add_argument('--out', default='results/eval.csv')
ap.add_argument('--eval_on', default='test', choices=['test', 'val', 'both'])
args = ap.parse_args()


cfg = yaml.safe_load(open(args.cfg))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tr, va, te = make_loaders(cfg['data_root'], cfg['img_size'], 1)

all_results = []

def evaluate_model(model, data_loader, data_split_name):
    """Evaluate model on data loader."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for bag, y, _ in data_loader:
            bag = bag.squeeze(0).to(device)
            out = model(bag)
            prob = torch.softmax(out['logits'], dim=-1).cpu().numpy()
            ys.append(int(y))
            ps.append(prob)

    if len(ys) == 0:
        return None

    yhat = np.argmax(np.vstack(ps), axis=1)
    bal = balanced_acc(ys, yhat)
    auc = auroc_ovr(ys, np.vstack(ps))
    return {"split": data_split_name, "balanced_acc": bal, "auroc_ovr": auc}


if not args.checkpoint_dir and not Path(args.results_dir).exists():
    model = SAMIL() if args.model=='samil' else ABMIL()
    model.to(device)

    if args.eval_on in ['test', 'both']:
        test_result = evaluate_model(model, te, 'test')
        if test_result:
            test_result['model'] = args.model
            all_results.append(test_result)

    if args.eval_on in ['val', 'both']:
        val_result = evaluate_model(model, va, 'val')
        if val_result:
            val_result['model'] = args.model
            all_results.append(val_result)


elif args.checkpoint_dir:
    ckpt_dir = Path(args.checkpoint_dir)
    model_file = ckpt_dir / "model_best.pt"

    if not model_file.exists():

        pt_files = list(ckpt_dir.glob("*.pt"))
        if pt_files:
            model_file = pt_files[0]

    if model_file.exists():
        model = SAMIL() if args.model=='samil' else ABMIL()
        model.to(device)
        try:
            model.load_state_dict(torch.load(model_file, map_location=device))
            print(f"Loaded checkpoint: {model_file}")

            if args.eval_on in ['test', 'both']:
                test_result = evaluate_model(model, te, 'test')
                if test_result:
                    test_result['experiment'] = ckpt_dir.name
                    all_results.append(test_result)

            if args.eval_on in ['val', 'both']:
                val_result = evaluate_model(model, va, 'val')
                if val_result:
                    val_result['experiment'] = ckpt_dir.name
                    all_results.append(val_result)
        except Exception as e:
            print(f"ERROR loading {model_file}: {e}")


else:
    results_dir = Path(args.results_dir)
    if results_dir.exists():
        for exp_dir in sorted(results_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            model_file = exp_dir / "model_best.pt"
            if not model_file.exists():
                pt_files = list(exp_dir.glob("*.pt"))
                if pt_files:
                    model_file = pt_files[0]

            if model_file.exists():
                model = SAMIL() if args.model=='samil' else ABMIL()
                model.to(device)
                try:
                    model.load_state_dict(torch.load(model_file, map_location=device))

                    for split, loader, split_name in [('val', va, 'val'), ('test', te, 'test')]:
                        if args.eval_on in [split, 'both']:
                            result = evaluate_model(model, loader, split_name)
                            if result:
                                result['experiment'] = exp_dir.name
                                all_results.append(result)
                except Exception as e:
                    print(f"ERROR loading {model_file}: {e}")


if all_results:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(args.out, index=False)
    print(f"✓ Saved results to: {args.out}")
    print(df.to_string(index=False))
else:
    print("WARNING: No results to save")
