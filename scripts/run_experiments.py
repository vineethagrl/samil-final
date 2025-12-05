
"""Run a grid of contrastive ablation experiments and record summary metrics.

Creates per-experiment log files under `results/ablation_runs/<exp_name>/log.txt`
and a summary CSV `results/ablation_runs/summary.csv` with final validation metrics.
"""
import os, yaml, subprocess, sys, csv, time

BASE_CFG = 'configs/ablation_contrastive.yaml'
OUT_ROOT = 'results/ablation_runs'

WEIGHTS = [0.0, 0.1, 1.0]
TEMPS = [0.05, 0.1, 0.2]
SEEDS = [42, 123, 999]

def make_cfg(base, outpath, weight, temp):
    with open(base) as f:
        cfg = yaml.safe_load(f)
    cfg['contrastive_weight'] = float(weight)
    cfg['contrastive_temp'] = float(temp)
    cfg['save_dir'] = os.path.join(cfg.get('save_dir', './results/ablation_contrastive'), f'w{weight}_t{temp}')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        yaml.safe_dump(cfg, f)
    return outpath

def run_train(cfg_path, exp_dir, seed=None):
    os.makedirs(exp_dir, exist_ok=True)
    logpath = os.path.join(exp_dir, 'log.txt')
    cmd = [sys.executable, 'scripts/train.py', '--cfg', cfg_path, '--model', 'samil']
    if seed is not None:
        cmd += ['--seed', str(seed)]
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.abspath('.')
    with open(logpath, 'wb') as logf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)

        out = b''
        for chunk in iter(lambda: p.stdout.read(1024), b''):
            logf.write(chunk)
            out += chunk
        p.wait()
    return out.decode('utf-8', 'ignore')

def parse_final_metrics(stdout):

    lines = [l.strip() for l in stdout.splitlines() if l.strip()]
    bal = None; auc = None
    for l in reversed(lines):
        if l.startswith('epoch') and 'val balanced-acc=' in l:
            try:
                parts = l.split('val')[-1]

                for tok in parts.split():
                    if tok.startswith('balanced-acc='):
                        bal = float(tok.split('=')[1])
                    if tok.startswith('auroc-ovr='):
                        try:
                            auc = float(tok.split('=')[1])
                        except:
                            auc = None
                break
            except Exception:
                continue
    return bal, auc

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    summary_path = os.path.join(OUT_ROOT, 'summary.csv')
    with open(summary_path, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=['exp','weight','temp','seed','balanced_acc','auroc'])
        writer.writeheader()
        for w in WEIGHTS:
            for t in TEMPS:
                for s in SEEDS:
                    exp_name = f'w{w}_t{t}_s{s}'
                    cfg_out = os.path.join(OUT_ROOT, f'cfg_{exp_name}.yaml')
                    cfg_path = make_cfg(BASE_CFG, cfg_out, w, t)
                    exp_dir = os.path.join(OUT_ROOT, exp_name)
                    print(f'Running {exp_name} -> cfg {cfg_path} ...')
                    start = time.time()
                    out = run_train(cfg_path, exp_dir, seed=s)
                    dur = time.time() - start
                    bal, auc = parse_final_metrics(out)
                    writer.writerow({'exp':exp_name,'weight':w,'temp':t,'seed':s,'balanced_acc':bal,'auroc':auc})
                    csvf.flush()
                    print(f'Finished {exp_name} in {dur:.1f}s bal={bal} auc={auc}')

if __name__=='__main__':
    main()
