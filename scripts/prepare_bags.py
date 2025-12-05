"""Synthetic data preparation helper for smoke testing.

This script creates a small synthetic dataset in the layout expected by
`samil.datamodules.EchoStudyBags`:

data/tmed2/train/study_000/*.png
data/tmed2/val/study_000/*.png
data/tmed2/test/study_000/*.png

and writes train/val/test label CSV files.
"""
import os
from PIL import Image
import numpy as np

def make_image(path, size=(96,96), val=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = (np.random.rand(*size)*255).astype('uint8')
    img = Image.fromarray(arr)
    img.save(path)

def make_study(dirpath, nimgs=8):
    os.makedirs(dirpath, exist_ok=True)
    for i in range(nimgs):
        p = os.path.join(dirpath, f"img_{i:03d}.png")
        make_image(p)

def write_labels(root, split, nstudies=4):
    out = []
    for i in range(nstudies):
        sid = f"study_{i:03d}"
        label = 1 if i%2==0 else 0
        out.append(f"{sid},{label}\n")
    with open(os.path.join(root, f"{split}_labels.csv"), 'w') as f:
        f.writelines(out)

def create_dataset(root="data/tmed2", nstudies=4, nimgs=8):
    for split in ["train","val","test"]:
        sroot = os.path.join(root, split)
        for i in range(nstudies):
            sd = os.path.join(sroot, f"study_{i:03d}")
            make_study(sd, nimgs)
        write_labels(root, split, nstudies)
    print(f"Created synthetic dataset at {root}")

if __name__=="__main__":
    create_dataset()
