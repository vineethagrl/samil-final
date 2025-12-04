import os, glob, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class EchoStudyBags(Dataset):
    def __init__(self, root, split="train", img_size=112):
        self.root = root
        self.split = split
        self.studies = sorted(glob.glob(os.path.join(root, split, "study_*")))
        self.labels = self._load_labels(os.path.join(root, f"{split}_labels.csv"))
        self.tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

    def _load_labels(self, csv_path):
        mapping = {}
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                for line in f:
                    sid, y = line.strip().split(",")
                    mapping[sid] = int(y)
        return mapping

    def __len__(self): return len(self.studies)

    def __getitem__(self, idx):
        sdir = self.studies[idx]
        sid = os.path.basename(sdir)
        imgs = []
        for p in sorted(glob.glob(os.path.join(sdir, "*.png"))):
            img = Image.open(p).convert("L")
            imgs.append(self.tf(img))
        bag = torch.stack(imgs)  # [K,1,H,W]
        y = self.labels.get(sid, 0)
        return bag, torch.tensor(y), sid

# Dummy bag dataset for simulated TMED-2
class DummyBagDataset(Dataset):
    def __init__(self, num_bags=20, bag_size=8, img_size=112, num_classes=2):
        self.num_bags = num_bags
        self.bag_size = bag_size
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self): return self.num_bags

    def __getitem__(self, idx):
        # Simulate grayscale (1 channel) echo frames
        bag = torch.randn(self.bag_size, 1, self.img_size, self.img_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return bag, torch.tensor(label), f"sim_{idx:03d}"

def make_loaders(root, img_size, bs=1, use_dummy=True):
    if use_dummy:
        train = DummyBagDataset(num_bags=100, bag_size=8, img_size=img_size)
        val   = DummyBagDataset(num_bags=20, bag_size=8, img_size=img_size)
        test  = DummyBagDataset(num_bags=20, bag_size=8, img_size=img_size)
    else:
        train = EchoStudyBags(root, "train", img_size)
        val   = EchoStudyBags(root, "val", img_size)
        test  = EchoStudyBags(root, "test", img_size)
    
    return (
        DataLoader(train, batch_size=bs, shuffle=True, num_workers=2),
        DataLoader(val,   batch_size=1, shuffle=False, num_workers=2),
        DataLoader(test,  batch_size=1, shuffle=False, num_workers=2)
    )
