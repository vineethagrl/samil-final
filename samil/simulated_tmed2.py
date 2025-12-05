
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SimulatedTMED2(Dataset):
    """
    Synthetic TMED-2-like dataset.

    - Each item is a "study" (bag) with K images.
    - K is sampled between 27 and 97 (approx TMED-2 10-90 percentile range).
    - Each image is a random 3xHxW tensor.
    - Label is one of {0, 1, 2} corresponding to {no AS, early AS, significant AS}.
    """

    def __init__(
        self,
        n_studies: int = 100,
        img_size: int = 96,
        split: str = "train",
        seed: int = 0,
        class_probs=None,
    ):
        super().__init__()
        self.n_studies = n_studies
        self.img_size = img_size
        self.split = split


        if class_probs is None:

            class_probs = np.array([0.6, 0.2, 0.2], dtype=np.float32)
        class_probs = class_probs / class_probs.sum()

        rng = np.random.RandomState(seed)
        self.labels = rng.choice(3, size=n_studies, p=class_probs)


        self.lengths = rng.randint(27, 98, size=n_studies)

    def __len__(self):
        return self.n_studies

    def __getitem__(self, idx):
        K = int(self.lengths[idx])
        C = 3
        H = W = self.img_size


        bag = torch.randn(K, C, H, W, dtype=torch.float32)

        y = int(self.labels[idx])
        study_id = f"{self.split}_study_{idx}"


        return bag, torch.tensor(y, dtype=torch.long), study_id


def make_synthetic_loaders(
    img_size: int,
    batch_size: int,
    n_train: int = 80,
    n_val: int = 10,
    n_test: int = 10,
    seed: int = 0,
):
    """
    Returns (train_loader, val_loader, test_loader) for the synthetic TMED-2 data.
    """

    train_ds = SimulatedTMED2(
        n_studies=n_train, img_size=img_size, split="train", seed=seed
    )
    val_ds = SimulatedTMED2(
        n_studies=n_val, img_size=img_size, split="val", seed=seed + 1
    )
    test_ds = SimulatedTMED2(
        n_studies=n_test, img_size=img_size, split="test", seed=seed + 2
    )


    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
