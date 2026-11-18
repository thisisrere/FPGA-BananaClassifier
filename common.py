# common.py
import numpy as np, torch, torch.nn as nn
from torchvision import datasets, transforms

CLASSES = ["busuk", "matang", "mentah"]   # urutan abjad dari ImageFolder

class BananaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)        # 3x32x32 -> 4x30x30
        self.pool = nn.MaxPool2d(2)           # -> 4x15x15
        self.fc = nn.Linear(4 * 15 * 15, 3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        return self.fc(x.flatten(1))

def load_split(seed=0, val_frac=0.2):
    full = datasets.ImageFolder("data", transform=transforms.ToTensor())
    targets = np.array(full.targets)
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in range(len(CLASSES)):           # split per kelas (stratified)
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_va = max(1, round(len(idx) * val_frac))
        va += list(idx[:n_va])
        tr += list(idx[n_va:])
    tr, va = np.array(tr), np.array(va)
    X = torch.stack([full[i][0] for i in range(len(full))])
    y = torch.tensor(targets, dtype=torch.long)
    return X[tr], y[tr], X[va], y[va]