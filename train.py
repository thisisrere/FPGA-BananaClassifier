# train.py
import torch, torch.nn as nn
from common import BananaCNN, load_split

EPOCHS = 150
BATCH = 16

def augment(x):
    x = x.clone()
    # flip horizontal
    flip = torch.rand(x.size(0)) < 0.5
    x[flip] = torch.flip(x[flip], dims=[3])
    # geser posisi acak sampai 2 piksel (per gambar)
    p = 2
    xp = torch.nn.functional.pad(x, (p, p, p, p), mode="replicate")
    out = torch.empty_like(x)
    for i in range(x.size(0)):
        dy, dx = torch.randint(0, 2 * p + 1, (2,)).tolist()
        out[i] = xp[i, :, dy:dy + 32, dx:dx + 32]
    # kecerahan
    gain = torch.empty(x.size(0), 1, 1, 1).uniform_(0.8, 1.2)
    return (out * gain).clamp(0, 1)

if __name__ == "__main__":
    torch.manual_seed(0)
    Xtr, ytr, Xva, yva = load_split(seed=0)
    print("Train:", len(Xtr), "| Validasi:", len(Xva))
    print("Train per kelas:", torch.bincount(ytr, minlength=3).tolist())
    print("Val per kelas  :", torch.bincount(yva, minlength=3).tolist())

    counts = torch.bincount(ytr, minlength=3).float()
    weights = counts.sum() / (3 * counts)
    print("Bobot kelas    :", [round(w, 3) for w in weights.tolist()])

    model = BananaCNN()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    lossf = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = augment(Xtr[idx]), ytr[idx]
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()

        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xva).argmax(1) == yva).float().mean().item()
            print(f"Epoch {ep+1:3d} | loss train: {total/len(Xtr):.4f} | akurasi validasi: {acc:.4f}")

    torch.save(model.state_dict(), "model.pt")
    print("Model disimpan ke model.pt")