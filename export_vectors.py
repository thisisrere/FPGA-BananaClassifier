import os, numpy as np, torch
from common import BananaCNN, load_split, CLASSES
from quantize import build_quant, int_infer

os.makedirs("output", exist_ok=True)
model = BananaCNN()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()
Xtr, _, Xva, yva = load_split(seed=0)
q = build_quant(model, Xtr)

imgs = np.round(Xva.numpy() * 255).astype(np.int64)

with open("output/test_all.mem", "w") as f:
    for x in imgs:
        for v in x.flatten():
            f.write(format(int(v), "02x") + "\n")

ok = 0
with open("output/expected.csv", "w") as f:
    f.write("idx,label,py_pred,s0,s1,s2\n")
    for i, x in enumerate(imgs):
        s = int_infer(q, x)
        p = int(np.argmax(s))
        lab = CLASSES[int(yva[i])]
        ok += int(CLASSES[p] == lab)
        f.write(f"{i},{lab},{CLASSES[p]},{s[0]},{s[1]},{s[2]}\n")

print("Jumlah gambar :", len(imgs))
print("Akurasi Python (int):", round(ok / len(imgs), 4))
