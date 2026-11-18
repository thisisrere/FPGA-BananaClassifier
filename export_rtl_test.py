# export_rtl_test.py
import numpy as np, torch
from common import BananaCNN, load_split, CLASSES
from quantize import build_quant, int_infer

model = BananaCNN()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

Xtr, _, Xva, yva = load_split(seed=0)
q = build_quant(model, Xtr)

x = np.round(Xva[0].numpy() * 255).astype(np.int64)   # (3,32,32), 0..255
with open("test_img.mem", "w") as f:
    for v in x.flatten():
        f.write(format(int(v), "02x") + "\n")

s = int_infer(q, x)
print("Label asli :", CLASSES[int(yva[0])])
print("Python skor:", s.tolist(), "| pred:", int(np.argmax(s)))
