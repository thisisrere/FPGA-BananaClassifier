# evaluate.py
import numpy as np, torch
from common import BananaCNN, load_split, CLASSES
from quantize import build_quant, int_predict

torch.manual_seed(0)
model = BananaCNN()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

Xtr, _, Xva, yva = load_split(seed=0)
y = yva.numpy()

with torch.no_grad():
    pf = model(Xva).argmax(1).numpy()

q = build_quant(model, Xtr)
pq = int_predict(q, Xva.numpy())

print(f"Jumlah validasi : {len(y)}")
print(f"Akurasi float   : {(pf == y).mean():.4f}")
print(f"Akurasi int     : {(pq == y).mean():.4f}")
print(f"Float vs int sama: {(pf == pq).mean():.4f}")

print("\nKelas:", CLASSES)
cm = np.zeros((3, 3), dtype=int)
for t, p in zip(y, pq):
    cm[t, p] += 1
print("Confusion matrix (baris = label asli, kolom = prediksi int):")
print(cm)

print("\nAkurasi per kelas:")
for c, name in enumerate(CLASSES):
    n = cm[c].sum()
    print(f"  {name:7s}: {cm[c, c]}/{n} = {cm[c, c]/max(n,1):.2f}")