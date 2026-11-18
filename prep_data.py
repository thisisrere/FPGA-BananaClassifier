# prep_data.py
import os
from PIL import Image

SRC = "raw"
DST = "data"
CLASSES = ["mentah", "matang", "busuk"]

for cls in CLASSES:
    folder = f"{SRC}/{cls}"
    out = f"{DST}/{cls}"
    os.makedirs(out, exist_ok=True)
    files = sorted(f for f in os.listdir(folder)
                   if os.path.isfile(os.path.join(folder, f)))
    ok = 0
    for f in files:
        try:
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            img = img.resize((32, 32), Image.BILINEAR)
            img.save(f"{out}/{ok:03d}.png")
            ok += 1
        except Exception:
            print("Lewati file rusak:", f)
    print(f"{cls}: {ok} gambar tersimpan")