# quantize.py
import numpy as np, torch
from common import BananaCNN, load_split

SH = 16   # requantisasi fixed-point: nilai >> 16

def build_quant(model, Xtr):
    Wc = model.conv.weight.detach().numpy().astype(np.float64)   # (4,3,3,3)
    bc = model.conv.bias.detach().numpy().astype(np.float64)     # (4,)
    Wf = model.fc.weight.detach().numpy().astype(np.float64)     # (3,900)
    bf = model.fc.bias.detach().numpy().astype(np.float64)       # (3,)

    # Kalibrasi: skala aktivasi pooling float -> 0..255
    with torch.no_grad():
        a1 = model.pool(torch.relu(model.conv(Xtr)))
    s_a = 255.0 / a1.max().item()

    sc = 127.0 / np.abs(Wc).max()
    sf = 127.0 / np.abs(Wf).max()
    wc_q = np.clip(np.round(Wc * sc), -128, 127).astype(np.int64)
    wf_q = np.clip(np.round(Wf * sf), -128, 127).astype(np.int64)
    bc_q = np.round(bc * 255 * sc).astype(np.int64)     # input integer 0..255
    bf_q = np.round(bf * s_a * sf).astype(np.int64)

    # acc_conv ~= 255*sc*a1  ->  p = acc_conv * s_a/(255*sc)
    M = s_a / (255 * sc)
    M_fix = int(round(M * (1 << SH)))

    return dict(wc=wc_q, bc=bc_q, wf=wf_q, bf=bf_q, M=M_fix, sh=SH,
                s_a=s_a, sc=sc, sf=sf)

def int_infer(q, x):
    """x: array int64 (3,32,32) nilai 0..255. Mengembalikan skor int (3,)."""
    wc, bc = q["wc"], q["bc"]
    acc = np.zeros((4, 30, 30), np.int64)
    for f in range(4):
        a = np.full((30, 30), bc[f], np.int64)
        for c in range(3):
            for i in range(3):
                for j in range(3):
                    a += wc[f, c, i, j] * x[c, i:i + 30, j:j + 30]
        acc[f] = a
    acc = np.maximum(acc, 0)                       # ReLU
    p = np.clip((acc * q["M"]) >> q["sh"], 0, 255) # requantisasi ke 0..255
    pooled = p.reshape(4, 15, 2, 15, 2).max(axis=(2, 4))   # MaxPool 2x2
    return q["wf"] @ pooled.flatten() + q["bf"]    # FC

def int_predict(q, X):
    out = []
    for x in X:
        xu = np.round(x * 255).astype(np.int64)
        out.append(int(np.argmax(int_infer(q, xu))))
    return np.array(out)

def write_mem(fn, arr, bits):
    with open(fn, "w") as f:
        for v in np.asarray(arr).flatten():
            f.write(format(int(v) & ((1 << bits) - 1), f"0{bits//4}x") + "\n")

if __name__ == "__main__":
    model = BananaCNN()
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()
    Xtr, _, _, _ = load_split(seed=0)
    q = build_quant(model, Xtr)

    write_mem("w_conv.mem", q["wc"].flatten(), 8)   # urutan: f, c, i, j
    write_mem("b_conv.mem", q["bc"], 32)
    write_mem("w_fc.mem", q["wf"].flatten(), 8)     # urutan: kelas, lalu 900 fitur (f, r, c)
    write_mem("b_fc.mem", q["bf"], 32)
    with open("params.txt", "w") as f:
        f.write(f"M={q['M']}\nSH={q['sh']}\n")

    print("s_a =", round(q["s_a"], 4), "| sc =", round(q["sc"], 4), "| sf =", round(q["sf"], 4))
    print("File .mem dan params.txt tersimpan")