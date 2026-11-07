"""Builds MultiSeed_BUSI.ipynb, the controlled re-run for conf-1 (audit items A1, A2, B2, B6).

Run:  python3 build_multiseed_notebook.py   -> writes MultiSeed_BUSI.ipynb next to this file.
Then open it in Colab (File > Upload notebook), choose a T4 GPU runtime and run the cells in order.
Every result is written to Drive under BreastCancerNew/multiseed_v1/ and nothing is ever deleted,
so the notebook can be re-run after a disconnect and continues where it stopped.
"""
import json, os

CELLS = []
def md(s): CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Controlled multi-seed re-run (conf-1)

Purpose (audit items A1, A2, B2, B6):
1. **A1 check** – does feeding *unscaled* test images to a [0,1]-trained model reproduce run A's
   "AUROC 1.0 but 50% accuracy, false positives only" pattern? Uses the run-B VGG16-PF checkpoint already on Drive.
2. **Main experiment** – the 10 configurations (5 backbones x PF/FT), [0,1] scaling, trained with
   **5 seeds**; every random source is seeded (`tf.keras.utils.set_random_seed`) and data generators are rebuilt per run.
3. **Same-seed repeat** – seed 0 is trained twice for two configurations to measure pure GPU nondeterminism.
4. **Shape-only control** – the classifier sees only the white lesion silhouette (no texture), 3 seeds, two backbones.

Run cells 1 → 6 in order. Results go to `MyDrive/BreastCancerNew/multiseed_v1/`. If Colab disconnects,
re-run cells 1–4 and then the cell you were on; finished runs are skipped automatically.
Approximate time on a T4: ~10–12 min per training run.
""")

code(r'''
# ===== CELL 1: setup, paths, environment log =====
from google.colab import drive
drive.mount('/content/drive')

import os, sys, json, time, glob, shutil, platform, subprocess, random
import numpy as np, cv2, tensorflow as tf, sklearn
from pathlib import Path

DRIVE_BASE = "/content/drive/MyDrive/BreastCancerNew"
OLD_PROCESSED = f"{DRIVE_BASE}/processed_images"          # existing masked crops (437 benign / 210 malignant)
OUT = f"{DRIVE_BASE}/multiseed_v1"                         # new, never deleted
LOCAL = "/content/ms"
for d in [OUT, f"{OUT}/runs", f"{OUT}/diagnostics", LOCAL]:
    os.makedirs(d, exist_ok=True)

IMG_SIZE, BATCH, EPOCHS, PATIENCE = 224, 32, 50, 20
SPLIT_SEED = 42

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
ENV = {"python": sys.version.split()[0], "tensorflow": tf.__version__, "keras": tf.keras.__version__ if hasattr(tf.keras, "__version__") else "n/a",
       "numpy": np.__version__, "opencv": cv2.__version__, "sklearn": sklearn.__version__,
       "gpu": gpu, "platform": platform.platform(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
json.dump(ENV, open(f"{OUT}/environment_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=1)
print(json.dumps(ENV, indent=1))
assert tf.config.list_physical_devices('GPU'), "No GPU: Runtime > Change runtime type > T4 GPU"
''')

code(r'''
# ===== CELL 2: data (reuse the masked crops on Drive; same split rule as the paper) =====
# The crops in processed_images were produced by the paper's preprocessing (mask binarized at 127,
# extra-lesion pixels zeroed, 5% padded box, square, 224x224). If they are missing, stop: this
# notebook deliberately reuses them so that all runs see identical images.
counts = {c: len([f for f in os.listdir(f"{OLD_PROCESSED}/{c}") if f.endswith('.png')]) for c in ["benign", "malignant"]}
print("processed images on Drive:", counts)
assert counts == {"benign": 437, "malignant": 210}, "processed_images incomplete"

DATA = f"{LOCAL}/BUSI_split"
def build_split():
    rows = []
    for c in ["benign", "malignant"]:
        imgs = sorted(f for f in os.listdir(f"{OLD_PROCESSED}/{c}") if f.endswith('.png'))
        random.seed(SPLIT_SEED); random.shuffle(imgs)          # identical to the original split_dataset()
        n = len(imgs); t_end, v_end = int(n * 0.70), int(n * 0.85)
        for split, part in (("train", imgs[:t_end]), ("val", imgs[t_end:v_end]), ("test", imgs[v_end:])):
            dst = f"{DATA}/{split}/{c}"; os.makedirs(dst, exist_ok=True)
            for f in part:
                shutil.copy(f"{OLD_PROCESSED}/{c}/{f}", f"{dst}/{f}")
                rows.append((split, c, f))
    return rows
if not os.path.exists(f"{DATA}/test/malignant"):
    rows = build_split()
    with open(f"{OUT}/split.csv", "w") as fh:
        fh.write("split,class,filename\n" + "".join(f"{s},{c},{f}\n" for s, c, f in rows))
for s in ["train", "val", "test"]:
    print(s, {c: len(os.listdir(f"{DATA}/{s}/{c}")) for c in ["benign", "malignant"]})

# shape-only copy: every non-zero pixel -> 255 (lesion silhouette, no texture)
SHAPE = f"{LOCAL}/BUSI_shape"
if not os.path.exists(f"{SHAPE}/test/malignant"):
    for s in ["train", "val", "test"]:
        for c in ["benign", "malignant"]:
            os.makedirs(f"{SHAPE}/{s}/{c}", exist_ok=True)
            for f in os.listdir(f"{DATA}/{s}/{c}"):
                im = cv2.imread(f"{DATA}/{s}/{c}/{f}")
                sil = ((im.max(axis=2) > 0) * 255).astype(np.uint8)
                cv2.imwrite(f"{SHAPE}/{s}/{c}/{f}", cv2.merge([sil, sil, sil]))
print("shape-only images ready")
''')

code(r'''
# ===== CELL 3: helpers (generators, model, evaluation) =====
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121, Xception, InceptionV3, ResNet50, VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

BACKBONES = {"DenseNet121": DenseNet121, "Xception": Xception, "InceptionV3": InceptionV3,
             "ResNet50": ResNet50, "VGG16": VGG16}
CFG = {  # identical to the paper (run B/C learning rates)
    "PF": {"freeze": 0.50, "freeze_vgg": 0.60, "lr": 1e-4},
    "FT": {"freeze": 0.40, "freeze_vgg": 0.50, "lr": 1e-5},
}

def gens(root, seed, scale=1/255.):
    aug = ImageDataGenerator(rescale=scale, horizontal_flip=True, vertical_flip=True, zoom_range=0.20,
                             rotation_range=15, width_shift_range=0.10, height_shift_range=0.10, fill_mode='nearest')
    plain = ImageDataGenerator(rescale=scale)
    kw = dict(target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH, class_mode='binary', classes=['benign', 'malignant'])
    tr = aug.flow_from_directory(f"{root}/train", shuffle=True, seed=seed, **kw)
    va = plain.flow_from_directory(f"{root}/val", shuffle=False, **kw)
    te = plain.flow_from_directory(f"{root}/test", shuffle=False, **kw)
    return tr, va, te

def build(backbone, strategy):
    c = CFG[strategy]
    base = BACKBONES[backbone](include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    frac = c["freeze_vgg"] if backbone == "VGG16" else c["freeze"]
    k = int(len(base.layers) * frac)
    for i, l in enumerate(base.layers):
        l.trainable = i >= k
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(512, activation='relu')(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x); x = layers.Dropout(0.3)(x)
    m = Model(base.input, layers.Dense(1, activation='sigmoid')(x))
    m.compile(optimizer=tf.keras.optimizers.Adam(c["lr"]), loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auroc')])
    return m, frac, c["lr"]

def youden(y, p):
    u = np.unique(p); cuts = np.concatenate(([u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]))
    j = np.array([(p[y == 1] >= t).mean() + (p[y == 0] < t).mean() - 1 for t in cuts])
    return float(np.median(cuts[np.isclose(j, j.max())]))

def counts(y, p, t):
    pr = (p >= t).astype(int)
    return {"tp": int(((pr == 1) & (y == 1)).sum()), "tn": int(((pr == 0) & (y == 0)).sum()),
            "fp": int(((pr == 1) & (y == 0)).sum()), "fn": int(((pr == 0) & (y == 1)).sum())}

def evaluate(model, va, te):
    va.reset(); te.reset()
    pv = model.predict(va, verbose=0).ravel(); pt = model.predict(te, verbose=0).ravel()
    yv, yt = va.classes.astype(int), te.classes.astype(int)
    t = youden(yv, pv)
    return {"val_auroc": float(roc_auc_score(yv, pv)), "test_auroc": float(roc_auc_score(yt, pt)),
            "val_at_0.5": counts(yv, pv, 0.5), "test_at_0.5": counts(yt, pt, 0.5),
            "thr_val": t, "test_at_thr": counts(yt, pt, t)}, pv, pt, yv, yt
print("helpers ready")
''')

code(r'''
# ===== CELL 4: one training run (skips runs that already finished) =====
def run_one(tag, backbone, strategy, seed, root=None, note=""):
    root = root or DATA
    d = f"{OUT}/runs/{tag}"
    if os.path.exists(f"{d}/result.json"):
        print("skip (done):", tag); return json.load(open(f"{d}/result.json"))
    os.makedirs(d, exist_ok=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)          # python, numpy and TF random state
    tr, va, te = gens(root, seed)
    y = tr.classes; cw = {0: len(y) / (2 * (y == 0).sum()), 1: len(y) / (2 * (y == 1).sum())}
    model, frac, lr = build(backbone, strategy)
    ck = f"{LOCAL}/{tag}.keras"
    t0 = time.time()
    h = model.fit(tr, epochs=EPOCHS, validation_data=va, class_weight=cw, verbose=2, callbacks=[
        ModelCheckpoint(ck, monitor='val_auroc', mode='max', save_best_only=True, verbose=0),
        EarlyStopping(monitor='val_auroc', mode='max', patience=PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-8, verbose=0)])
    model = tf.keras.models.load_model(ck)        # evaluate exactly the saved best checkpoint
    res, pv, pt, yv, yt = evaluate(model, va, te)
    hist = {k: [float(v) for v in vs] for k, vs in h.history.items()}
    best_ep = int(np.argmax(hist["val_auroc"])) + 1
    res.update({"tag": tag, "backbone": backbone, "strategy": strategy, "seed": seed, "data": root, "note": note,
                "freeze": frac, "lr": lr, "epochs_run": len(hist["loss"]), "best_epoch": best_ep,
                "val_loss_at_best": hist["val_loss"][best_ep - 1], "class_weights": cw,
                "minutes": round((time.time() - t0) / 60, 1), "env": ENV})
    json.dump(hist, open(f"{d}/history.json", "w"))
    np.savez_compressed(f"{d}/scores.npz", val=pv, test=pt, y_val=yv, y_test=yt,
                        val_files=np.array(va.filenames), test_files=np.array(te.filenames))
    json.dump(res, open(f"{d}/result.json", "w"), indent=1)   # written last = run complete
    os.remove(ck)
    a = res["test_at_0.5"]; print(f"DONE {tag}: test AUROC {res['test_auroc']:.4f}  FP {a['fp']} FN {a['fn']}  ({res['minutes']} min)")
    del model; tf.keras.backend.clear_session()
    return res
print("run_one ready")
''')

code(r'''
# ===== CELL 5: A1 diagnostic (about 2 minutes) =====
# Run A reported VGG16-PF test accuracy 50% (49 FP, AUROC 1.0) although its validation accuracy at the saved
# epoch was 95.9%. Hypothesis: test images were scored without the 1/255 rescale. Test it on the run-B
# VGG16-PF checkpoint (same architecture and training scale).
ck = f"{DRIVE_BASE}/results/models/VGG16_feature_extractor_best.keras"
out = {}
if os.path.exists(ck):
    m = tf.keras.models.load_model(ck)
    for name, scale in (("rescaled_1_255", 1/255.), ("unscaled_0_255", None)):
        _, va, te = gens(DATA, 0, scale=scale)
        r, pv, pt, yv, yt = evaluate(m, va, te)
        r["benign_test_score_mean"] = float(pt[yt == 0].mean()); r["malignant_test_score_mean"] = float(pt[yt == 1].mean())
        out[name] = r
        a = r["test_at_0.5"]; print(f"{name:16s} test AUROC {r['test_auroc']:.4f}  acc {(a['tp']+a['tn'])/98*100:.2f}%  FP {a['fp']} FN {a['fn']}")
    json.dump(out, open(f"{OUT}/diagnostics/A1_scaling_check.json", "w"), indent=1)
    del m; tf.keras.backend.clear_session()
else:
    print("checkpoint not found:", ck)
''')

code(r'''
# ===== CELL 6: main experiment + controls (long; safe to re-run after a disconnect) =====
SEEDS = [0, 1, 2, 3, 4]           # stop after 3 seeds if time is short; every finished run is kept
ORDER = [(b, s) for s in ("PF", "FT") for b in BACKBONES]

for seed in SEEDS:
    for b, s in ORDER:
        run_one(f"main_{b}_{s}_seed{seed}", b, s, seed)
    if seed == 0:
        # same seed again: differences here come only from GPU nondeterminism
        for b, s in (("VGG16", "PF"), ("DenseNet121", "PF")):
            run_one(f"repeat_{b}_{s}_seed0", b, s, 0, note="same seed as main_*_seed0")
        # shape-only control: lesion silhouette only
        for b in ("DenseNet121", "VGG16"):
            for sd in (0, 1, 2):
                run_one(f"shape_{b}_PF_seed{sd}", b, "PF", sd, root=SHAPE, note="shape-only input")
print("ALL DONE")
''')

code(r'''
# ===== CELL 7: summary table (run any time) =====
import pandas as pd
rows = []
for f in sorted(glob.glob(f"{OUT}/runs/*/result.json")):
    r = json.load(open(f)); a, t = r["test_at_0.5"], r["test_at_thr"]
    rows.append({"tag": r["tag"], "backbone": r["backbone"], "strategy": r["strategy"], "seed": r["seed"],
                 "val_auroc": r["val_auroc"], "test_auroc": r["test_auroc"],
                 "test_acc": (a["tp"] + a["tn"]) / 98 * 100, "FP": a["fp"], "FN": a["fn"],
                 "thr_val": r["thr_val"], "acc_at_thr": (t["tp"] + t["tn"]) / 98 * 100,
                 "best_epoch": r["best_epoch"], "minutes": r["minutes"]})
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/summary_all_runs.csv", index=False)
main = df[df.tag.str.startswith("main_")]
print(main.groupby(["backbone", "strategy"])[["test_auroc", "test_acc", "FP", "FN"]].agg(["mean", "std", "min", "max"]).round(3))
print("\nfinished runs:", len(df), "| saved:", f"{OUT}/summary_all_runs.csv")
''')

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MultiSeed_BUSI.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
