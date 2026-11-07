# ============================================
# CELL 7: VALIDATION-TUNED THRESHOLD RE-SCORING (audit item M7)
# For every saved model:
#   1. predict on the validation and test splits (no retraining)
#   2. choose the threshold that maximizes Youden's J on VALIDATION only
#   3. apply that threshold to TEST and report metrics next to the 0.5 results
# Also saves per-case predictions (audit item M10) and runs exact McNemar tests
# between the two strategies of each backbone (audit item M3).
# Run after CELL 1 (Drive mounted) and CELL 2 (BUSI_dataset split on local disk).
# ============================================

import os, gc
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE     = 224
BATCH_SIZE   = 32
DATASET_PATH = "/content/BUSI_dataset"
SAVE_DIR     = "/content/drive/MyDrive/BreastCancerNew/results"
OUT_DIR      = f"{SAVE_DIR}/threshold_rescoring"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    ('DenseNet121', 'feature_extractor'), ('DenseNet121', 'finetune'),
    ('Xception',    'feature_extractor'), ('Xception',    'finetune'),
    ('InceptionV3', 'feature_extractor'), ('InceptionV3', 'finetune'),
    ('ResNet50',    'feature_extractor'), ('ResNet50',    'finetune'),
    ('VGG16',       'feature_extractor'), ('VGG16',       'finetune'),
]
STRATEGY_LABEL = {'feature_extractor': 'PF', 'finetune': 'FT'}

# ============================================
# Generators — identical scaling to training (rescale 1/255), no shuffle
# ============================================
vt_datagen = ImageDataGenerator(rescale=1./255)

def make_gen(split):
    return vt_datagen.flow_from_directory(
        f"{DATASET_PATH}/{split}",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['benign', 'malignant'],   # benign = 0, malignant = 1
        shuffle=False
    )

val_gen  = make_gen('val')
test_gen = make_gen('test')
y_val    = val_gen.classes.astype(int)
y_test   = test_gen.classes.astype(int)
print(f"Val: {len(y_val)} ({y_val.sum()} malignant) | "
      f"Test: {len(y_test)} ({y_test.sum()} malignant)")

# ============================================
# Helpers
# ============================================
def youden_threshold(y, p):
    """Threshold maximizing Youden's J = sensitivity + specificity - 1.

    Candidate cut-points are midpoints between consecutive distinct scores
    (plus one below the minimum and one above the maximum), so the chosen
    threshold never sits exactly on a validation score. When several
    cut-points tie for the best J (e.g. a perfectly separating model has a
    whole gap of optimal thresholds), the median of the tied cut-points is
    used, i.e. the middle of the gap, rather than its edge.
    """
    u = np.unique(p)
    cuts = np.concatenate((
        [u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]))
    pos, neg = (y == 1), (y == 0)
    sens = np.array([(p[pos] >= c).mean() for c in cuts])
    spec = np.array([(p[neg] <  c).mean() for c in cuts])
    j = sens + spec - 1
    best = np.flatnonzero(np.isclose(j, j.max()))
    return float(np.median(cuts[best])), float(j.max())

def confusion(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return pred, tp, tn, fp, fn

def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d  = 1 + z*z/n
    c  = (ph + z*z/(2*n)) / d
    h  = z * np.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / d
    return (100*(c-h), 100*(c+h))

def metrics(y, p, thr):
    pred, tp, tn, fp, fn = confusion(y, p, thr)
    n = tp + tn + fp + fn
    acc_lo, acc_hi = wilson(tp + tn, n)
    return pred, {
        'Acc':  100*(tp+tn)/n,
        'Sen':  100*tp/(tp+fn) if tp+fn else np.nan,
        'Spe':  100*tn/(tn+fp) if tn+fp else np.nan,
        'Acc_CI': f"{acc_lo:.2f}-{acc_hi:.2f}",
        'FP': fp, 'FN': fn,
    }

def mcnemar_exact(y, pred_a, pred_b):
    """Exact two-sided McNemar test on paired correctness."""
    ok_a, ok_b = (pred_a == y), (pred_b == y)
    b = int((ok_a & ~ok_b).sum())   # A right, B wrong
    c = int((~ok_a & ok_b).sum())   # A wrong, B right
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return b, c, p

# ============================================
# Main loop
# ============================================
rows, per_case = [], {
    'split':    ['val']*len(y_val) + ['test']*len(y_test),
    'filename': list(val_gen.filenames) + list(test_gen.filenames),
    'label':    np.concatenate([y_val, y_test]),
}
test_preds_05, test_preds_val = {}, {}

for name, strategy in MODELS:
    tag  = f"{name}_{STRATEGY_LABEL[strategy]}"
    path = f"{SAVE_DIR}/models/{name}_{strategy}_best.keras"
    if not os.path.exists(path):
        print(f"MISSING: {path} — skipped")
        continue
    print(f"Scoring {tag} ...")

    model = tf.keras.models.load_model(path)
    val_gen.reset(); test_gen.reset()
    p_val  = model.predict(val_gen,  verbose=0).ravel()
    p_test = model.predict(test_gen, verbose=0).ravel()
    del model; gc.collect(); tf.keras.backend.clear_session()

    thr, j_val = youden_threshold(y_val, p_val)
    pred05, m05 = metrics(y_test, p_test, 0.5)
    predv,  mv  = metrics(y_test, p_test, thr)
    test_preds_05[tag], test_preds_val[tag] = pred05, predv
    per_case[f"{tag}_prob"] = np.concatenate([p_val, p_test])

    rows.append({
        'Backbone': name, 'Strategy': STRATEGY_LABEL[strategy],
        'Val_AUROC':  round(roc_auc_score(y_val,  p_val),  4),
        'Test_AUROC': round(roc_auc_score(y_test, p_test), 4),
        'Thr_val': round(thr, 4), 'Val_J': round(j_val, 4),
        **{f"{k}@0.5": (round(v, 2) if isinstance(v, float) else v)
           for k, v in m05.items()},
        **{f"{k}@val": (round(v, 2) if isinstance(v, float) else v)
           for k, v in mv.items()},
    })

df = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print("\n" + "="*100)
print("TEST RESULTS AT 0.5 vs. AT THE VALIDATION-TUNED THRESHOLD")
print("="*100)
print(df.to_string(index=False))
df.to_csv(f"{OUT_DIR}/threshold_rescoring.csv", index=False)

pd.DataFrame(per_case).to_csv(f"{OUT_DIR}/per_case_predictions.csv", index=False)
print(f"\nSaved: {OUT_DIR}/threshold_rescoring.csv")
print(f"Saved: {OUT_DIR}/per_case_predictions.csv  (release this with the paper)")

# ============================================
# McNemar: PF vs FT for each backbone, at 0.5 and at the tuned threshold
# ============================================
print("\n" + "="*100)
print("EXACT McNEMAR TEST, PF vs FT (test set)   b = PF right & FT wrong, c = PF wrong & FT right")
print("="*100)
mc_rows = []
for name, _ in MODELS[::2]:
    a, b_ = f"{name}_PF", f"{name}_FT"
    if a not in test_preds_05 or b_ not in test_preds_05:
        continue
    for label, preds in (('0.5', test_preds_05), ('val-tuned', test_preds_val)):
        b, c, p = mcnemar_exact(y_test, preds[a], preds[b_])
        mc_rows.append({'Backbone': name, 'Threshold': label,
                        'b': b, 'c': c, 'p_exact': round(p, 4)})
mc = pd.DataFrame(mc_rows)
print(mc.to_string(index=False))
mc.to_csv(f"{OUT_DIR}/mcnemar_pf_vs_ft.csv", index=False)

# ============================================
# LaTeX rows for Table III: threshold + Acc / Sen / Spe / FP / FN at that threshold
# ============================================
print("\n% LaTeX rows: Backbone & thr & Acc & Sen & Spe & FP & FN  (validation-tuned threshold)")
for strat in ('PF', 'FT'):
    print(f"% --- {strat} ---")
    for _, r in df[df.Strategy == strat].iterrows():
        print(f"{r.Backbone} & {r.Thr_val:.2f} & {r['Acc@val']:.2f} & "
              f"{r['Sen@val']:.2f} & {r['Spe@val']:.2f} & "
              f"{r['FP@val']} & {r['FN@val']} \\\\ \\hline")
