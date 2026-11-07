# ===== FOLLOW-UP experiments (second review). Requires cells 1-4 of MultiSeed_BUSI.ipynb to have run =====
# 1. native preprocessing (keras preprocess_input instead of [0,1]) for VGG16 and ResNet50, PF and FT, seeds 0-4
# 2. mask-derived silhouette control (silhouette built from the expert mask, not from non-zero pixels), seeds 0-2
# 3. whole-image control (B-mode image without mask or crop, padded to square), seeds 0-2
# 4. determinism check: VGG16-PF seed 0 trained twice with tf.config.experimental.enable_op_determinism()
# Results go to the same Drive folder (multiseed_v1/runs) with tag prefixes native_, masksil_, whole_, det_.
# Finished runs are skipped, so this cell can be re-run after a disconnect or crash.
import gc, zipfile
from pathlib import Path
from tensorflow.keras.applications import vgg16, resnet50

NATIVE = {"VGG16": vgg16.preprocess_input, "ResNet50": resnet50.preprocess_input}
_gens_01 = gens                                   # [0,1] generators from cell 3

def gens_native(fn):
    def g(root, seed, scale=None):
        aug = ImageDataGenerator(preprocessing_function=fn, horizontal_flip=True, vertical_flip=True, zoom_range=0.20,
                                 rotation_range=15, width_shift_range=0.10, height_shift_range=0.10, fill_mode='nearest')
        plain = ImageDataGenerator(preprocessing_function=fn)
        kw = dict(target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH, class_mode='binary', classes=['benign', 'malignant'])
        return (aug.flow_from_directory(f"{root}/train", shuffle=True, seed=seed, **kw),
                plain.flow_from_directory(f"{root}/val", shuffle=False, **kw),
                plain.flow_from_directory(f"{root}/test", shuffle=False, **kw))
    return g

def run(tag, backbone, strategy, seed, root=None, note="", native=False):
    global gens
    gens = gens_native(NATIVE[backbone]) if native else _gens_01
    try:
        return run_one(tag, backbone, strategy, seed, root=root, note=note)
    finally:
        gens = _gens_01
        gc.collect()

# ---- raw BUSI (same Kaggle source and primary masks as the original preprocessing) ----
RAW = "/content/BUSI_raw"
def raw_root():
    for p in Path(RAW).rglob("benign"):
        if p.is_dir() and any(p.glob("*_mask.png")):
            return p.parent
    return None
if raw_root() is None and os.path.exists("/content/drive/MyDrive/kaggle.json"):
    os.makedirs(RAW, exist_ok=True); os.makedirs("/root/.kaggle", exist_ok=True)
    shutil.copy("/content/drive/MyDrive/kaggle.json", "/root/.kaggle/kaggle.json"); os.chmod("/root/.kaggle/kaggle.json", 0o600)
    os.system(f"kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset -p {RAW} --force")
    with zipfile.ZipFile(f"{RAW}/breast-ultrasound-images-dataset.zip") as z:
        z.extractall(RAW)
RR = raw_root()
print("raw BUSI:", RR)

def square_resize(a):
    h, w = a.shape[:2]; m = max(h, w)
    sq = np.zeros((m, m) + a.shape[2:], dtype=np.uint8)
    sq[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w] = a
    return cv2.resize(sq, (IMG_SIZE, IMG_SIZE))

def mask_silhouette(mask_path, shape):
    """Same crop box, padding and resize as the paper's preprocessing, applied to the binary mask."""
    mask = cv2.resize(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE), (shape[1], shape[0]))
    _, b = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(b))
    px, py = int(w * 0.05), int(h * 0.05)
    crop = b[max(0, y - py):min(shape[0], y + h + py), max(0, x - px):min(shape[1], x + w + px)]
    s = square_resize(crop)
    s = ((s > 127) * 255).astype(np.uint8)
    return cv2.merge([s, s, s])

MASKSIL, WHOLE = f"{LOCAL}/BUSI_masksil", f"{LOCAL}/BUSI_whole"
if RR is not None and not os.path.exists(f"{WHOLE}/test/malignant"):
    import csv as _csv
    for r in _csv.DictReader(open(f"{OUT}/split.csv")):
        s, c, f = r["split"], r["class"], r["filename"]
        img = cv2.imread(str(RR / c / f))
        for d in (MASKSIL, WHOLE):
            os.makedirs(f"{d}/{s}/{c}", exist_ok=True)
        cv2.imwrite(f"{MASKSIL}/{s}/{c}/{f}", mask_silhouette(RR / c / f.replace(".png", "_mask.png"), img.shape))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # same channel handling as the original preprocessing
        cv2.imwrite(f"{WHOLE}/{s}/{c}/{f}", square_resize(rgb))
if RR is not None:
    for d in (MASKSIL, WHOLE):
        print(d, {s: sum(len(os.listdir(f"{d}/{s}/{c}")) for c in ("benign", "malignant")) for s in ("train", "val", "test")})

# ---- runs (most informative first) ----
for sd in range(5):
    run(f"native_VGG16_PF_seed{sd}", "VGG16", "PF", sd, note="native preprocess_input", native=True)
if RR is not None:
    for b in ("DenseNet121", "VGG16"):
        for sd in (0, 1, 2):
            run(f"masksil_{b}_PF_seed{sd}", b, "PF", sd, root=MASKSIL, note="mask-derived silhouette")
for sd in range(5):
    run(f"native_ResNet50_PF_seed{sd}", "ResNet50", "PF", sd, note="native preprocess_input", native=True)
for b in ("VGG16", "ResNet50"):
    for sd in range(5):
        run(f"native_{b}_FT_seed{sd}", b, "FT", sd, note="native preprocess_input", native=True)

# whole-image runs (moved after the native runs during the actual run; errors do not stop the cell)
if RR is not None:
    for b in ("DenseNet121", "VGG16"):
        for sd in (0, 1, 2):
            try:
                run(f"whole_{b}_PF_seed{sd}", b, "PF", sd, root=WHOLE, note="whole image, no mask")
            except Exception as ex:
                print("ERROR whole", b, sd, repr(ex))

# ---- determinism check last: op determinism stays on for the rest of the session ----
tf.config.experimental.enable_op_determinism()
for rep in ("a", "b"):
    run(f"det_VGG16_PF_seed0{rep}", "VGG16", "PF", 0, note="enable_op_determinism, same seed twice")
print("FOLLOWUP ALL DONE")
