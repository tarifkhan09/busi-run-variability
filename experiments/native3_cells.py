# NATIVE-3 (third review): native keras preprocess_input for Xception, InceptionV3 and DenseNet121, PF and FT, seeds 0-4.
# Runs cells 1-4 of MultiSeed_BUSI.ipynb first. Same data, split, head and training settings as the main runs; only the
# input normalization changes. Op determinism is NOT enabled (same as the main and native_ runs).
# Results go to multiseed_v1/runs/native_<backbone>_<strategy>_seed<k>. Finished runs are skipped, so re-run after a crash.
from google.colab import drive
drive.mount("/content/drive")
import json, gc
NB = "/content/drive/MyDrive/BreastCancerNew/multiseed_v1/MultiSeed_BUSI.ipynb"
CS = [''.join(c["source"]) for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
for i in (0, 1, 2, 3):
    exec(CS[i], globals())
from tensorflow.keras.applications import xception, inception_v3, densenet

NATIVE = {"Xception": xception.preprocess_input,          # scales to [-1, 1]
          "InceptionV3": inception_v3.preprocess_input,   # scales to [-1, 1]
          "DenseNet121": densenet.preprocess_input}       # torch mode: /255, ImageNet mean/std
_gens_01 = gens

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

def run(tag, backbone, strategy, seed, note=""):
    global gens
    gens = gens_native(NATIVE[backbone])
    try:
        return run_one(tag, backbone, strategy, seed, note=note)
    finally:
        gens = _gens_01
        gc.collect()

for strategy in ("PF", "FT"):
    for b in ("Xception", "InceptionV3", "DenseNet121"):
        for sd in range(5):
            try:
                run(f"native_{b}_{strategy}_seed{sd}", b, strategy, sd, note="native preprocess_input")
            except Exception as ex:
                print("ERROR", b, strategy, sd, repr(ex))
print("NATIVE3 ALL DONE")
