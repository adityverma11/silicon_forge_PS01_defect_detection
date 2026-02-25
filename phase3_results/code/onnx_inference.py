import os
import time
import logging
import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG =================

BASE_DIR = r"D:\Academics\Hackathon NXP\PHASE 3\Phase 3 Final"

MODEL_PATH = os.path.join(BASE_DIR, "model.onnx")
DATA_PATH = os.path.join(BASE_DIR, "Training_split", "test")
TRAIN_PATH = os.path.join(BASE_DIR, "Training_split", "train")

IMG_SIZE = 128

# ================= LOGGING =================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "inference_log.txt")),
        logging.StreamHandler()
    ]
)

logging.info("Loading ONNX model")

# ================= SESSION =================

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
    if ort.get_device() == "GPU" else ["CPUExecutionProvider"]

session = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = session.get_inputs()[0].name

logging.info(f"Using providers: {providers}")

# ================= LOAD CLASSES =================

classes = sorted([
    d for d in os.listdir(TRAIN_PATH)
    if os.path.isdir(os.path.join(TRAIN_PATH, d))
])

logging.info(f"Classes: {classes}")

# ================= PREPROCESS =================

def preprocess(image_path):
    img = Image.open(image_path).convert("L")  # grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype(np.float32) / 255.0

    # expand grayscale to 3 channels
    img = np.stack([img, img, img], axis=-1)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)

# ================= INFERENCE =================

y_true = []
y_pred = []
latencies = []

logging.info("Starting inference...")

for class_name in sorted(os.listdir(DATA_PATH)):
    class_folder = os.path.join(DATA_PATH, class_name)

    if not os.path.isdir(class_folder):
        continue

    for file in sorted(os.listdir(class_folder)):

        image_path = os.path.join(class_folder, file)

        if not os.path.isfile(image_path):
            continue

        if not file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue

        input_tensor = preprocess(image_path)

        start = time.time()
        outputs = session.run(None, {input_name: input_tensor})
        end = time.time()

        latencies.append((end - start) * 1000)

        pred_class = int(np.argmax(outputs[0]))
        y_pred.append(pred_class)
        y_true.append(classes.index(class_name))

accuracy = np.mean(np.array(y_true) == np.array(y_pred))
avg_latency = np.mean(latencies)

logging.info(f"Accuracy: {accuracy}")
logging.info(f"Average latency per image (ms): {avg_latency}")

report = classification_report(y_true, y_pred, target_names=classes)
logging.info("\n" + report)

# ================= CONFUSION MATRIX =================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes,
    cmap="viridis"
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

logging.info("Confusion matrix saved")
logging.info("Inference completed successfully")