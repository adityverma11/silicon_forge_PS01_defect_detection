import os
import numpy as np
import onnxruntime as ort
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
import logging

# Configuration

DATA_DIR = "hackathon_test_dataset"
MODEL_PATH = "model.onnx"
IMG_SIZE = 160
BATCH_SIZE = 32
LOG_FILE = "phase2_inference_log.txt"

# Logging 

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

print("Loading ONNX model...")
logging.info("Loading ONNX model")

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# Transform 

val_tfms = transforms.Compose([
    transforms.Resize(
        (IMG_SIZE, IMG_SIZE),
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Dataset

test_ds = datasets.ImageFolder(DATA_DIR, val_tfms)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Original dataset classes:", test_ds.classes)
logging.info(f"Original dataset classes: {test_ds.classes}")

# Trained classes
trained_classes = [
    'Bridge', 'Clean', 'Crack', 'LER',
    'Open', 'Other', 'Scratch', 'Vias'
]

# Inference

y_true = []
y_pred = []
latencies = []

print("Running inference...")
logging.info("Starting inference without TTA")

for images, labels in test_loader:

    images_np = images.numpy()

    start = time.time()

    # Only original logits
    logits = session.run(None, {input_name: images_np})[0]

    end = time.time()

    preds = np.argmax(logits, axis=1)

    y_pred.extend(preds)
    y_true.extend(labels.numpy())
    latencies.append((end - start) / images.size(0) * 1000)

# Particle -> Other mapping

class_to_idx = test_ds.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

mapped_y_true = []

for label in y_true:
    class_name = idx_to_class[label]

    if class_name == "Particle":
        mapped_y_true.append(trained_classes.index("Other"))
    elif class_name in trained_classes:
        mapped_y_true.append(trained_classes.index(class_name))
    else:
        mapped_y_true.append(trained_classes.index("Other"))

mapped_y_true = np.array(mapped_y_true)
y_pred = np.array(y_pred)

# metrics

accuracy = np.mean(mapped_y_true == y_pred)
avg_latency = np.mean(latencies)

print("\nFINAL TEST ACCURACY:", accuracy)
print("Avg Latency per image (ms):", avg_latency)

logging.info(f"Accuracy: {accuracy}")
logging.info(f"Avg Latency per image (ms): {avg_latency}")

report = classification_report(
    mapped_y_true,
    y_pred,
    labels=range(len(trained_classes)),
    target_names=trained_classes,
    zero_division=0
)

print("\n", report)
logging.info("\n" + report)

# Confusion Matrix 

cm = confusion_matrix(
    mapped_y_true,
    y_pred,
    labels=range(len(trained_classes))
)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=trained_classes,
    yticklabels=trained_classes,
    cmap="viridis"
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Phase-2 Confusion Matrix (Particle merged to Other)")
plt.tight_layout()
plt.savefig("phase2_confusion_matrix.png")
plt.close()

print("Confusion matrix saved as phase2_confusion_matrix.png")
logging.info("Confusion matrix saved")

print("Log file saved as phase2_inference_log.txt")
logging.info("Phase-2 inference completed successfully")
