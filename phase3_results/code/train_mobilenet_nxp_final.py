import os, time, random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch.backends.cudnn as cudnn

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = r"D:\Academics\Hackathon NXP\PHASE 3\Phase 3 Final\Training_split"
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
IMG_SIZE = 128
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# =========================================================
# FIX RANDOMNESS
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True
cudnn.benchmark = False

# =========================================================
# TRANSFORMS (NO AUGMENTATION)
# =========================================================

train_tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================================================
# DATASETS
# =========================================================

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_tfms)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), test_tfms)

classes = train_ds.classes
num_classes = len(classes)
print("Classes:", classes)

# =========================================================
# CLASS-WEIGHTED LOSS
# =========================================================

targets = [y for _, y in train_ds]
class_count = np.bincount(targets)

class_weights = torch.tensor(1.0 / class_count, dtype=torch.float)
class_weights = class_weights / class_weights.sum() * len(class_count)
class_weights = class_weights.to(DEVICE)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0   # Windows-safe
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =========================================================
# MODEL (FULL FINE-TUNE)
# =========================================================

model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")

for param in model.features.parameters():
    param.requires_grad = True

in_features = model.classifier[0].in_features

model.classifier = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.Hardswish(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

model.to(DEVICE)

# =========================================================
# LOSS + OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# =========================================================
# TRAINING LOOP
# =========================================================

best_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct, total = 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        optimizer.step()

        running_loss += loss.item() * y.size(0)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    scheduler.step()

    epoch_loss = running_loss / total
    train_acc = 100 * correct / total

    print(f"Epoch {epoch+1:02d} | Loss {epoch_loss:.4f} | Train Acc {train_acc:.2f}%")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "best_model.pth")

# =========================================================
# TEST EVALUATION
# =========================================================

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

y_true, y_pred = [], []
latencies = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)

        start = time.time()
        out = model(x)
        end = time.time()

        latencies.append((end - start) / x.size(0) * 1000)

        preds = out.argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.numpy())

print("\nFINAL TEST ACCURACY:",
      np.mean(np.array(y_true) == np.array(y_pred)))

print("Avg Latency per image (ms):",
      np.mean(latencies))

print("\n", classification_report(y_true, y_pred, target_names=classes))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=classes,
            yticklabels=classes,
            cmap="viridis")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()