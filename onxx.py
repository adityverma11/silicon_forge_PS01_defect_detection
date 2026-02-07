import os, time, random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn, optim
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch.backends.cudnn as cudnn

#  CONFIG

DATA_DIR = "dataset"
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
PATIENCE = 6
IMG_SIZE = 160
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

#  FIX RANDOMNESS 

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

cudnn.deterministic = True
cudnn.benchmark = False

g = torch.Generator()
g.manual_seed(SEED)

# TRANSFORMS 

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

#  DATASETS 

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_tfms)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), val_tfms)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), val_tfms)

classes = train_ds.classes
num_classes = len(classes)
print("Classes:", classes)

# WEIGHTED SAMPLER 

targets = [y for _, y in train_ds]
class_count = np.bincount(targets)
class_weights = 1. / class_count
sample_weights = [class_weights[t] for t in targets]

sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    generator=g
)

val_loader  = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

#  MODEL 

model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    num_classes
)
model.to(DEVICE)

#  LOSS + OPTIMIZER 

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    patience=2,
    factor=0.3
)

# TRAINING LOOP 

best_val = 0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = 100 * correct / total

    #  VALIDATION 
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = 100 * correct / total
    scheduler.step(val_acc)

    print(f"Epoch {epoch+1:02d} | Train {train_acc:.2f}% | Val {val_acc:.2f}%")

    if val_acc > best_val:
        best_val = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# TEST EVALUATION 

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

#  METRICS 

print("\nFINAL TEST ACCURACY:",
      np.mean(np.array(y_true) == np.array(y_pred)))

print("Avg Latency per image (ms):",
      np.mean(latencies))

print("\n", classification_report(y_true, y_pred, target_names=classes))

#  CONFUSION MATRIX 

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

# EXPORT TO ONNX 

model.eval()
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(" Model exported to ONNX format as 'model.onnx'")
