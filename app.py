"""
Skin Cancer Detection Model
============================
Dataset: jaiahuja/skin-cancer-detection (Kaggle)
Model  : Transfer Learning with EfficientNetB0
Task   : Binary Classification — Benign vs Malignant
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
class Config:
    # ✅ UPDATE THIS PATH to where your dataset lives
    DATA_DIR = "C:/Users/Mannadithya/Downloads/skin_cancer/Skin cancer ISIC The International Skin Imaging Collaboration/Train"
    IMAGE_SIZE   = 224
    BATCH_SIZE   = 32
    EPOCHS       = 25
    LR           = 3e-5
    WEIGHT_DECAY = 1e-5
    NUM_CLASSES  = 9         # benign / malignant
    NUM_WORKERS  = 2
    SEED         = 42

    # class names (folder names inside DATA_DIR)
    CLASSES = [
        "actinic keratosis",
        "basal cell carcinoma",
        "dermatofibroma",
        "melanoma",
        "nevus",
        "pigmented benign keratosis",
        "seborrheic keratosis",
        "squamous cell carcinoma",
        "vascular lesion"
    ]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_PATH = "best_skin_cancer_model.pth"

cfg = Config()
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
print(f"Using device: {cfg.DEVICE}")


# ─────────────────────────────────────────────
# 2. DATASET HELPER  (supports two layouts)
# ─────────────────────────────────────────────
def discover_images(data_dir: str, classes: list):
    """
    Supports two common layouts:
      A) data_dir/
           train/benign/*.jpg   train/malignant/*.jpg
           test/benign/*.jpg    test/malignant/*.jpg

      B) data_dir/
           benign/*.jpg
           malignant/*.jpg
    Returns a DataFrame with columns [path, label, class, split].
    """
    data_dir = Path(data_dir)
    records  = []

    # Layout A — pre-split folders
    for split in ["train", "test", "valid", "val", "Train", "Test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            for lbl, cls in enumerate(classes):
                for variant in [cls, cls.capitalize(), cls.upper()]:
                    cls_dir = split_dir / variant
                    if cls_dir.exists():
                        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                            for img in cls_dir.glob(ext):
                                records.append({
                                    "path": str(img), "label": lbl,
                                    "class": cls,
                                    "split": "val" if split in ("valid", "val") else split
                                })
                        break

    # Layout B — flat class folders
    if not records:
        for lbl, cls in enumerate(classes):
            for variant in [cls, cls.capitalize(), cls.upper()]:
                cls_dir = data_dir / variant
                if cls_dir.exists():
                    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                        for img in cls_dir.glob(ext):
                            records.append({
                                "path": str(img), "label": lbl,
                                "class": cls, "split": None
                            })
                    break

    if not records:
        raise FileNotFoundError(
            f"No images found in '{data_dir}'.\n"
            "Please update cfg.DATA_DIR to the correct folder.\n"
            "Expected sub-folders: benign/ and malignant/\n"
            "  OR: train/benign/, train/malignant/, test/benign/, test/malignant/"
        )

    df = pd.DataFrame(records)
    print(f"\nDiscovered {len(df)} images total:")
    print(df.groupby(["split", "class"]).size().to_string())
    return df


class SkinCancerDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


# ─────────────────────────────────────────────
# 3. TRANSFORMS
# ─────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# 4. BUILD DATALOADERS
# ─────────────────────────────────────────────
def build_loaders(data_dir):
    df = discover_images(data_dir, cfg.CLASSES)

    # Use pre-defined splits if available, else create 70 / 15 / 15
    if df["split"].notna().any():
        train_df = df[df["split"] == "train"]
        test_df  = df[df["split"] == "test"]
        val_df   = df[df["split"].isin(["val"])]
        if val_df.empty:
            train_df, val_df = train_test_split(
                train_df, test_size=0.15, stratify=train_df["label"],
                random_state=cfg.SEED)
    else:
        train_df, tmp = train_test_split(
            df, test_size=0.30, stratify=df["label"], random_state=cfg.SEED)
        val_df, test_df = train_test_split(
            tmp, test_size=0.50, stratify=tmp["label"], random_state=cfg.SEED)

    print(f"\nSplit sizes  →  train: {len(train_df)}  |  val: {len(val_df)}  |  test: {len(test_df)}")

    # Compute class weights to handle class imbalance
    counts = np.bincount(train_df["label"], minlength=cfg.NUM_CLASSES)
    class_weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float)
    class_weights = (class_weights / class_weights.sum() * len(counts)).to(cfg.DEVICE)

    loaders = {
        "train": DataLoader(
            SkinCancerDataset(train_df, train_transforms),
            batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS, pin_memory=True),
        "val": DataLoader(
            SkinCancerDataset(val_df, val_transforms),
            batch_size=cfg.BATCH_SIZE, shuffle=False,
            num_workers=cfg.NUM_WORKERS, pin_memory=True),
        "test": DataLoader(
            SkinCancerDataset(test_df, val_transforms),
            batch_size=cfg.BATCH_SIZE, shuffle=False,
            num_workers=cfg.NUM_WORKERS, pin_memory=True),
    }
    return loaders, class_weights


# ─────────────────────────────────────────────
# 5. MODEL  (EfficientNetB0 + custom classifier head)
# ─────────────────────────────────────────────
def build_model():
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    # Freeze backbone; fine-tune last 3 blocks
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-5:].parameters():
        param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, cfg.NUM_CLASSES)   # ✅ 9 classes
    )
    return model.to(cfg.DEVICE)


# ─────────────────────────────────────────────
# 6. TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(cfg.DEVICE == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds    = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        probs    = torch.softmax(outputs, dim=1)   # shape [batch, 9]
        preds    = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())   # ← must be INSIDE loop, AFTER extend probs

    # AUC calculated AFTER loop, once all data is collected
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(all_labels, classes=list(range(cfg.NUM_CLASSES)))
    auc = roc_auc_score(y_bin, np.array(all_probs), multi_class="ovr", average="macro")
    return total_loss / total, correct / total, auc

def train(model, loaders, class_weights):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    scaler    = torch.cuda.amp.GradScaler(enabled=(cfg.DEVICE == "cuda"))

    history  = {"train_loss": [], "val_loss": [],
                "train_acc":  [], "val_acc":  [], "val_auc": []}
    best_auc = 0.0

    print("\n" + "=" * 62)
    print("  Training EfficientNetB0 — Skin Cancer Detection")
    print("=" * 62)

    for epoch in range(1, cfg.EPOCHS + 1):
        tr_loss, tr_acc          = train_one_epoch(model, loaders["train"],
                                                   criterion, optimizer, scaler)
        vl_loss, vl_acc, vl_auc = evaluate(model, loaders["val"], criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_auc"].append(vl_auc)

        tag = ""
        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(), cfg.SAVE_PATH)
            tag = "  ✅ saved"

        print(f"Epoch {epoch:02d}/{cfg.EPOCHS}  |  "
              f"train loss {tr_loss:.4f}  acc {tr_acc:.4f}  |  "
              f"val loss {vl_loss:.4f}  acc {vl_acc:.4f}  auc {vl_auc:.4f}{tag}")

    print(f"\nBest Validation AUC : {best_auc:.4f}")
    print(f"Model saved to      : {cfg.SAVE_PATH}")
    return history


# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
def plot_training(history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].plot(history["val_auc"], color="green")
    axes[2].set_title("Val AUC"); axes[2].set_xlabel("Epoch")

    plt.suptitle("Skin Cancer Detection — Training Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved: training_curves.png")
    plt.show()


# ─────────────────────────────────────────────
# 8. FULL TEST EVALUATION
# ─────────────────────────────────────────────
@torch.no_grad()
def full_test_evaluation(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images  = images.to(cfg.DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)   # shape: [batch, 9]
        preds   = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    print("\n" + "=" * 62)
    print("  TEST SET RESULTS")
    print("=" * 62)
    print(classification_report(all_labels, all_preds,
                                 target_names=cfg.CLASSES, digits=4))
    


    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(all_labels, classes=list(range(cfg.NUM_CLASSES)))
    auc = roc_auc_score(y_bin, np.array(all_probs), multi_class="ovr", average="macro")
    print(f"ROC-AUC Score : {auc:.4f}\n")


    # ── Confusion Matrix ──────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cfg.CLASSES, yticklabels=cfg.CLASSES)
    plt.title("Confusion Matrix — Test Set")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved: confusion_matrix.png")
    plt.show()

    # ── ROC Curve ────────────────────────────────────────────────
    plt.figure(figsize=(6, 5))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Skin Cancer Detection")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    print("Saved: roc_curve.png")
    plt.show()

    return auc


# ─────────────────────────────────────────────
# 9. SINGLE IMAGE INFERENCE
# ─────────────────────────────────────────────
def predict_single(model, image_path: str):
    """Run inference on a single image and print result."""
    model.eval()
    img    = Image.open(image_path).convert("RGB")
    tensor = val_transforms(img).unsqueeze(0).to(cfg.DEVICE)

    with torch.no_grad():
        output = model(tensor)
        prob   = torch.softmax(output, dim=1).cpu().numpy()[0]

    pred_class = cfg.CLASSES[prob.argmax()]

    # Show image with prediction
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    title_color = "red" if pred_class == "malignant" else "green"
    plt.title(
        f"Prediction: {pred_class.upper()}\n"
        f"Benign: {prob[0]*100:.1f}%  |  Malignant: {prob[1]*100:.1f}%",
        color=title_color, fontsize=11
    )
    plt.tight_layout()
    plt.savefig("single_prediction.png", dpi=150)
    plt.show()

    print(f"\nPrediction for: {image_path}")
    for cls, p in zip(cfg.CLASSES, prob):
        bar = "█" * int(p * 30)
        print(f"  {cls:12s} {p*100:6.2f}%  {bar}")
    print(f"\n  → RESULT: {pred_class.upper()}")
    return pred_class, prob


# ─────────────────────────────────────────────
# 10. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── Step 1: Build data loaders ───────────────────────────────
    loaders, class_weights = build_loaders(cfg.DATA_DIR)

    # ── Step 2: Build model ──────────────────────────────────────
    model = build_model()
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel     : EfficientNetB0")
    print(f"Parameters: {total:,} total  |  {trainable:,} trainable")

    # ── Step 3: Train ────────────────────────────────────────────
    history = train(model, loaders, class_weights)
    plot_training(history)

    # ── Step 4: Test (load best checkpoint) ─────────────────────
    model.load_state_dict(torch.load(cfg.SAVE_PATH, map_location=cfg.DEVICE))
    full_test_evaluation(model, loaders["test"])

    # ── Step 5: Predict on a single image ───────────────────────
    # Uncomment and set your path to run inference on one image:
    # predict_single(model, "path/to/your/lesion_image.jpg")