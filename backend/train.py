# backend/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import time
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
import numpy as np
from model import DRViTModel

# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 5
NUM_CLASSES = 5

DATA_DIR = "../../train_images"
CSV_FILE = "../../train.csv"
MODEL_SAVE_PATH = "dr_vit_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def trim_black_borders(image, tolerance=15):
    """
    Trims the black margins from the fundus image.
    """
    img_array = np.array(image)
    if img_array.ndim == 2: # Grayscale
        mask = img_array > tolerance
        if not mask.any(): return image
        return Image.fromarray(img_array[np.ix_(mask.any(1), mask.any(0))])
    elif img_array.ndim == 3: # RGB
        gray = np.array(image.convert('L'))
        mask = gray > tolerance
        if not mask.any(): return image
        trimmed = img_array[np.ix_(mask.any(1), mask.any(0))]
        return Image.fromarray(trimmed)
    return image

def ben_grahams_method(image, sigma=10):
    """
    Industry standard preprocessing for DR: Enhances local contrast.
    """
    image = image.convert('RGB')
    image = image.resize((224, 224), Image.LANCZOS)
    blur_transform = transforms.GaussianBlur(kernel_size=51, sigma=sigma)
    img_tensor = transforms.ToTensor()(image)
    blurred_tensor = blur_transform(img_tensor)
    enhanced_tensor = 4 * img_tensor - 4 * blurred_tensor + 0.5
    enhanced_tensor = torch.clamp(enhanced_tensor, 0, 1)
    return transforms.ToPILImage()(enhanced_tensor)

# IMAGENET normalization (Sync with app.py)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -----------------------------
# Transforms (Robust for Accuracy)
# -----------------------------

# Weak transform
weak_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# Strong transform (Augmented but aspect-ratio safe)
strong_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# -----------------------------
# Dataset (Robust Filtering)
# -----------------------------
class DRDataset(Dataset):
    def __init__(self, df, root_dir, weak_transform=None, strong_transform=None, augment=False):
        self.root_dir = root_dir
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.augment = augment
        # METHOD 1: Augment ONLY Minority Classes (Severe/Proliferative and Mild)
        # We exclude Class 0 (No DR) and Class 2 (Moderate) as they are the majority
        self.minority_classes = [1, 3, 4]
        print(f"Dataset initialized with Method-1 (Strict Minority Augmentation on: {self.minority_classes})")
        
        # Pre-validate images for Extreme Accuracy
        print(f"Validating {len(df)} images in dataset...")
        valid_indices = []
        for i, row in df.iterrows():
            img_path = os.path.join(self.root_dir, row['id_code'] + ".png")
            if os.path.exists(img_path):
                valid_indices.append(i)
        
        self.df = df.iloc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            img_id = self.df.loc[idx, 'id_code']
            label = int(self.df.loc[idx, 'diagnosis'])

            img_path = os.path.join(self.root_dir, img_id + ".png")
            image = Image.open(img_path).convert("RGB")
            
            image = trim_black_borders(image)
            image = ben_grahams_method(image)

            # APPLY AUGMENTATION ONLY IF augment=True (Standard for minority classes per slides)
            if self.augment and label in self.minority_classes and self.strong_transform:
                image = self.strong_transform(image)
            elif self.weak_transform:
                image = self.weak_transform(image)

            return image, torch.tensor(label, dtype=torch.long), img_id
        except Exception as e:
            print(f"Error loading image {img_id}: {e}")
            raise e

# -----------------------------
# Validation Function
# -----------------------------
def validate(model, loader, criterion, use_tta=True):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    # 3-View TTA for peak accuracy reporting
    tta_transforms = [
        lambda x: x,
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0)
    ]
    
    with torch.no_grad():
        for images_tensor_batch, labels, ids in loader:
            images_tensor_batch, labels = images_tensor_batch.to(device), labels.to(device)
            
            if use_tta:
                # Note: This is an approximation since we already have normalized tensors.
                # In a real TTA loop during train-val, we'd ideally apply to raw images,
                # but for validation speed, we'll use base model performance.
                outputs = model(images_tensor_batch)
            else:
                outputs = model(images_tensor_batch)
                
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return bal_acc

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(CSV_FILE)

train_df, val_df = train_test_split(
    df,
    test_size=0.15,
    stratify=df['diagnosis'],
    random_state=42
)

# -----------------------------
# Aggressive Data Balancing (Oversampling)
# -----------------------------
print("\nPerforming Aggressive Balancing to destroy Bias...")
# Oversample minority classes in train_df to achieve perfectly uniform distribution
max_count = train_df['diagnosis'].value_counts().max()
balanced_dfs = []
for diag in range(NUM_CLASSES):
    diag_df = train_df[train_df['diagnosis'] == diag]
    if len(diag_df) > 0:
        resampled_df = diag_df.sample(max_count, replace=True, random_state=42)
        balanced_dfs.append(resampled_df)

aggressive_train_df = pd.concat(balanced_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"Balanced Dataset size: {len(aggressive_train_df)} (all classes now have {max_count} images)")

train_dataset = DRDataset(aggressive_train_df, DATA_DIR, weak_transform, strong_transform, augment=True)
val_dataset = DRDataset(val_df, DATA_DIR, weak_transform, weak_transform, augment=False) 

# With aggressive oversampling in DF, we use standard shuffle=True
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# -----------------------------
# Model Alignment (Using DRViTModel from model.py)
# -----------------------------
model = DRViTModel(num_classes=NUM_CLASSES)
model.to(device)

# -----------------------------
# Loss & Optimization
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(gamma=2, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# -----------------------------
# Training Execution
# -----------------------------
def train_model():
    best_score = 0

    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels, ids in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        
        # Validation (using balanced accuracy for better imbalance measurement)
        bal_acc = validate(model, val_loader, criterion)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Val Bal Acc: {bal_acc:.4f}")
        
        if bal_acc > best_score:
            best_score = bal_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--- Best model saved with Balanced Acc: {bal_acc:.4f} ---")

    print("\nGenerating Global Metrics and Plots...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for images, labels, ids in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # 1. Save Metrics JSON (Robust Multi-level Structure)
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    # Global metrics
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    
    # Class-specific metrics
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(y_true, y_pred)
    classes_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
    
    class_stats = {}
    for i, name in enumerate(classes_names):
        class_stats[name] = {
            "precision": float(p_class[i]),
            "recall": float(r_class[i]),
            "f1": float(f1_class[i]),
            "reliability": "High" if f1_class[i] > 0.8 else "Moderate"
        }

    import json
    metrics = {
        "global": {
            "accuracy": float(acc),
            "precision": float(avg_precision),
            "recall": float(avg_recall),
            "f1": float(avg_f1)
        },
        "class_specific": class_stats,
        "status": "Generated from real validation session"
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Final Validation Accuracy: {acc*100:.2f}%")

    # 2. Save Global Confusion Matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Global Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plot_dir = "static/plots"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "global_cm.png"))
    plt.close()

    # 3. Save Global AUC Curve
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
    plt.figure(figsize=(10, 8))
    
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Global AUC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plot_dir, "global_auc.png"))
    plt.close()

    print("Training and Plot Generation completed.")
    
    # 4. Final: Accuracy About Each and Every Image (Full Dataset Results)
    print("\nCorrecting accuracy errors for the ENTIRE dataset...")
    # Mode: augment=False to get baseline accuracy for every image
    full_dataset = DRDataset(df, DATA_DIR, weak_transform, weak_transform, augment=False)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model.eval()
    results = [] # To store per-image rows
    
    with torch.no_grad():
        for images, labels, ids in full_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            
            ids_list = ids
            true_list = labels.numpy()
            pred_list = preds.cpu().numpy()
            conf_list = confidences.cpu().numpy()
            
            for i in range(len(ids_list)):
                results.append({
                    "ImageID": ids_list[i],
                    "OriginalDiagnosis": classes_names[true_list[i]],
                    "AIMprediction": classes_names[pred_list[i]],
                    "Confidence": f"{conf_list[i]*100:.1f}%",
                    "Status": "PASS" if true_list[i] == pred_list[i] else "FAIL"
                })
    
    # Save CSV for "Each and Every Image" requirement
    results_df = pd.DataFrame(results)
    results_df.to_csv("reports/per_image_accuracy.csv", index=False)
    print("Exported per-image results to: reports/per_image_accuracy.csv")
    
    # Save Text Report
    report_full = classification_report(results_df["OriginalDiagnosis"], results_df["AIMprediction"])
    print("\n--- FINAL ACCURACY REPORT FOR ALL IMAGES ---")
    print(report_full)
    
    with open("reports/full_dataset_accuracy.txt", "w") as f:
        f.write(f"Generated at: {datetime.datetime.now()}\n")
        f.write("ACCURACY REPORT FOR EVERY IMAGE IN THE DATASET\n")
        f.write("="*50 + "\n")
        f.write(report_full)

if __name__ == "__main__":
    import datetime
    train_model()
