import os
import torch
import timm
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, accuracy_score

# =========================
# SETTINGS
# =========================

DATASET_DIR = "dataset"
MODEL_PATH = "deepfake_model_xception.pth"

BATCH_SIZE = 8
EPOCHS = 5
IMAGE_SIZE = 299

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# =========================
# TRANSFORMS
# =========================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD DATASET
# =========================

full_dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=train_transform
)

print("Classes found:", full_dataset.classes)
print("Total images:", len(full_dataset))

# IMPORTANT:
# ImageFolder normally sorts alphabetically.
# fake = 0
# real = 1

# 80% training, 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Validation should NOT use augmentation
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =========================
# CREATE XCEPTION
# =========================

print("Loading pretrained Xception...")

model = timm.create_model(
    "xception",
    pretrained=True,
    num_classes=2
)

model = model.to(DEVICE)

print("Xception model loaded.")

# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.0001
)

# =========================
# TRAINING
# =========================

best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    print("\n==============================")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("==============================")

    # -------------------------
    # TRAIN
    # -------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    val_accuracy = 100 * val_correct / val_total

    print(f"Training Loss: {running_loss / len(train_loader):.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # -------------------------
    # SAVE BEST MODEL
    # -------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": full_dataset.classes
            },
            MODEL_PATH
        )

        print("Best Xception model saved!")

# =========================
# CONFUSION MATRIX
# =========================

print("\n==============================")
print("FINAL RESULTS")
print("==============================")

print("Best Validation Accuracy:",
      f"{best_val_accuracy:.2f}%")

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)

print("\nModel saved as:")
print(MODEL_PATH)