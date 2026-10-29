import os
import torch
import timm
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

# ==========================================
# SETTINGS
# ==========================================

DATASET_DIR = "dataset_v4"
MODEL_PATH = "deepfake_model_xception_v4.pth"

IMAGE_SIZE = 299
BATCH_SIZE = 8
EPOCHS = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

# ==========================================
# TRANSFORMS
# ==========================================

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

# ==========================================
# LOAD DATASET TWICE
# ==========================================

# Separate datasets are used so that
# training augmentation does NOT affect validation.

train_full = datasets.ImageFolder(
    DATASET_DIR,
    transform=train_transform
)

val_full = datasets.ImageFolder(
    DATASET_DIR,
    transform=val_transform
)

print("Classes found:", train_full.classes)
print("Total images:", len(train_full))

# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================

total_size = len(train_full)

train_size = int(0.8 * total_size)

val_size = total_size - train_size

generator = torch.Generator().manual_seed(42)

train_indices, val_indices = random_split(
    range(total_size),
    [train_size, val_size],
    generator=generator
)

# Create subsets with independent transforms

train_dataset = torch.utils.data.Subset(
    train_full,
    train_indices.indices
)

val_dataset = torch.utils.data.Subset(
    val_full,
    val_indices.indices
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# ==========================================
# DATA LOADERS
# ==========================================

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

# ==========================================
# LOAD XCEPTION
# ==========================================

print("\nLoading pretrained Xception...")

model = timm.create_model(
    "xception",
    pretrained=True,
    num_classes=2
)

model = model.to(DEVICE)

print("Xception model loaded.")

# ==========================================
# LOSS + OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.0001
)

# ==========================================
# TRAINING
# ==========================================

best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    print("\n================================")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("================================")

    # ------------------------------
    # TRAIN
    # ------------------------------

    model.train()

    running_loss = 0.0

    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predictions = torch.max(
            outputs,
            1
        )

        train_total += labels.size(0)

        train_correct += (
            predictions == labels
        ).sum().item()

    train_accuracy = (
        100 * train_correct / train_total
    )

    # ------------------------------
    # VALIDATION
    # ------------------------------

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

            _, predictions = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    val_accuracy = (
        100 * val_correct / val_total
    )

    average_loss = (
        running_loss / len(train_loader)
    )

    print(
        f"Training Loss: {average_loss:.4f}"
    )

    print(
        f"Training Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )

    # ------------------------------
    # SAVE BEST MODEL
    # ------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    train_full.classes
            },
            MODEL_PATH
        )

        print("Best model saved!")

# ==========================================
# FINAL RESULTS
# ==========================================

print("\n================================")
print("FINAL RESULTS")
print("================================")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)

print("\nClasses:")
print(train_full.classes)

print("\nModel saved as:")
print(MODEL_PATH)