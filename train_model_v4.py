import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "dataset"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.0001

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=======================================================")
print("VERSION 4 - TRANSFER LEARNING")
print("=======================================================")

print("Device:", device)


# ============================================================
# COLLECT ORIGINAL DATA
# ============================================================

image_paths = []
labels = []

for label_name, label in [
    ("real", 0),
    ("fake", 1)
]:

    folder = os.path.join(
        DATASET_DIR,
        label_name
    )

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ])

    print(
        f"{label_name.capitalize()} images: "
        f"{len(files)}"
    )

    for filename in files:

        image_paths.append(
            os.path.join(folder, filename)
        )

        labels.append(label)


image_paths = np.array(image_paths)
labels = np.array(labels)


print("\nTotal images:", len(image_paths))


# ============================================================
# CLEAN TRAIN / VALIDATION SPLIT
# ============================================================

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)


print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=8
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# DATASET CLASS
# ============================================================

class DeepfakeDataset(Dataset):

    def __init__(
        self,
        paths,
        labels,
        transform
    ):

        self.paths = paths
        self.labels = labels
        self.transform = transform


    def __len__(self):

        return len(self.paths)


    def __getitem__(self, index):

        path = self.paths[index]
        label = int(self.labels[index])

        image = Image.open(
            path
        ).convert("RGB")

        image = self.transform(
            image
        )

        return image, label


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = DeepfakeDataset(
    train_paths,
    train_labels,
    train_transform
)

val_dataset = DeepfakeDataset(
    val_paths,
    val_labels,
    val_transform
)


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


# ============================================================
# LOAD PRETRAINED RESNET18
# ============================================================

print("\nLoading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model = model.to(device)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\nStarting V4 training...")

best_accuracy = 0.0


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for images, targets in train_loader:

        images = images.to(device)
        targets = targets.to(device)


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            targets
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            predictions == targets
        ).sum().item()

        total += targets.size(0)


    train_accuracy = (
        correct / total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    all_targets = []
    all_predictions = []


    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(device)
            targets = targets.to(device)


            outputs = model(
                images
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_correct += (
                predictions == targets
            ).sum().item()

            val_total += targets.size(0)


            all_targets.extend(
                targets.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )


    val_accuracy = (
        val_correct / val_total
    )


    average_loss = (
        running_loss / total
    )


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "deepfake_model_v4.pth"
        )


# ============================================================
# FINAL VALIDATION RESULTS
# ============================================================

print("\n=======================================================")
print("VERSION 4 VALIDATION RESULTS")
print("=======================================================")

print(
    f"Best validation accuracy: "
    f"{best_accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        all_targets,
        all_predictions,
        target_names=[
            "Real",
            "Fake"
        ]
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        all_targets,
        all_predictions
    )
)


print("\nModel saved as:")
print("deepfake_model_v4.pth")

print("\nOriginal V1/V2/V3 models were NOT modified.")
print("External test_images were NOT used for training.")