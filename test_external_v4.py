import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =======================================================
# VERSION 4 - EXTERNAL TEST
# =======================================================

MODEL_PATH = "deepfake_model_v4.pth"
TEST_DIR = "test_images"

print("=" * 55)
print("VERSION 4 - EXTERNAL TEST")
print("=" * 55)

# -------------------------------------------------------
# Device
# -------------------------------------------------------
device = torch.device("cpu")
print("Device:", device)

# -------------------------------------------------------
# Image transformation
# Same preprocessing used for validation
# -------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------------
# Load ResNet18 architecture
# -------------------------------------------------------
print("\nLoading V4 model...")

model = models.resnet18(weights=None)

num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)

model.load_state_dict(torch.load(
    MODEL_PATH,
    map_location=device
))

model = model.to(device)
model.eval()

print("V4 model loaded successfully.")

# -------------------------------------------------------
# Test images
# -------------------------------------------------------
images = []
labels = []
filenames = []

class_info = {
    "real": 0,
    "fake": 1
}

for class_name, label in class_info.items():

    folder = os.path.join(TEST_DIR, class_name)

    if not os.path.exists(folder):
        print("Folder not found:", folder)
        continue

    for filename in sorted(os.listdir(folder)):

        filepath = os.path.join(folder, filename)

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            continue

        try:
            image = Image.open(filepath).convert("RGB")

            image_tensor = transform(image)
            images.append(image_tensor)
            labels.append(label)
            filenames.append(filename)

        except Exception as e:
            print("Error:", filename, e)

print("\nTotal images:", len(images))
print("Real images:", labels.count(0))
print("Fake images:", labels.count(1))

# -------------------------------------------------------
# Prediction
# -------------------------------------------------------
predictions = []
confidences = []

print("\nRunning predictions...")

with torch.no_grad():

    for image_tensor in images:

        image_tensor = image_tensor.unsqueeze(0).to(device)

        output = model(image_tensor)

        probabilities = torch.softmax(output, dim=1)

        confidence, prediction = torch.max(probabilities, dim=1)

        predictions.append(prediction.item())
        confidences.append(confidence.item() * 100)

# -------------------------------------------------------
# Results
# -------------------------------------------------------
accuracy = accuracy_score(labels, predictions)

print("\n" + "=" * 55)
print("V4 EXTERNAL TEST RESULTS")
print("=" * 55)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        labels,
        predictions,
        target_names=["Real", "Fake"],
        digits=2
    )
)

print("Confusion Matrix:")

print(confusion_matrix(labels, predictions))

# -------------------------------------------------------
# Class-wise accuracy
# -------------------------------------------------------
cm = confusion_matrix(labels, predictions)

real_correct = cm[0][0]
real_total = cm[0][0] + cm[0][1]

fake_correct = cm[1][1]
fake_total = cm[1][0] + cm[1][1]

print("\nClass-wise Results:")

print(
    f"Real: {real_correct}/{real_total} "
    f"({real_correct / real_total * 100:.2f}%)"
)

print(
    f"Fake: {fake_correct}/{fake_total} "
    f"({fake_correct / fake_total * 100:.2f}%)"
)

# -------------------------------------------------------
# Individual predictions
# -------------------------------------------------------
print("\nIndividual Predictions:")

for filename, true_label, prediction, confidence in zip(
    filenames,
    labels,
    predictions,
    confidences
):

    actual = "Real" if true_label == 0 else "Fake"
    predicted = "Real" if prediction == 0 else "Fake"

    result = "CORRECT" if true_label == prediction else "WRONG"

    print(
        f"{filename:20} | "
        f"Actual: {actual:5} | "
        f"Predicted: {predicted:5} | "
        f"{confidence:6.2f}% | {result}"
    )

print("\n" + "=" * 55)
print("TEST COMPLETE")
print("=" * 55)