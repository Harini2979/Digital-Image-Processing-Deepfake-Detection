import os
import torch
import timm
from torchvision import transforms
from PIL import Image

# -----------------------------
# SETTINGS
# -----------------------------

MODEL_PATH = "deepfake_model_xception.pth"
TEST_DIR = "test_images"

IMAGE_SIZE = 299
DEVICE = torch.device("cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------

print("Loading Xception model...")

model = timm.create_model(
    "xception",
    pretrained=False,
    num_classes=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])

class_names = checkpoint["class_names"]

model = model.to(DEVICE)
model.eval()

print("Classes:", class_names)
print("Model loaded successfully.")

# -----------------------------
# IMAGE TRANSFORMATION
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# TEST IMAGES
# -----------------------------

correct = 0
total = 0

fake_correct = 0
fake_total = 0

real_correct = 0
real_total = 0

for class_name in ["fake", "real"]:

    class_folder = os.path.join(TEST_DIR, class_name)

    if not os.path.exists(class_folder):
        print("Folder not found:", class_folder)
        continue

    print("\nTesting:", class_name)

    for filename in os.listdir(class_folder):

        image_path = os.path.join(class_folder, filename)

        try:
            image = Image.open(image_path).convert("RGB")

            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(DEVICE)

            with torch.no_grad():

                outputs = model(image_tensor)

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                confidence, prediction = torch.max(
                    probabilities,
                    dim=1
                )

            predicted_class = class_names[prediction.item()]
            confidence = confidence.item() * 100

            total += 1

            if predicted_class == class_name:
                correct += 1

                if class_name == "fake":
                    fake_correct += 1

                else:
                    real_correct += 1

            if class_name == "fake":
                fake_total += 1

            else:
                real_total += 1

            print(
                filename,
                "→",
                predicted_class,
                f"({confidence:.2f}%)"
            )

        except Exception as e:
            print("Error:", filename, e)

# -----------------------------
# RESULTS
# -----------------------------

print("\n==============================")
print("XCEPTION EXTERNAL TEST")
print("==============================")

if total > 0:

    accuracy = (correct / total) * 100

    print(f"Total images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Overall Accuracy: {accuracy:.2f}%")

if fake_total > 0:

    fake_accuracy = (fake_correct / fake_total) * 100

    print(
        f"Fake Accuracy: {fake_accuracy:.2f}%"
    )

if real_total > 0:

    real_accuracy = (real_correct / real_total) * 100

    print(
        f"Real Accuracy: {real_accuracy:.2f}%"
    )