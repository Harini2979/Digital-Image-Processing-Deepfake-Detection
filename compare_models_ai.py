import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import timm

# ==========================================
# SETTINGS
# ==========================================

AI_FOLDER = "ai_test_images"

RESNET_MODEL_PATH = "deepfake_model_v4.pth"
XCEPTION_MODEL_PATH = "deepfake_model_xception_v4.pth"

DEVICE = torch.device("cpu")

# ==========================================
# IMAGE TRANSFORMS
# ==========================================

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

xception_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# LOAD RESNET18
# ==========================================

print("\nLoading ResNet18...")

resnet = models.resnet18(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, 2)

resnet.load_state_dict(
    torch.load(
        RESNET_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )
)

resnet.to(DEVICE)
resnet.eval()

print("ResNet18 loaded.")

# ==========================================
# LOAD XCEPTION V4
# ==========================================

print("\nLoading Xception v4...")

xception = timm.create_model(
    "xception",
    pretrained=False,
    num_classes=2
)

# Load checkpoint
xception_checkpoint = torch.load(
    XCEPTION_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

# The model was saved inside "model_state_dict"
xception.load_state_dict(
    xception_checkpoint["model_state_dict"]
)

xception.to(DEVICE)
xception.eval()

print("Xception v4 loaded.")

# ==========================================
# CLASS NAMES
# ==========================================

class_names = ["fake", "real"]

# ==========================================
# FIND IMAGES
# ==========================================

if not os.path.exists(AI_FOLDER):
    print(f"\nERROR: Folder '{AI_FOLDER}' not found.")
    exit()

image_files = [
    f for f in os.listdir(AI_FOLDER)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    )
]

image_files.sort()

print("\n==========================================")
print("AI-GENERATED IMAGE MODEL COMPARISON")
print("==========================================")
print(f"Images found: {len(image_files)}")

# ==========================================
# RESULTS
# ==========================================

resnet_fake = 0
resnet_real = 0

xception_fake = 0
xception_real = 0

agreement_count = 0

# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict(model, image, transform):

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(output, dim=1)

    confidence, predicted = torch.max(
        probabilities,
        1
    )

    predicted_class = class_names[predicted.item()]

    confidence_percent = confidence.item() * 100

    return predicted_class, confidence_percent


# ==========================================
# TEST EACH IMAGE
# ==========================================

for filename in image_files:

    image_path = os.path.join(
        AI_FOLDER,
        filename
    )

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        # -------------------------------
        # ResNet18
        # -------------------------------

        resnet_class, resnet_conf = predict(
            resnet,
            image,
            resnet_transform
        )

        # -------------------------------
        # Xception v4
        # -------------------------------

        xception_class, xception_conf = predict(
            xception,
            image,
            xception_transform
        )

        # -------------------------------
        # Count predictions
        # -------------------------------

        if resnet_class == "fake":
            resnet_fake += 1
        else:
            resnet_real += 1

        if xception_class == "fake":
            xception_fake += 1
        else:
            xception_real += 1

        # -------------------------------
        # Agreement
        # -------------------------------

        if resnet_class == xception_class:

            agreement = "BOTH MODELS AGREE"

            agreement_count += 1

        else:

            agreement = "MODELS DISAGREE"

        # -------------------------------
        # Display result
        # -------------------------------

        print("\n------------------------------------------")

        print(
            f"Image: {filename}"
        )

        print(
            f"ResNet18  → "
            f"{resnet_class.upper()} "
            f"({resnet_conf:.2f}%)"
        )

        print(
            f"Xception  → "
            f"{xception_class.upper()} "
            f"({xception_conf:.2f}%)"
        )

        print(
            f"Agreement  → {agreement}"
        )

    except Exception as e:

        print("\n------------------------------------------")

        print(
            f"Error processing: {filename}"
        )

        print(
            f"Error: {e}"
        )


# ==========================================
# FINAL RESULTS
# ==========================================

total_images = len(image_files)

print("\n")
print("==========================================")
print("FINAL MODEL COMPARISON")
print("==========================================")

print("\nResNet18:")

print(
    f"Potentially Manipulated: "
    f"{resnet_fake}/{total_images}"
)

print(
    f"Likely Authentic: "
    f"{resnet_real}/{total_images}"
)

print("\nXception v4:")

print(
    f"Potentially Manipulated: "
    f"{xception_fake}/{total_images}"
)

print(
    f"Likely Authentic: "
    f"{xception_real}/{total_images}"
)

print("\n==========================================")
print("MODEL AGREEMENT")
print("==========================================")

print(
    f"Both models agree on "
    f"{agreement_count}/{total_images} images."
)

if total_images > 0:

    agreement_percentage = (
        agreement_count /
        total_images
    ) * 100

    print(
        f"Agreement percentage: "
        f"{agreement_percentage:.2f}%"
    )

print("\n==========================================")
print("TEST COMPLETE")
print("==========================================")