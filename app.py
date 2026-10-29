
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    send_file,
    abort
)

import os
import uuid
from datetime import datetime

import cv2
import numpy as np

import torch
from PIL import Image

from torchvision import models, transforms

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as ReportImage,
    Table,
    TableStyle
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# ALLOWED FILES
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# LOAD RESNET18 V4 MODEL
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "deepfake_model_v4.pth"
)


model = models.resnet18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    2
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)


model.to(device)

model.eval()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

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
# MODEL PREDICTION
# ============================================================

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)


    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]


    predicted_class = int(
        torch.argmax(probabilities).item()
    )


    confidence = float(
        probabilities[predicted_class].item()
        * 100
    )


    if predicted_class == 0:

        label = "Likely Authentic"

    else:

        label = "Potentially Manipulated"


    return label, round(confidence, 2)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image_path):

    image = cv2.imread(
        image_path
    )


    if image is None:

        raise ValueError(
            "Unable to read uploaded image."
        )


    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    height, width = image.shape[:2]


    # --------------------------------------------------------
    # Megapixels
    # --------------------------------------------------------

    megapixels = round(
        (width * height) / 1000000,
        2
    )


    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness = round(
        float(np.mean(gray)),
        2
    )


    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast = round(
        float(np.std(gray)),
        2
    )


    # --------------------------------------------------------
    # Sharpness
    # Variance of Laplacian
    # --------------------------------------------------------

    sharpness = round(
        float(cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()),
        2
    )


    # --------------------------------------------------------
    # Canny Edge Detection
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        100,
        200
    )


    # --------------------------------------------------------
    # Edge Density
    # --------------------------------------------------------

    edge_density = round(
        float(np.mean(edges > 0)),
        4
    )


    return {
        "width": width,
        "height": height,
        "megapixels": megapixels,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "edge_density": edge_density
    }


# ============================================================
# CREATE GRAYSCALE + EDGE IMAGES
# ============================================================

def process_image(image_path, original_filename):

    image = cv2.imread(
        image_path
    )


    if image is None:

        raise ValueError(
            "Unable to process uploaded image."
        )


    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Canny Edge Detection
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        100,
        200
    )


    # --------------------------------------------------------
    # Output filenames
    # --------------------------------------------------------

    gray_filename = (
        "gray_" + original_filename
    )

    edge_filename = (
        "edge_" + original_filename
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Save processed images in PROCESSED_FOLDER
    # --------------------------------------------------------

    gray_path = os.path.join(
        PROCESSED_FOLDER,
        gray_filename
    )


    edge_path = os.path.join(
        PROCESSED_FOLDER,
        edge_filename
    )


    # --------------------------------------------------------
    # Save images
    # --------------------------------------------------------

    gray_success = cv2.imwrite(
        gray_path,
        gray
    )


    edge_success = cv2.imwrite(
        edge_path,
        edges
    )


    if not gray_success:

        raise ValueError(
            "Could not save grayscale image."
        )


    if not edge_success:

        raise ValueError(
            "Could not save edge detection image."
        )


    return {
        "gray_filename": gray_filename,
        "edge_filename": edge_filename
    }


# ============================================================
# GENERATE PDF REPORT
# ============================================================

def create_pdf_report(
    original_path,
    filename,
    label,
    confidence,
    analysis
):

    report_filename = (
        os.path.splitext(filename)[0]
        + "_analysis_report.pdf"
    )


    report_path = os.path.join(
        REPORT_FOLDER,
        report_filename
    )


    document = SimpleDocTemplate(
        report_path,
        pagesize=A4,

        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )


    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=23,
        leading=28,
        textColor=colors.HexColor(
            "#4B35A8"
        ),
        spaceAfter=10
    )


    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor(
            "#777777"
        ),
        spaceAfter=20
    )


    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor(
            "#4B35A8"
        ),
        spaceBefore=15,
        spaceAfter=8
    )


    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        leading=14
    )


    story = []


    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "Digital Image Forensics",
            title_style
        )
    )


    story.append(
        Paragraph(
            "Deepfake Detection Analysis Report",
            subtitle_style
        )
    )


    story.append(
        Paragraph(
            "Generated: "
            + datetime.now().strftime(
                "%d %B %Y, %I:%M %p"
            ),
            body_style
        )
    )


    story.append(
        Spacer(1, 15)
    )


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    story.append(
        Paragraph(
            "Analyzed Image",
            heading_style
        )
    )


    try:

        report_image = ReportImage(
            original_path,
            width=5.5 * inch,
            height=4.0 * inch
        )

        report_image.hAlign = "CENTER"

        story.append(
            report_image
        )

    except Exception:

        story.append(
            Paragraph(
                "Original image could not be embedded.",
                body_style
            )
        )


    story.append(
        Spacer(1, 15)
    )


    # ========================================================
    # DETECTION RESULT
    # ========================================================

    story.append(
        Paragraph(
            "Detection Result",
            heading_style
        )
    )


    result_data = [

        ["Classification", label],

        ["Model", "ResNet18 CNN"],

        [
            "Confidence",
            str(confidence) + "%"
        ],

    ]


    result_table = Table(
        result_data,
        colWidths=[
            2.0 * inch,
            4.2 * inch
        ]
    )


    result_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor(
                    "#EEE9FF"
                )
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (0, -1),
                colors.HexColor(
                    "#4B35A8"
                )
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                "Helvetica"
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor(
                    "#DDDDDD"
                )
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])
    )


    story.append(
        result_table
    )


    # ========================================================
    # IMAGE METRICS
    # ========================================================

    story.append(
        Paragraph(
            "Image Analysis Metrics",
            heading_style
        )
    )


    metrics_data = [

        [
            "Metric",
            "Value"
        ],

        [
            "Dimensions",
            f"{analysis['width']} × "
            f"{analysis['height']}"
        ],

        [
            "Megapixels",
            str(analysis["megapixels"])
        ],

        [
            "Brightness",
            str(analysis["brightness"])
        ],

        [
            "Contrast",
            str(analysis["contrast"])
        ],

        [
            "Sharpness",
            str(analysis["sharpness"])
        ],

        [
            "Edge Density",
            str(analysis["edge_density"])
        ]

    ]


    metrics_table = Table(
        metrics_data,
        colWidths=[
            2.0 * inch,
            4.2 * inch
        ]
    )


    metrics_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    "#4B35A8"
                )
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor(
                    "#DDDDDD"
                )
            ),

            (
                "FONTNAME",
                (0, 1),
                (-1, -1),
                "Helvetica"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )

        ])
    )


    story.append(
        metrics_table
    )


    # ========================================================
    # PROCESSING PIPELINE
    # ========================================================

    story.append(
        Paragraph(
            "Processing Pipeline",
            heading_style
        )
    )


    pipeline_text = (
        "1. Image Upload<br/>"
        "2. Image Validation<br/>"
        "3. Grayscale Conversion<br/>"
        "4. Canny Edge Detection<br/>"
        "5. Image Feature Analysis<br/>"
        "6. Resize to 224 × 224 pixels<br/>"
        "7. ImageNet Normalization<br/>"
        "8. ResNet18 CNN Prediction"
    )


    story.append(
        Paragraph(
            pipeline_text,
            body_style
        )
    )


    story.append(
        Spacer(1, 15)
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "Model Information",
            heading_style
        )
    )


    model_text = (
        "The system uses a ResNet18 convolutional neural "
        "network with transfer learning for binary image "
        "classification. The model classifies an uploaded "
        "image as either Likely Authentic or Potentially "
        "Manipulated."
    )


    story.append(
        Paragraph(
            model_text,
            body_style
        )
    )


    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(story)


    return report_filename


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# ANALYZE IMAGE
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if "image" not in request.files:

        return (
            "No image file uploaded.",
            400
        )


    file = request.files["image"]


    if file.filename == "":

        return (
            "No image selected.",
            400
        )


    if not allowed_file(
        file.filename
    ):

        return (
            "Invalid file format. "
            "Please upload JPG, JPEG or PNG.",
            400
        )


    # --------------------------------------------------------
    # Generate safe unique filename
    # --------------------------------------------------------

    extension = os.path.splitext(
        file.filename
    )[1].lower()


    filename = (
        uuid.uuid4().hex
        + extension
    )


    upload_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    file.save(
        upload_path
    )


    try:

        # ====================================================
        # IMAGE ANALYSIS
        # ====================================================

        analysis = analyze_image(
            upload_path
        )


        # ====================================================
        # CREATE GRAYSCALE + EDGE IMAGES
        # ====================================================

        processed = process_image(
            upload_path,
            filename
        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        label, confidence = predict_image(
            upload_path
        )


        # ====================================================
        # CREATE PDF REPORT
        # ====================================================

        report_filename = create_pdf_report(
            upload_path,
            filename,
            label,
            confidence,
            analysis
        )


        # ====================================================
        # RENDER RESULT
        # ====================================================

        return render_template(
            "index.html",

            filename=filename,

            label=label,

            confidence=confidence,

            analysis=analysis,

            gray_filename=processed[
                "gray_filename"
            ],

            edge_filename=processed[
                "edge_filename"
            ],

            report_filename=report_filename
        )


    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )


        return (
            f"Error while analyzing image: {str(e)}",
            500
        )


# ============================================================
# SERVE UPLOADED ORIGINAL IMAGE
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ============================================================
# SERVE PROCESSED IMAGES
# ============================================================

@app.route(
    "/processed/<filename>"
)
def processed_file(filename):

    file_path = os.path.join(
        PROCESSED_FOLDER,
        filename
    )


    if not os.path.exists(
        file_path
    ):

        abort(404)


    return send_from_directory(
        PROCESSED_FOLDER,
        filename
    )


# ============================================================
# VIEW ANALYSIS REPORT
# ============================================================

@app.route(
    "/report/<filename>"
)
def analysis_report(filename):

    report_filename = (
        os.path.splitext(filename)[0]
        + "_analysis_report.pdf"
    )


    report_path = os.path.join(
        REPORT_FOLDER,
        report_filename
    )


    if not os.path.exists(
        report_path
    ):

        abort(404)


    return send_file(
        report_path,
        mimetype="application/pdf"
    )


# ============================================================
# DOWNLOAD PDF REPORT
# ============================================================

@app.route(
    "/download_report/<filename>"
)
def download_report(filename):

    report_filename = (
        os.path.splitext(filename)[0]
        + "_analysis_report.pdf"
    )


    report_path = os.path.join(
        REPORT_FOLDER,
        report_filename
    )


    if not os.path.exists(
        report_path
    ):

        abort(404)


    return send_file(
        report_path,

        as_attachment=True,

        download_name=report_filename,

        mimetype="application/pdf"
    )


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def too_large(error):

    return (
        "File is too large. "
        "Maximum allowed size is 10 MB.",
        413
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )

