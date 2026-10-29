# Digital Image Processing Based Deepfake Detection System

A BCA mini-project exploring digital image processing and deep learning techniques for preliminary detection of potentially manipulated images.

## Project Overview

This project uses image processing techniques and a deep learning model to analyze uploaded images through a Flask web application.

The application extracts image properties and generates a classification result using ResNet18.

## Features

- Image upload and validation
- RGB and grayscale image processing
- Image resizing and edge detection
- Image dimensions and megapixel calculation
- Brightness, contrast, sharpness, and edge-density analysis
- SHA-256 hash generation
- Deep learning-based image classification
- Web-based analysis report

## Technologies Used

- Python
- Flask
- OpenCV
- PyTorch
- ResNet18
- Xception
- NumPy
- HTML
- CSS

## Models

**ResNet18:** Primary model integrated into the application.

**Xception:** Evaluated separately for model comparison.

## Project Workflow

1. Upload an image.
2. Validate the image.
3. Apply image preprocessing.
4. Extract image properties.
5. Generate a classification prediction using ResNet18.
6. Display the analysis results.

## Repository Contents

- `app.py` — Flask application
- `templates/` — HTML templates
- `static/` — CSS styling
- `train_model_v4.py` — ResNet18 training script
- `train_model_xception.py` — Xception training script
- `train_xception_v4.py` — Additional Xception training script
- Testing scripts — External model evaluation
- `requirements.txt` — Python dependencies

## How to Run

1. Install Python.
2. Clone this repository.
3. Install the required dependencies using `pip install -r requirements.txt`.
4. Place the compatible trained model file in the location expected by the application.
5. Run the application using `python app.py`.
6. Open the local Flask address displayed in the terminal.

## Limitations

This project is an academic prototype. Its predictions are not definitive forensic proof of whether an image is authentic or manipulated. Performance may vary across datasets and unseen images.

## Project Information

- **Course:** Bachelor of Computer Applications (BCA)
- **Semester:** V Semester
- **Academic Year:** 2026–2027
- **Domain:** Digital Image Processing and Deep Learning

## Contributors

- Harini R
- Monika P
