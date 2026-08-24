# EfficientNetB3 Texture Classification

A Streamlit-based image classification application using a fine-tuned EfficientNetB3 model to classify images into 47 texture categories.

## Model

- Architecture: EfficientNetB3
- Input size: 300 × 300 × 3
- Number of classes: 47
- Framework: TensorFlow / Keras
- Image processing: OpenCV
- Frontend: Streamlit

## Features

- Upload texture images
- OpenCV-based preprocessing
- EfficientNetB3 inference
- Predicted texture class
- Prediction confidence
- Top-5 predictions

## Run Locally

Create and activate a virtual environment:

```bash
python -m venv eff