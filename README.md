# AI Crop Disease Prediction (Streamlit + TensorFlow)

This project is an AI-powered web app to predict crop diseases from leaf images using a TensorFlow/Keras model.

## Features
- Upload a leaf image (JPG/PNG/JPEG)
- Preprocess image to the required input size
- Predict disease class using the trained model
- Display the predicted disease label

## Project Structure
- `app.py` - Streamlit application (image upload + prediction UI)
- `train.py` - Training script (model training)
- `predict.py` - Standalone prediction script (if provided/used)
- `plant_disease_model.keras` - Trained model file (expected by `app.py`)
- `requirements.txt` - Python dependencies

## Prerequisites
- Python 3.8+ recommended
- A trained model file: `plant_disease_model.keras`

## Installation
1. Create a virtual environment (recommended)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App
From the project directory:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (typically `http://localhost:8501`).

## How It Works
1. User uploads a leaf image.
2. The image is resized to **224x224**.
3. Pixel values are normalized to `[0, 1]`.
4. The model predicts probabilities for each class.
5. The class with the highest probability is shown.

## Classes (Tomato)
- `Tomato_Early_Blight`
- `Tomato_Healthy`
- `Tomato_Late_Blight`

## Notes
- Ensure `plant_disease_model.keras` is in the same directory as `app.py`.
- If you retrain the model or change class labels, update the `classes` list in `app.py` accordingly.

"# AI-crop-Disease-prediction" 
