import io
import json

import numpy as np
from PIL import Image


import tritonclient.http as httpclient
from tritonclient.http import InferInput


# =========================================
# CONFIG
# =========================================

import os

TRITON_URL = os.getenv(
    "TRITON_URL",
    "localhost:8000"
)

MODEL_NAME = "plant_disease_classifier"


# =========================================
# LOAD CLASS MAPPINGS
# =========================================

with open("idx_to_class.json", "r") as f:
    idx_to_class = json.load(f)


# =========================================
# IMAGE TRANSFORMS
# =========================================

def preprocess_image(image):

    # Resize
    image = image.resize((224, 224))

    # Convert to numpy
    image = np.array(image).astype(np.float32) / 255.0

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (image - mean) / std

    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image.astype(np.float32)


# =========================================
# CREATE TRITON CLIENT
# =========================================

client = httpclient.InferenceServerClient(
    url=TRITON_URL
)


# =========================================
# PREDICTION FUNCTION
# =========================================

def predict_image(img_bytes):

    # Load image
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Preprocess
    input_data = preprocess_image(image)
    
    # Create Triton input
    triton_input = InferInput(
        "input",
        input_data.shape,
        "FP32"
    )

    triton_input.set_data_from_numpy(input_data)

    # Run inference
    response = client.infer(
        model_name=MODEL_NAME,
        inputs=[triton_input]
    )

    # Get output
    output = response.as_numpy("output")

    # Prediction index
    predicted_index = np.argmax(output, axis=1)[0]

    # Convert index -> label
    predicted_label = idx_to_class[str(predicted_index)]

    return predicted_label