import json
import numpy as np
from PIL import Image

import tritonclient.http as httpclient
from tritonclient.http import InferInput

import torchvision.transforms as transforms


# =========================================
# CONFIG
# =========================================

TRITON_URL = "localhost:8000"

MODEL_NAME = "plant_disease_classifier"

IMAGE_PATH = r"TestImages\test1.JPG"


# =========================================
# LOAD CLASS MAPPINGS
# =========================================

with open("idx_to_class.json", "r") as f:
    idx_to_class = json.load(f)

print("Class mappings loaded.")


# =========================================
# IMAGE TRANSFORMS
# =========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================
# LOAD IMAGE
# =========================================

image = Image.open(IMAGE_PATH).convert("RGB")

image_tensor = transform(image)

# Add batch dimension
image_tensor = image_tensor.unsqueeze(0)

print("Image preprocessing complete.")
print("Tensor Shape:", image_tensor.shape)


# =========================================
# CONVERT TO NUMPY FP32
# =========================================

input_data = image_tensor.numpy().astype(np.float32)


# =========================================
# CREATE TRITON CLIENT
# =========================================

client = httpclient.InferenceServerClient(
    url=TRITON_URL
)

print("Connected to Triton server.")


# =========================================
# CREATE INPUT TENSOR
# =========================================

inputs = []

triton_input = InferInput(
    "input",
    input_data.shape,
    "FP32"
)

triton_input.set_data_from_numpy(input_data)

inputs.append(triton_input)

print("Input tensor prepared.")


# =========================================
# SEND INFERENCE REQUEST
# =========================================

response = client.infer(
    model_name=MODEL_NAME,
    inputs=inputs
)

print("Inference request successful.")


# =========================================
# GET OUTPUT
# =========================================

output = response.as_numpy("output")

print("Output Shape:", output.shape)


# =========================================
# GET PREDICTION
# =========================================

predicted_index = np.argmax(output, axis=1)[0]

predicted_label = idx_to_class[str(predicted_index)]

confidence = float(np.max(output))

print("\n===== PREDICTION =====")

print(f"Predicted Index : {predicted_index}")
print(f"Predicted Label : {predicted_label}")
print(f"Confidence Score: {confidence:.4f}")