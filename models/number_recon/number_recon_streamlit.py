import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import os
from streamlit_drawable_canvas import st_canvas
import cv2


class MNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):

        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7, out_features=output_shape),
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


model = MNISTModelV1(input_shape=1, hidden_units=10, output_shape=10).to("cpu")

MODEL_SAVE_PATH = os.path.join(
    os.getcwd(), "models", "number_recon/digit_recon_model_v1.pt"
)

# MODEL_SAVE_PATH = os.path.join(
#    "/home/daidan/Desktop/Documents/Code/LearningAI/models/number_recon/digit_recon_model_v1.pt"
# )

model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
model.eval()


# Streamlit page setup
st.set_page_config(page_title="Draw a Number", layout="centered")

st.title("Draw a Number (28x28 Canvas)")

# Create two columns: one for the canvas and one for the grayscale preview
col1, col2 = st.columns(2)

with col1:
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="white",  # Fill color for new shapes
        stroke_width=25,  # Stroke width
        stroke_color="black",  # Stroke color
        background_color="white",
        width=280,  # 10x scaling for easier drawing
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

if canvas_result.image_data is not None:
    # Downscale the image to 28x28
    img = canvas_result.image_data
    img = cv2.cvtColor(
        img.astype(np.uint8), cv2.COLOR_RGBA2GRAY
    )  # Convert to grayscale

    # Threshold the image to create a binary image
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours to determine the bounding box of the digit
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])  # Get bounding box
        digit = img[y : y + h, x : x + w]  # Crop the digit

        # Resize the digit to fit within a 20x20 box
        digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

        # Create a 28x28 canvas and center the digit
        mnist_like_img = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - 20) // 2
        y_offset = (28 - 20) // 2
        mnist_like_img[y_offset : y_offset + 20, x_offset : x_offset + 20] = digit

        # Normalize the image for the model
        normalized_img = mnist_like_img / 255.0

        with col2:
            # Display the MNIST-like image
            st.image(mnist_like_img, width=280, caption="Centered and Resized Image")

        # Prepare the image for the model
        input_tensor = (
            torch.tensor(normalized_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )

        # Perform inference
        with torch.no_grad():
            logits = model(input_tensor)
            pred_probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(pred_probs)

        # Display the prediction
        st.success(f"The model predicts: **{pred.item()}**")

        # Display the prediction probabilities as a bar chart
        st.markdown("### Prediction Probabilities")
        prob_chart_data = {f"{i}": pred_probs[0][i].item() for i in range(10)}
        st.bar_chart(prob_chart_data)
