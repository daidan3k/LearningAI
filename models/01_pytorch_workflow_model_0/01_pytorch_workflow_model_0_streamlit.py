import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt


# Create a linear regression model class
class LinearRegressionModel(
    nn.Module
):  # <- almost everything in PyTorch inherits from nn.module
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(
                1,  # <- start with a random weight
                requires_grad=True,  # <- can this parameter be updated via gradient descent?
                dtype=torch.float32,
            )
        )

        self.bias = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float32)
        )

    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # <- "x" is the input data
        return self.weight * x + self.bias  # regression formula


MODEL_SAVE_PATH = (
    "../../models/01_pytorch_workflow_model_0/01_pytorch_workflow_model_0_streamlit.pt"
)
model = LinearRegressionModel()
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

st.title("Predicción con modelo lineal")
# Entrada del usuario
input_val = st.number_input("Introduce un valor de X:", value=0.0)

# Cuando el usuario pulsa el botón
if st.button("Predecir"):
    # Convertir input a tensor
    x_tensor = torch.tensor([[input_val]], dtype=torch.float32)

    # Predicción
    with torch.inference_mode():
        y_pred = model(x_tensor).item()

        st.success(f"El modelo predice: {y_pred:.2f}")

        x_vals = torch.linspace(-10, 10, 100).unsqueeze(1)
        y_vals = model(x_vals).squeeze().numpy()

        x_vals_np = x_vals.squeeze().numpy()

        # Crear el gráfico
        fig, ax = plt.subplots()
        ax.plot(x_vals_np, y_vals, label="Modelo lineal")
        ax.scatter(input_val, y_pred, color="red", label="Predicción", zorder=5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Visualización de la predicción")
        ax.legend()

        st.pyplot(fig)
