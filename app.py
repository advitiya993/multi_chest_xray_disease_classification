import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------------------
# Load Model (Matches Training)
# ---------------------------
@st.cache_resource
def load_model():
    num_classes = 3  # <-- CHANGE if your project has different classes

    # SAME architecture you used during training
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    # Load weights safely (handles DataParallel issues)
    state_dict = torch.load("best_model_v2 (1).pth", map_location="cpu")

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # remove DataParallel prefix if present
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)

    model.eval()
    return model


model = load_model()

# ---------------------------
# Preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🩺 Chest X-Ray Disease Classification")
st.write("Upload a chest X-ray image to predict disease class.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()

    # Your class names
    classes = ["Class A", "Class B", "Class C"]
    st.success(f"Prediction: **{classes[pred]}**")
