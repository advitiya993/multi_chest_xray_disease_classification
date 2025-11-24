import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import io

# ----------- Load Model -----------
@st.cache_resource
def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 3)
    model.load_state_dict(torch.load("best_model_v2(1).pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ----------- Preprocess Image -----------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ----------- UI -----------
st.title("🩺 Disease Classification App")
st.write("Upload an image to get the predicted label.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        pred = model(img_tensor)
        label = torch.argmax(pred, dim=1).item()

    classes = ['Atelectasis', 'COVID', 'Cardiomegaly', 'Effusion', 'Mass', 'No_Finding']

    st.success(f"Predicted Class: **{classes[label]}**")
