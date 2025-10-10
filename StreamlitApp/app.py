# app.py
import streamlit as st
import os, json, io
import gdown
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# ---------------- CONFIG ----------------
# Replace these with your Drive file IDs (from the shareable links)
MODEL_DRIVE_ID = "1hrDdf2FuuTlt6XMvSPMtZj2GV_OuJVNC"
CLASSES_DRIVE_ID = None  # optionally put the ID for classes.json; set to None if classes.json in repo

# Local names used by the app
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_LOCAL = os.path.join(MODELS_DIR, "EfficientNet-B7.pth")
CLASSES_LOCAL = os.path.join(MODELS_DIR, "classes.json")

# If you included classes.json in repo, copy it to models/ or set CLASSES_DRIVE_ID=None

# ---------------- UTILITIES ----------------
def download_drive_file(drive_id, out_path):
    url = f"https://drive.google.com/uc?id={drive_id}"
    # gdown will skip download if file exists by default only if path exists; we guard
    gdown.download(url, out_path, quiet=False)
    return out_path

@st.cache_resource
def load_classes():
    # try repo-local first
    if os.path.exists(CLASSES_LOCAL):
        with open(CLASSES_LOCAL, "r") as f:
            return json.load(f)
    if CLASSES_DRIVE_ID:
        download_drive_file(CLASSES_DRIVE_ID, CLASSES_LOCAL)
        with open(CLASSES_LOCAL, "r") as f:
            return json.load(f)
    st.error("classes.json not found locally. Put classes.json in repo/models/ or set CLASSES_DRIVE_ID.")
    st.stop()

@st.cache_resource
def load_model():
    # download model if missing
    if not os.path.exists(MODEL_LOCAL):
        if not MODEL_DRIVE_ID:
            st.error("MODEL_DRIVE_ID not set and model file not found.")
            st.stop()
        download_drive_file(MODEL_DRIVE_ID, MODEL_LOCAL)
    # build architecture (must match training)
    classes = load_classes()
    model = timm.create_model('tf_efficientnet_b7', pretrained=False, num_classes=len(classes))
    state = torch.load(MODEL_LOCAL, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

# ---------------- PREPROCESS (match Colab val_transform) ----------------
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),   # no normalization because training val_transform used only ToTensor
])

def predict(img_pil, model):
    x = preprocess(img_pil).unsqueeze(0)  # shape (1,C,H,W)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return probs

# ---------------- Grad-CAM helper ----------------
def find_last_conv(module):
    last = None
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last = (name, m)
    return last

def gradcam_overlay(model, img_pil, target_class=None, upsample_size=(224,224)):
    # returns overlay (PIL RGB), pred_idx, prob
    model.eval()
    name_conv, conv_mod = find_last_conv(model)
    if conv_mod is None:
        raise RuntimeError("No conv layer found for Grad-CAM")
    activations = {}
    def fhook(module, inp, out):
        out.retain_grad()
        activations['feat'] = out
    hook = conv_mod.register_forward_hook(fhook)

    x = preprocess(img_pil).unsqueeze(0)
    out = model(x)
    probs = torch.softmax(out, dim=1)
    pred = int(probs.argmax(dim=1).item())
    if target_class is None:
        target_class = pred

    score = out[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)

    feat = activations['feat']               # (1,C,H,W)
    grad = feat.grad
    if grad is None:
        grad = torch.autograd.grad(score, feat, retain_graph=False)[0]

    weights = grad.mean(dim=(2,3), keepdim=True)
    cam = (weights * feat).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, upsample_size)
    img_np = np.array(img_pil.resize(upsample_size))[:,:,::-1]  # BGR
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + img_np * 0.6
    hook.remove()
    overlay_rgb = overlay[:,:,::-1].astype(np.uint8)
    return Image.fromarray(overlay_rgb), pred, float(probs[0, target_class].item())

# ---------------- UI ----------------
st.title("🧠 Brain Tumor MRI Classifier — EfficientNet-B7")
st.write("Upload an MRI image (jpg/png).")

# load model & classes (cached)
classes = load_classes()
model = load_model()

uploaded = st.file_uploader("Upload MRI image", type=["jpg","jpeg","png"])
if uploaded is None:
    st.info("Upload an image to predict.")
else:
    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(img, caption="Input Image", use_column_width=True)

    probs = predict(img, model)
    topk = 4 if len(classes) >= 4 else len(classes)
    top_idx = np.argsort(probs)[::-1][:topk]

    with col2:
        st.subheader("Prediction")
        pred_idx = int(top_idx[0])
        st.success(f"{classes[pred_idx]}  —  {probs[pred_idx]:.4f}")
        st.markdown("**All class probabilities**")
        for i in top_idx:
            st.write(f"- {classes[int(i)]}: {probs[int(i)]:.4f}")

        # show a bar chart of full probability vector
        st.bar_chart({classes[i]: float(probs[i]) for i in range(len(classes))})

    # Grad-CAM on demand
    if st.button("Show Grad-CAM overlay"):
        try:
            overlay_img, pred_idx2, conf = gradcam_overlay(model, img)
            st.image(overlay_img, caption=f"Grad-CAM (pred={classes[pred_idx2]} conf={conf:.3f})", use_column_width=True)
        except Exception as e:
            st.error(f"Grad-CAM failed: {e}")

# footer
st.info("Model loaded from Drive. If you want faster startup, store classes.json in repo/models/ and keep the model in a fast host.")