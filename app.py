import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sqlite3
import datetime
import pandas as pd
import os
import warnings

# Suppress specific matplotlib UserWarnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Streamlit Page Config
st.set_page_config(page_title="Image Forgery Detector", layout="wide")

# 2. Model Definition
class ImageForgeryDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.efficient_net._fc.in_features
        self.efficient_net._fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.efficient_net(x)

# 3. Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# 4. Heatmap Sliding Window
def generate_heatmap(image, model, window_size=224, stride=112):
    width, height = image.size
    heatmap = np.zeros((height, width))
    counts = np.zeros((height, width))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            window = image.crop((x, y, x + window_size, y + window_size))
            window_tensor = transform(window).unsqueeze(0).to(model.device)

            with torch.no_grad():
                score = model(window_tensor).item()

            heatmap[y:y + window_size, x:x + window_size] += score
            counts[y:y + window_size, x:x + window_size] += 1

    heatmap = np.divide(heatmap, counts, where=counts != 0)
    heatmap = np.nan_to_num(heatmap, nan=0.0)
    heatmap = np.clip(heatmap, 0.0, 1.0)  # Clip to avoid overflow in plotting
    return heatmap

# 5. Cache model loading so it loads once
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageForgeryDetector()
    model.load_state_dict(torch.load("casia2_forgery_model.pt", map_location=device))
    model.to(device)
    model.eval()
    model.device = device
    return model

# 6. SQLite DB setup
def init_db():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            model_score REAL,
            threshold REAL,
            predicted_label TEXT,
            true_label TEXT
        )
    ''')
    conn.commit()
    return conn, c

def insert_prediction(c, conn, timestamp, filename, score, threshold, predicted_label, true_label):
    c.execute('''
        INSERT INTO predictions (timestamp, filename, model_score, threshold, predicted_label, true_label)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, filename, score, threshold, predicted_label, true_label))
    conn.commit()

def fetch_all_predictions(c):
    c.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = c.fetchall()
    return rows

def clear_db(c, conn):
    c.execute("DELETE FROM predictions")
    conn.commit()

# 7. Main App
def main():
    st.title("🧠 Image Forgery Detection with EfficientNet-B0")

    model = load_model()

    conn, c = init_db()

    # Reset DB button
    if st.sidebar.button("🗑️ Reset Stored Predictions Database"):
        clear_db(c, conn)
        st.sidebar.success("Database reset successfully.")

    # Export DB to CSV
    if st.sidebar.button("📤 Export Stored Predictions to CSV"):
        rows = fetch_all_predictions(c)
        if rows:
            df = pd.DataFrame(rows, columns=["id", "timestamp", "filename", "model_score", "threshold", "predicted_label", "true_label"])
            csv = df.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button(label="Download CSV", data=csv, file_name="predictions.csv", mime="text/csv")
        else:
            st.sidebar.info("No data to export.")

    # Initialize confusion matrix session state
    if 'y_true' not in st.session_state:
        st.session_state['y_true'] = []
        st.session_state['y_pred'] = []

    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Analyzing..."):
            input_tensor = preprocess_image(image).to(model.device)
            with torch.no_grad():
                score = model(input_tensor).item()

        confidence_percent = score * 100
        threshold = 30.0
        label = "Forged" if confidence_percent < threshold else "Real"
        label_color = "red" if label == "Forged" else "green"

        st.markdown(f"### ✅ Prediction: <span style='color:{label_color}'><strong>{label}</strong></span>", unsafe_allow_html=True)
        st.write(f"🔢 **Model Confidence Score**: {confidence_percent:.2f}%")
        st.write(f"📉 **Threshold**: {threshold}% → Forged if below")

        true_label = st.radio("What is the actual label of the image?", ["Select", "Real", "Forged"])
        if true_label != "Select":
            y_true = 0 if true_label == "Real" else 1
            y_pred = 0 if label == "Real" else 1
            st.session_state.y_true.append(y_true)
            st.session_state.y_pred.append(y_pred)

            # Save to DB with timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insert_prediction(c, conn, timestamp, uploaded_file.name, confidence_percent, threshold, label, true_label)
            st.success(f"✅ Saved prediction to database (True label: {true_label})")

        # Confusion Matrix Plot (no emojis here)
        if len(st.session_state.y_true) >= 1:
            fig_cm, ax = plt.subplots()
            cm = confusion_matrix(st.session_state.y_true, st.session_state.y_pred, labels=[0, 1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Forged"])
            disp.plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig_cm)
            plt.close()

        # Heatmap
        with st.expander("🔥 Show Heatmap Analysis"):
            heatmap = generate_heatmap(image, model)
            fig, ax = plt.subplots()
            im = ax.imshow(heatmap, cmap='hot')
            ax.set_title("Local Region Suspicion Heatmap")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Forgery Likelihood")
            st.pyplot(fig)
            plt.close()

    # Show DB table in expander
    with st.expander("📜 Show Stored Predictions Table"):
        rows = fetch_all_predictions(c)
        if rows:
            df = pd.DataFrame(rows, columns=["id", "timestamp", "filename", "model_score", "threshold", "predicted_label", "true_label"])
            st.dataframe(df)
        else:
            st.info("No predictions stored yet.")

    # Technical Info
    with st.expander("ℹ️ Model Details"):
        st.markdown("""
        - **Model**: EfficientNet-B0 fine-tuned on CASIA2  
        - **Classification**: Binary - Real (0) vs Forged (1)  
        - **Threshold Rule**:  
            - Confidence < 30% → **Forged**  
            - Confidence ≥ 30% → **Real**  
        - **Heatmap**: Created using sliding window predictions  
        """)

if __name__ == "__main__":
    main()
