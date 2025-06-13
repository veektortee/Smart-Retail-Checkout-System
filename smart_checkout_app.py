# smart_checkout_app.py
import requests
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
from utils import get_price_map, get_class_names

# ---------------------
# CONFIGURATION
# ---------------------
LOCAL_PATH = "best_model.pt" # once you have your model saved locally
CLASS_NAMES = get_class_names()
PRICE_MAP = get_price_map()
EXAMPLE_IMAGE_PATH = "1082ae68-33Booooox_jpg.rf.b1324bf46bbd1400bf00c476877b8b4f.jpg"  # Optional: path to a sample image. Can adjust to yours.

# ---------------------
# LOAD MODEL
# ---------------------
model = YOLO(LOCAL_PATH)

# ---------------------
# STREAMLIT UI
# ---------------------
st.set_page_config(page_title="Smart Retail Checkout", layout="centered")
st.title("🛒 Smart Retail Object Detection")
st.info(
    "⚠️ **Note:** This system detects and prices products by *class/category*, not individual item variants. "
    "All detected instances of the same product share a fixed price. For example, all 'Box' detections "
    "are priced identically, regardless of size or brand variant."
)
st.markdown("### 🧠 Model Info")
st.markdown("This system detects and bills products it was trained on. Please ensure the image contains clear views of the following supported products:")
st.write("**Supported Products:**", ", ".join(CLASS_NAMES))

if os.path.exists(EXAMPLE_IMAGE_PATH):
    st.markdown("#### 📷 Example Image")
    st.image(EXAMPLE_IMAGE_PATH, caption="Example of a supported product image", use_container_width=True)

uploaded_file = st.file_uploader("Upload an image of your cart", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict
    results = model.predict(image, conf=0.5)[0]

    # Count products
    counts = {}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = CLASS_NAMES[cls_id]
        counts[class_name] = counts.get(class_name, 0) + 1

    if counts:
        # Display detections and total
        st.subheader("🧾 Detected Items")
        total = 0.0
        for cls, count in counts.items():
            price = PRICE_MAP.get(cls, 0)
            subtotal = price * count
            total += subtotal
            st.write(f"{cls}: {count} x ${price:.2f} = ${subtotal:.2f}")

        st.markdown("---")
        st.subheader(f"💰 Total: ${total:.2f}")

        # Save image with predictions
        annotated = results.plot()
        st.image(annotated, caption="Detected Products", use_container_width=True)
    else:
        st.warning("No known products were detected. Please upload a clear image of supported items.")
else:
    st.info("Please upload an image to begin.")

