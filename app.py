# app.py
import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
import tempfile
import os

# Page config
st.set_page_config(page_title="🚀 Spaceship Equipment Detector", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🚀 Spaceship Equipment Detector</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>Upload an image to detect Toolbox 🧰, Fire Extinguisher 🔥, and Oxygen Tank 🫧 using YOLOv8!</p>",
    unsafe_allow_html=True,
)

# Load model
model = YOLO("D:/HackByte_Dataset/spaceship_equipment_detector_best.pt")

# Class labels
class_names = ['Fire Extinguisher 🔥', 'Toolbox 🧰', 'Oxygen Tank 🫧']

# Upload section
uploaded_file = st.file_uploader("📷 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.markdown("---")
    st.image(uploaded_file, caption='🖼️ Uploaded Image', use_container_width=True)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Run inference
    results = model(tmp_path)

    # Process results
    for result in results:
        img = cv2.imread(tmp_path)

        if result.boxes is not None and len(result.boxes) > 0:
            st.markdown("### 🧠 Detected Objects on Image")

            # Text summary list
            detections_summary = []

            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = f"{class_names[class_id]} ({conf:.2f})"
                detections_summary.append(f"{i+1}. **{class_names[class_id]}** with **{conf:.2f}** confidence")

                # Draw box on image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Convert image to RGB and display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="✅ Prediction Output", use_container_width=True)

            # Display summary below
            st.markdown("### 📋 Detected Items")
            for item in detections_summary:
                st.success(item)
        else:
            st.warning("⚠️ No objects detected in the image.")

    # Cleanup
    os.remove(tmp_path)

else:
    st.info("👆 Upload an image to get started.")
