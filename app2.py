from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Folders
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load U-Net model
model = load_model("unet_pe_model.h5")

IMG_SIZE = 256


# 🔍 Feature Extraction
def extract_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 50:  # ignore noise
            detected = True

            x, y, w, h = cv2.boundingRect(cnt)

            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

            features.append({
                "area": round(area, 2),
                "centroid": (cx, cy),
                "bbox": (x, y, w, h)
            })

    return detected, features


# 🧠 Process Image
def process_image(filepath):
    # Read image
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize
    img_norm = img_resized / 255.0
    img_input = img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Predict using U-Net
    pred = model.predict(img_input)[0].squeeze()

    # Binary mask for features
    mask = (pred > 0.2).astype(np.uint8) * 255

    # Convert original to color
    img_color = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

    # 🌈 Heatmap generation
    heatmap = (pred * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)

    # Overlay heatmap
    overlay = cv2.addWeighted(img_color, 0.6, heatmap, 0.4, 0)

    # Draw bounding boxes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract features
    detected, features = extract_features(mask)

    # Save output image
    result_path = os.path.join(RESULT_FOLDER, "output.png")
    cv2.imwrite(result_path, overlay)

    return detected, features, result_path


# 🌐 Route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        file = request.files["image"]
        age = request.form["age"]
        gender = request.form["gender"]
        symptoms = request.form["symptoms"]

        # Save uploaded image
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Process image
        detected, features, result_img = process_image(filepath)

        # Final Result Logic (basic)
        if detected or symptoms == "Chest Pain":
            result_text = "⚠️ Pulmonary Embolism DETECTED"
        else:
            result_text = "✅ Normal"

        return render_template("index.html",
                               input_image=filepath,
                               result_image=result_img,
                               result=result_text,
                               age=age,
                               gender=gender,
                               symptoms=symptoms,
                               features=features)

    return render_template("index.html")


# Run app
if __name__ == "__main__":
    app.run(debug=True)