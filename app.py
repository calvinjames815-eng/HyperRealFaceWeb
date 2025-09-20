# ==========================
# HyperRealFaceWeb: app.py (Render-ready & stable)
# ==========================
import os
from flask import Flask, request, render_template, send_file
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import gdown

# ==========================
# Settings
# ==========================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
GRADCAM_FOLDER = "gradcam_results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

IMG_SIZE = 224
CLASS_NAMES = ["AI_Faces", "Real_Faces"]

# ==========================
# Download model from Google Drive if missing
# ==========================
MODEL_FILE = "CNN_HyperRealFaces_BestModel.h5"
GDRIVE_ID = "1nVn9DMoMUEt_csgDNTA-JOy5SWVN5U1X"  # your file id
if not os.path.exists(MODEL_FILE):
    url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
    gdown.download(url, MODEL_FILE, quiet=False)

# ==========================
# Load Model
# ==========================
model = tf.keras.models.load_model(MODEL_FILE)

# ==========================
# TTA Prediction
# ==========================
def predict_tta(img_path, model, tta_rounds=7):
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = tf.keras.utils.img_to_array(img)/255.0
    preds = []
    for _ in range(tta_rounds):
        aug = tf.image.random_flip_left_right(x)
        aug = tf.image.random_brightness(aug, max_delta=0.1)
        aug = tf.expand_dims(aug, 0)
        preds.append(model.predict(aug, verbose=0)[0])
    final_pred = np.mean(preds, axis=0)
    label_idx = int(final_pred > 0.5)
    return CLASS_NAMES[label_idx], final_pred[label_idx]*100, x[np.newaxis,...]

# ==========================
# Stable Grad-CAM
# ==========================
def grad_cam(img_array, model, last_conv="Conv_1", alpha=0.4):
    # Ensure batch dimension
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, 0)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img_array)
        loss = pred[:,0]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_out = conv_out[0].numpy()
    heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)

    for i, w in enumerate(pooled_grads):
        heatmap += w * conv_out[:,:,i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.uint8(img_array[0]*255)
    overlay = cv2.addWeighted(original, 1-alpha, heatmap, alpha, 0)

    return overlay

# ==========================
# Flask Routes
# ==========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file uploaded", 400
        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400

        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Prediction
        label, prob, img_array = predict_tta(filepath, model)

        # Grad-CAM
        cam_overlay = grad_cam(img_array, model)
        cam_path = os.path.join(GRADCAM_FOLDER, f"gradcam_{filename}")
        cv2.imwrite(cam_path, cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))

        return f"""
        <h2>Prediction: {label}</h2>
        <h3>Confidence: {prob:.2f}%</h3>
        <h3>Grad-CAM:</h3>
        <img src="/gradcam/{filename}" width="300">
        """
    return render_template("index.html")

@app.route("/gradcam/<filename>")
def gradcam_file(filename):
    return send_file(os.path.join(GRADCAM_FOLDER, f"gradcam_{filename}"))

# ==========================
# Run on Render
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
