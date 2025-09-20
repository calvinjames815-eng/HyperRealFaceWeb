from flask import Flask, request, render_template, send_file
import tensorflow as tf
import numpy as np
import os
import cv2
from werkzeug.utils import secure_filename
from PIL import Image
import gdown

# ==========================
# Paths & Settings
# ==========================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
GRADCAM_FOLDER = "gradcam_results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
IMG_SIZE = 224
CLASS_NAMES = ["AI_Faces", "Real_Faces"]

# ==========================
# Download Model from Google Drive if not exists
# ==========================
MODEL_PATH = "CNN_HyperRealFaces_BestModel.h5"
GDRIVE_ID = "1nVn9DMoMUEt_csgDNTA-JOy5SWVN5U1X"  # Your link ID

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", MODEL_PATH, quiet=False)

# ==========================
# Load Model
# ==========================
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# Grad-CAM function
# ==========================
def grad_cam(img_array, model, last_conv="Conv_1", alpha=0.4):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img_array)
        loss = pred[:,0]
    grads = tape.gradient(loss, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0,1))
    cam = np.dot(conv_out[0], weights.numpy())
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = np.maximum(cam,0)
    cam = cam / (cam.max()+1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.uint8(img_array[0]*255), 1-alpha, heatmap, alpha, 0)
    return overlay

# ==========================
# TTA Prediction
# ==========================
def predict_tta(img_path, model, tta_rounds=7):
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = tf.keras.utils.img_to_array(img)/255.0
    preds=[]
    for _ in range(tta_rounds):
        aug = tf.image.random_flip_left_right(x)
        aug = tf.image.random_brightness(aug, max_delta=0.1)
        aug = tf.expand_dims(aug,0)
        preds.append(model.predict(aug, verbose=0)[0])
    final_pred = np.mean(preds, axis=0)
    label_idx = int(final_pred>0.5)
    return CLASS_NAMES[label_idx], final_pred[label_idx]*100, x[np.newaxis,...]

# ==========================
# Flask Routes
# ==========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Prediction + TTA
        label, prob, img_array = predict_tta(filepath, model)

        # Grad-CAM
        cam_overlay = grad_cam(img_array, model)
        cam_path = os.path.join(GRADCAM_FOLDER, f"gradcam_{filename}")
        cv2.imwrite(cam_path, cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))

        return render_template("index.html", prediction=label, confidence=prob, cam_img=f"gradcam_{filename}")

    return render_template("index.html")

@app.route("/gradcam/<filename>")
def gradcam_file(filename):
    return send_file(os.path.join(GRADCAM_FOLDER, filename))

# ==========================
# Run Flask
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # TODO: Add Grad-CAM + Prediction here
        prediction = "Fake/Real (placeholder)"  

        return f"✅ Uploaded {filename}, Prediction: {prediction}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
