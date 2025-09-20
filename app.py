from flask import Flask, render_template, request, redirect, url_for
import os
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
app.config["RESULT_FOLDER"] = "gradcam_results/"

# ✅ Google Drive file ID (replace with yours if different)
DRIVE_FILE_ID = "1nVn9DMoMUEt_csgDNTA-JOy5SWVN5U1X"
MODEL_PATH = "CNN_HyperRealFaces_BestModel.h5"

# ✅ Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# ✅ Load trained model
model = load_model(MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # TODO: Add Grad-CAM + Prediction here
        prediction = "Fake/Real (placeholder)"  

        return f"✅ Uploaded {filename}, Prediction: {prediction}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
