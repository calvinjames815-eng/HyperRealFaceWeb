let model;
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

async function loadModel() {
    model = await tf.loadLayersModel('TFJS_model/model.json');
    console.log('Model loaded');
}

async function setupWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcam.srcObject = stream;
}

document.getElementById('capture').addEventListener('click', async () => {
    ctx.drawImage(webcam, 0, 0, 224, 224);
    const img = tf.browser.fromPixels(canvas).expandDims(0).div(255.0);
    const pred = await model.predict(img);
    const label = (pred.dataSync()[0] > 0.5) ? 'AI Face' : 'Real Face';
    document.getElementById('result').innerText = `Prediction: ${label}`;
});

loadModel();
setupWebcam();
