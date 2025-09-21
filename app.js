const fileInput = document.getElementById('file-input');
const imgEl = document.getElementById('input-image');
const predEl = document.getElementById('prediction');

let model;

// Load the TFJS model
tf.loadLayersModel('TFJS_model/model.json').then(m => {
  model = m;
  console.log('✅ Model loaded!');
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(evt) {
    imgEl.src = evt.target.result;
    imgEl.style.display = 'block';
    imgEl.onload = async () => {
      let tensor = tf.browser.fromPixels(imgEl)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(tf.scalar(255))
        .expandDims();
      
      const prediction = model.predict(tensor);
      prediction.array().then(arr => {
        predEl.innerText = `AI_Faces: ${arr[0][0].toFixed(3)}, Real_Faces: ${arr[0][1].toFixed(3)}`;
      });
    };
  };
  reader.readAsDataURL(file);
});
