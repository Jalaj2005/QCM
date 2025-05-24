let model;

async function loadModel() {
  model = await tf.loadLayersModel('model/model.json');
  console.log("Model loaded");
}

function preprocessImage(image) {
  const canvas = document.createElement('canvas');
  canvas.width = 28;
  canvas.height = 28;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, 28, 28);
  const imageData = ctx.getImageData(0, 0, 28, 28);

  const gray = [];
  for (let i = 0; i < imageData.data.length; i += 4) {
    const avg = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
    gray.push(avg / 255);
  }

  return tf.tensor(gray, [1, 28, 28, 1]);
}

document.getElementById('imageInput').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const img = new Image();
  const reader = new FileReader();

  reader.onload = function (e) {
    img.src = e.target.result;
    img.onload = async function () {
      // Preview canvas for internal use
      const previewCanvas = document.getElementById('previewCanvas');
      const ctx = previewCanvas.getContext('2d');
      ctx.drawImage(img, 0, 0, 28, 28);

      // Enlarged display canvas
      const displayCanvas = document.getElementById('displayCanvas');
      const displayCtx = displayCanvas.getContext('2d');
      displayCtx.clearRect(0, 0, 140, 140);
      displayCtx.drawImage(img, 0, 0, 140, 140);

      const tensor = preprocessImage(img);
      const prediction = model.predict(tensor);
      const predictedClass = prediction.argMax(1).dataSync()[0];

      document.getElementById('prediction').innerHTML = `Prediction: <span>${predictedClass}</span>`;
    };
  };

  reader.readAsDataURL(file);
});

loadModel();
