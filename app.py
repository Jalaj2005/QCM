import sys
print(sys.executable)

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Make sure the uploads folder exists

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model('mnist_model.keras')

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)  # Added channel dimension
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        # Save file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Predict
        image = preprocess_image(filepath)
        pred = model.predict(image)
        prediction = int(np.argmax(pred))

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
