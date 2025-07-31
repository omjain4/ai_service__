from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from preprocessing.image_utils import prepare_image
import numpy as np

app = Flask(__name__)

# Load the trained model and class names on startup
model = load_model('models/category_model.h5')
with open('models/class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

@app.route('/predict-category', methods=['POST'])
def predict_category():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    try:
        # Read image bytes and preprocess
        img_bytes = file.read()
        preprocessed_image = prepare_image(img_bytes)

        # Make a prediction
        prediction = model.predict(preprocessed_image)

        # Decode the prediction
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_category = class_names[predicted_class_index]

        return jsonify({'category': predicted_category})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000) # Run on a different port
