import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import io
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'category_model.h5')
CLASSES_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'category_classes.npy')

model = None
class_names = None

def load_model_once():
    global model, class_names
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = np.load(CLASSES_PATH, allow_pickle=True)

async def predict_category(file):
    load_model_once()
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    arr = np.expand_dims(np.array(img), 0)
    arr = preprocess_input(arr)
    preds = model.predict(arr)
    pred_idx = np.argmax(preds[0])
    category = class_names[pred_idx]
    return {"category": str(category)}
