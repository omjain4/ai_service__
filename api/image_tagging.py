from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

IMG_MODEL_PATH = '../models/clothing_classifier.h5'
model = load_model(IMG_MODEL_PATH)
category_labels = ["Top", "Bottom", "Dress", "Jacket", ...] # Your real categories

def predict_img_tags(file):
    img = Image.open(file).resize((224,224)).convert('RGB')
    x = np.array(img) / 255.0
    preds = model.predict(x.reshape(1,224,224,3))[0]
    category = category_labels[np.argmax(preds)]
    # Placeholder: You’ll expand for style, color, etc.
    return {"category": category, "color": "Blue", "style": "Casual"}
