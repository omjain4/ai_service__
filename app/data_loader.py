import os
import pandas as pd
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, '..', 'datasets')
SHAPE_ANN_DIR = os.path.join(DATASET_DIR, 'shape_ann')
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, 'train_images')
VAL_IMG_DIR = os.path.join(DATASET_DIR, 'train_images')  # Adjust if val images are in a different folder
TEST_IMG_DIR = os.path.join(DATASET_DIR, 'test_images')

def load_annotations(txt_file, img_dir):
    """
    Returns DataFrame with columns: image_path (absolute), category (str/int)
    Assumes each line: image_name [optional_attrs] category (last item as label)
    """
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()  # Split on any whitespace
            if len(parts) < 2:
                continue  # Skip invalid lines
            image_name = parts[0]  # First item is image name
            category = parts[-1]   # Last item is category
            image_path = os.path.join(img_dir, image_name)
            data.append({'image_path': image_path, 'category': category})
    df = pd.DataFrame(data)
    return df

def load_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    return np.array(img)
