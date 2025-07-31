import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import requests
import io

# Load trained model for re-categorization if needed
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'category_model.h5')
CLASSES_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'category_classes.npy')
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(CLASSES_PATH, allow_pickle=True)
except Exception as e:
    model = None
    class_names = []
    print(f"Warning: Model loading failed - {e}. Using defaults for categorization.")

# Expanded virtual catalog (unchanged from before)
VIRTUAL_CATALOG = {
    'Tops': [
        {'_id': 'vc_top_casual_1', 'name': 'White Cotton T-Shirt', 'category': 'Tops', 'color': 'White', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/ffffff/000000?text=White+T-Shirt'},
        {'_id': 'vc_top_work_1', 'name': 'Blue Button-Down Shirt', 'category': 'Tops', 'color': 'Blue', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/0000ff/ffffff?text=Button-Down'},
        {'_id': 'vc_top_party_1', 'name': 'Sequined Blouse', 'category': 'Tops', 'color': 'Silver', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/c0c0c0/000000?text=Sequined+Blouse'},
        {'_id': 'vc_top_formal_1', 'name': 'Black Blazer', 'category': 'Tops', 'color': 'Black', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/000000/ffffff?text=Black+Blazer'},
    ],
    'Bottoms': [
        {'_id': 'vc_bottom_casual_1', 'name': 'Blue Jeans', 'category': 'Bottoms', 'color': 'Blue', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/0000ff/ffffff?text=Blue+Jeans'},
        {'_id': 'vc_bottom_work_1', 'name': 'Gray Slacks', 'category': 'Bottoms', 'color': 'Gray', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/808080/ffffff?text=Gray+Slacks'},
        {'_id': 'vc_bottom_party_1', 'name': 'Red Mini Skirt', 'category': 'Bottoms', 'color': 'Red', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/ff0000/ffffff?text=Red+Skirt'},
        {'_id': 'vc_bottom_formal_1', 'name': 'Black Tailored Pants', 'category': 'Bottoms', 'color': 'Black', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/000000/ffffff?text=Tailored+Pants'},
    ],
    'Shoes': [
        {'_id': 'vc_shoes_casual_1', 'name': 'White Sneakers', 'category': 'Shoes', 'color': 'White', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/ffffff/000000?text=White+Sneakers'},
        {'_id': 'vc_shoes_work_1', 'name': 'Brown Loafers', 'category': 'Shoes', 'color': 'Brown', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/8b4513/ffffff?text=Loafers'},
        {'_id': 'vc_shoes_party_1', 'name': 'Gold Heels', 'category': 'Shoes', 'color': 'Gold', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/ffd700/000000?text=Gold+Heels'},
        {'_id': 'vc_shoes_formal_1', 'name': 'Black Oxfords', 'category': 'Shoes', 'color': 'Black', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/000000/ffffff?text=Oxfords'},
    ]
}

# Scoring functions (with safeguards for empty lists)
def occasion_filter(items, style_preference):
    allowed_styles = {
        'casual': ['Casual', 'Neutral', 'Work'],
        'work': ['Work', 'Formal', 'Neutral', 'Casual'],
        'party': ['Party', 'Casual', 'Neutral'],
        'formal': ['Formal', 'Work', 'Neutral']
    }.get(style_preference.lower(), ['Neutral'])
    return [item for item in items if item.get('style', '').lower() in [s.lower() for s in allowed_styles]]

def style_score(items, style_preference):
    if not items:
        return 0
    score = 0
    for item in items:
        if item.get('style', '').lower() == style_preference.lower():
            score += 10
        elif 'neutral' in item.get('style', '').lower():
            score += 3
    return score / len(items)

def color_score(items, body_color):
    if not items:
        return 0
    score = 0
    warm_colors = ['red', 'orange', 'yellow', 'gold']
    cool_colors = ['blue', 'green', 'purple', 'silver']
    party_colors = ['red', 'gold', 'silver']
    formal_colors = ['black', 'gray', 'navy']
    for item in items:
        item_color = item.get('color', '').lower()
        if body_color.lower() == 'warm' and item_color in warm_colors:
            score += 5
        elif body_color.lower() == 'cool' and item_color in cool_colors:
            score += 5
        elif item_color in party_colors and 'party' in item.get('style', '').lower():
            score += 4
        elif item_color in formal_colors and 'formal' in item.get('style', '').lower():
            score += 4
        elif item_color in ['black', 'white', 'gray']:
            score += 2
    return score / len(items)

def body_type_score(items, height, weight):
    if not items or not height or not weight:
        return 0
    bmi_proxy = weight / (height ** 2)
    score = 0
    for item in items:
        if height < 165 and 'elongating' in item.get('name', '').lower():
            score += 4
        elif bmi_proxy > 25 and 'loose' in item.get('name', '').lower():
            score += 3
        elif bmi_proxy < 18.5 and 'fitted' in item.get('name', '').lower():
            score += 3
    return score / len(items)

def generate_outfit(payload):
    wardrobe = payload.get('wardrobe', [])
    style_preference = payload.get('style', 'Casual').lower()
    height = payload.get('height')
    weight = payload.get('weight')
    body_color = payload.get('body_color', 'Neutral').lower()

    # Re-categorize wardrobe using trained model if categories are missing/inaccurate
    for item in wardrobe:
        if 'category' not in item or not item['category']:
            if model is not None and 'imageUrl' in item:
                try:
                    response = requests.get(item['imageUrl'], timeout=5)
                    response.raise_for_status()
                    img = Image.open(io.BytesIO(response.content)).convert('RGB').resize((224, 224))
                    arr = preprocess_input(np.expand_dims(np.array(img), 0))
                    preds = model.predict(arr)
                    item['category'] = class_names[np.argmax(preds[0])]
                except Exception as e:
                    item['category'] = 'Unknown'  # Fallback to avoid crash
            else:
                item['category'] = 'Tops'  # Default

    # Filter wardrobe by occasion
    tops = occasion_filter([item for item in wardrobe if item['category'] == 'Tops'], style_preference)
    bottoms = occasion_filter([item for item in wardrobe if item['category'] == 'Bottoms'], style_preference)
    shoes = occasion_filter([item for item in wardrobe if item['category'] == 'Shoes'], style_preference)

    candidates = []
    catalog_tops = [t for t in VIRTUAL_CATALOG['Tops'] if t['style'].lower() == style_preference]
    catalog_bottoms = [b for b in VIRTUAL_CATALOG['Bottoms'] if b['style'].lower() == style_preference]
    catalog_shoes = [s for s in VIRTUAL_CATALOG['Shoes'] if s['style'].lower() == style_preference]

    # Generate combinations (relaxed to allow more mixed results)
    sources_tops = tops if tops else catalog_tops
    sources_bottoms = bottoms if bottoms else catalog_bottoms
    sources_shoes = shoes if shoes else catalog_shoes

    for top in sources_tops:
        for bottom in sources_bottoms:
            for shoe in sources_shoes:
                current = [top, bottom, shoe]
                num_suggested = sum(1 for i in current if '_id' in i and 'vc_' in i['_id'])
                if num_suggested >= len(current):  # Avoid all-suggested, but allow partial
                    continue
                score = style_score(current, style_preference) + color_score(current, body_color) + body_type_score(current, height, weight)
                candidates.append((current, score))

    if not candidates:
        # Fallback to a basic suggestion if nothing matches
        fallback_top = catalog_tops[0] if catalog_tops else {'_id': 'fallback', 'name': 'Basic Top', 'category': 'Tops', 'color': 'White', 'style': style_preference, 'imageUrl': 'https://placehold.co/400x400'}
        fallback_bottom = catalog_bottoms[0] if catalog_bottoms else {'_id': 'fallback', 'name': 'Basic Bottom', 'category': 'Bottoms', 'color': 'Black', 'style': style_preference, 'imageUrl': 'https://placehold.co/400x400'}
        fallback_shoe = catalog_shoes[0] if catalog_shoes else {'_id': 'fallback', 'name': 'Basic Shoes', 'category': 'Shoes', 'color': 'Black', 'style': style_preference, 'imageUrl': 'https://placehold.co/400x400'}
        best_outfit = [fallback_top, fallback_bottom, fallback_shoe]
        best_score = 0
        user_items = []
        suggested_items = best_outfit
    else:
        best_outfit, best_score = max(candidates, key=lambda x: x[1])
        user_items = [item for item in best_outfit if '_id' in item and 'vc_' not in item['_id']]
        suggested_items = [item for item in best_outfit if '_id' in item and 'vc_' in item['_id']]

    return {
        "userItems": user_items,
        "suggestedItems": suggested_items,
        "score": best_score
    }
