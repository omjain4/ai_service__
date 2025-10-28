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

# Expanded virtual catalog with more variety
VIRTUAL_CATALOG = {
    'Tops': [
        # Casual Tops
        {'_id': 'vc_top_casual_1', 'name': 'White Cotton T-Shirt', 'category': 'Tops', 'color': 'White', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/ffffff/000000?text=White+T-Shirt'},
        {'_id': 'vc_top_casual_2', 'name': 'Navy Polo Shirt', 'category': 'Tops', 'color': 'Navy', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/000080/ffffff?text=Navy+Polo'},
        {'_id': 'vc_top_casual_3', 'name': 'Gray Hoodie', 'category': 'Tops', 'color': 'Gray', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/808080/ffffff?text=Gray+Hoodie'},
        {'_id': 'vc_top_casual_4', 'name': 'Red Tank Top', 'category': 'Tops', 'color': 'Red', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/ff0000/ffffff?text=Red+Tank'},
        {'_id': 'vc_top_casual_5', 'name': 'Green Flannel Shirt', 'category': 'Tops', 'color': 'Green', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/008000/ffffff?text=Green+Flannel'},
        # Work Tops
        {'_id': 'vc_top_work_1', 'name': 'Blue Button-Down Shirt', 'category': 'Tops', 'color': 'Blue', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/0000ff/ffffff?text=Button-Down'},
        {'_id': 'vc_top_work_2', 'name': 'White Dress Shirt', 'category': 'Tops', 'color': 'White', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/ffffff/000000?text=Dress+Shirt'},
        {'_id': 'vc_top_work_3', 'name': 'Navy Blazer', 'category': 'Tops', 'color': 'Navy', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/000080/ffffff?text=Navy+Blazer'},
        {'_id': 'vc_top_work_4', 'name': 'Gray Cardigan', 'category': 'Tops', 'color': 'Gray', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/808080/ffffff?text=Gray+Cardigan'},
        {'_id': 'vc_top_work_5', 'name': 'Burgundy Blouse', 'category': 'Tops', 'color': 'Burgundy', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/800020/ffffff?text=Burgundy+Blouse'},
        # Party Tops
        {'_id': 'vc_top_party_1', 'name': 'Sequined Blouse', 'category': 'Tops', 'color': 'Silver', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/c0c0c0/000000?text=Sequined+Blouse'},
        {'_id': 'vc_top_party_2', 'name': 'Gold Metallic Top', 'category': 'Tops', 'color': 'Gold', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/ffd700/000000?text=Gold+Metallic'},
        {'_id': 'vc_top_party_3', 'name': 'Black Silk Camisole', 'category': 'Tops', 'color': 'Black', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/000000/ffffff?text=Silk+Camisole'},
        {'_id': 'vc_top_party_4', 'name': 'Red Velvet Top', 'category': 'Tops', 'color': 'Red', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/ff0000/ffffff?text=Velvet+Top'},
        {'_id': 'vc_top_party_5', 'name': 'Purple Satin Blouse', 'category': 'Tops', 'color': 'Purple', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/800080/ffffff?text=Satin+Blouse'},
        # Formal Tops
        {'_id': 'vc_top_formal_1', 'name': 'Black Blazer', 'category': 'Tops', 'color': 'Black', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/000000/ffffff?text=Black+Blazer'},
        {'_id': 'vc_top_formal_2', 'name': 'Charcoal Suit Jacket', 'category': 'Tops', 'color': 'Charcoal', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/36454f/ffffff?text=Suit+Jacket'},
        {'_id': 'vc_top_formal_3', 'name': 'Navy Tuxedo', 'category': 'Tops', 'color': 'Navy', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/000080/ffffff?text=Navy+Tuxedo'},
        {'_id': 'vc_top_formal_4', 'name': 'White Evening Shirt', 'category': 'Tops', 'color': 'White', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/ffffff/000000?text=Evening+Shirt'},
        {'_id': 'vc_top_formal_5', 'name': 'Burgundy Formal Blouse', 'category': 'Tops', 'color': 'Burgundy', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/800020/ffffff?text=Formal+Blouse'},
    ],
    'Bottoms': [
        # Casual Bottoms
        {'_id': 'vc_bottom_casual_1', 'name': 'Blue Jeans', 'category': 'Bottoms', 'color': 'Blue', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/0000ff/ffffff?text=Blue+Jeans'},
        {'_id': 'vc_bottom_casual_2', 'name': 'Black Skinny Jeans', 'category': 'Bottoms', 'color': 'Black', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/000000/ffffff?text=Skinny+Jeans'},
        {'_id': 'vc_bottom_casual_3', 'name': 'Khaki Chinos', 'category': 'Bottoms', 'color': 'Khaki', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/f0e68c/000000?text=Khaki+Chinos'},
        {'_id': 'vc_bottom_casual_4', 'name': 'Denim Shorts', 'category': 'Bottoms', 'color': 'Blue', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/4169e1/ffffff?text=Denim+Shorts'},
        {'_id': 'vc_bottom_casual_5', 'name': 'Gray Sweatpants', 'category': 'Bottoms', 'color': 'Gray', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/808080/ffffff?text=Sweatpants'},
        # Work Bottoms
        {'_id': 'vc_bottom_work_1', 'name': 'Gray Slacks', 'category': 'Bottoms', 'color': 'Gray', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/808080/ffffff?text=Gray+Slacks'},
        {'_id': 'vc_bottom_work_2', 'name': 'Navy Dress Pants', 'category': 'Bottoms', 'color': 'Navy', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/000080/ffffff?text=Dress+Pants'},
        {'_id': 'vc_bottom_work_3', 'name': 'Black Pencil Skirt', 'category': 'Bottoms', 'color': 'Black', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/000000/ffffff?text=Pencil+Skirt'},
        {'_id': 'vc_bottom_work_4', 'name': 'Charcoal Trousers', 'category': 'Bottoms', 'color': 'Charcoal', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/36454f/ffffff?text=Trousers'},
        {'_id': 'vc_bottom_work_5', 'name': 'Brown A-Line Skirt', 'category': 'Bottoms', 'color': 'Brown', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/8b4513/ffffff?text=A-Line+Skirt'},
        # Party Bottoms
        {'_id': 'vc_bottom_party_1', 'name': 'Red Mini Skirt', 'category': 'Bottoms', 'color': 'Red', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/ff0000/ffffff?text=Red+Skirt'},
        {'_id': 'vc_bottom_party_2', 'name': 'Black Leather Pants', 'category': 'Bottoms', 'color': 'Black', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/000000/ffffff?text=Leather+Pants'},
        {'_id': 'vc_bottom_party_3', 'name': 'Gold Sequin Skirt', 'category': 'Bottoms', 'color': 'Gold', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/ffd700/000000?text=Sequin+Skirt'},
        {'_id': 'vc_bottom_party_4', 'name': 'Silver Metallic Shorts', 'category': 'Bottoms', 'color': 'Silver', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/c0c0c0/000000?text=Metallic+Shorts'},
        {'_id': 'vc_bottom_party_5', 'name': 'Purple Satin Pants', 'category': 'Bottoms', 'color': 'Purple', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/800080/ffffff?text=Satin+Pants'},
        # Formal Bottoms
        {'_id': 'vc_bottom_formal_1', 'name': 'Black Tailored Pants', 'category': 'Bottoms', 'color': 'Black', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/000000/ffffff?text=Tailored+Pants'},
        {'_id': 'vc_bottom_formal_2', 'name': 'Charcoal Suit Pants', 'category': 'Bottoms', 'color': 'Charcoal', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/36454f/ffffff?text=Suit+Pants'},
        {'_id': 'vc_bottom_formal_3', 'name': 'Navy Formal Trousers', 'category': 'Bottoms', 'color': 'Navy', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/000080/ffffff?text=Formal+Trousers'},
        {'_id': 'vc_bottom_formal_4', 'name': 'Black Evening Skirt', 'category': 'Bottoms', 'color': 'Black', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/000000/ffffff?text=Evening+Skirt'},
        {'_id': 'vc_bottom_formal_5', 'name': 'Burgundy Dress Pants', 'category': 'Bottoms', 'color': 'Burgundy', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/800020/ffffff?text=Dress+Pants'},
    ],
    'Shoes': [
        # Casual Shoes
        {'_id': 'vc_shoes_casual_1', 'name': 'White Sneakers', 'category': 'Shoes', 'color': 'White', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/ffffff/000000?text=White+Sneakers'},
        {'_id': 'vc_shoes_casual_2', 'name': 'Black Canvas Shoes', 'category': 'Shoes', 'color': 'Black', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/000000/ffffff?text=Canvas+Shoes'},
        {'_id': 'vc_shoes_casual_3', 'name': 'Blue Running Shoes', 'category': 'Shoes', 'color': 'Blue', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/0000ff/ffffff?text=Running+Shoes'},
        {'_id': 'vc_shoes_casual_4', 'name': 'Brown Sandals', 'category': 'Shoes', 'color': 'Brown', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/8b4513/ffffff?text=Sandals'},
        {'_id': 'vc_shoes_casual_5', 'name': 'Gray Slip-Ons', 'category': 'Shoes', 'color': 'Gray', 'style': 'Casual', 'imageUrl': 'https://placehold.co/400x400/808080/ffffff?text=Slip-Ons'},
        # Work Shoes
        {'_id': 'vc_shoes_work_1', 'name': 'Brown Loafers', 'category': 'Shoes', 'color': 'Brown', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/8b4513/ffffff?text=Loafers'},
        {'_id': 'vc_shoes_work_2', 'name': 'Black Pumps', 'category': 'Shoes', 'color': 'Black', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/000000/ffffff?text=Black+Pumps'},
        {'_id': 'vc_shoes_work_3', 'name': 'Navy Flats', 'category': 'Shoes', 'color': 'Navy', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/000080/ffffff?text=Navy+Flats'},
        {'_id': 'vc_shoes_work_4', 'name': 'Burgundy Oxfords', 'category': 'Shoes', 'color': 'Burgundy', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/800020/ffffff?text=Oxfords'},
        {'_id': 'vc_shoes_work_5', 'name': 'Gray Block Heels', 'category': 'Shoes', 'color': 'Gray', 'style': 'Work', 'imageUrl': 'https://placehold.co/400x400/808080/ffffff?text=Block+Heels'},
        # Party Shoes
        {'_id': 'vc_shoes_party_1', 'name': 'Gold Heels', 'category': 'Shoes', 'color': 'Gold', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/ffd700/000000?text=Gold+Heels'},
        {'_id': 'vc_shoes_party_2', 'name': 'Silver Stilettos', 'category': 'Shoes', 'color': 'Silver', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/c0c0c0/000000?text=Stilettos'},
        {'_id': 'vc_shoes_party_3', 'name': 'Red Platform Heels', 'category': 'Shoes', 'color': 'Red', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/ff0000/ffffff?text=Platform+Heels'},
        {'_id': 'vc_shoes_party_4', 'name': 'Black Ankle Boots', 'category': 'Shoes', 'color': 'Black', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/000000/ffffff?text=Ankle+Boots'},
        {'_id': 'vc_shoes_party_5', 'name': 'Purple Strappy Heels', 'category': 'Shoes', 'color': 'Purple', 'style': 'Party', 'imageUrl': 'https://placehold.co/400x400/800080/ffffff?text=Strappy+Heels'},
        # Formal Shoes
        {'_id': 'vc_shoes_formal_1', 'name': 'Black Oxfords', 'category': 'Shoes', 'color': 'Black', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/000000/ffffff?text=Oxfords'},
        {'_id': 'vc_shoes_formal_2', 'name': 'Patent Leather Shoes', 'category': 'Shoes', 'color': 'Black', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/000000/ffffff?text=Patent+Leather'},
        {'_id': 'vc_shoes_formal_3', 'name': 'Navy Dress Shoes', 'category': 'Shoes', 'color': 'Navy', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/000080/ffffff?text=Dress+Shoes'},
        {'_id': 'vc_shoes_formal_4', 'name': 'Burgundy Loafers', 'category': 'Shoes', 'color': 'Burgundy', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/800020/ffffff?text=Burgundy+Loafers'},
        {'_id': 'vc_shoes_formal_5', 'name': 'Black Evening Heels', 'category': 'Shoes', 'color': 'Black', 'style': 'Formal', 'imageUrl': 'https://placehold.co/400x400/000000/ffffff?text=Evening+Heels'},
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