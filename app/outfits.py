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

# Remove hardcoded catalog - use trained model predictions only

# Scoring functions (with safeguards for empty lists)
def occasion_filter(items, style_preference):
    # Return all items if no specific style filtering needed
    return items

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

def color_score(items, body_color, style_preference):
    if not items:
        return 0
    score = 0
    
    # Enhanced color matching based on skin tone and occasion
    skin_tone_colors = {
        'warm': {
            'casual': ['coral', 'peach', 'warm_yellow', 'orange', 'brown'],
            'work': ['navy', 'burgundy', 'forest_green', 'cream'],
            'party': ['gold', 'red', 'emerald', 'bronze'],
            'formal': ['charcoal', 'deep_blue', 'burgundy', 'cream']
        },
        'cool': {
            'casual': ['mint', 'lavender', 'cool_blue', 'gray', 'white'],
            'work': ['navy', 'gray', 'black', 'white', 'cool_blue'],
            'party': ['silver', 'royal_blue', 'purple', 'emerald'],
            'formal': ['black', 'navy', 'gray', 'white', 'silver']
        },
        'neutral': {
            'casual': ['beige', 'olive', 'denim', 'white', 'gray'],
            'work': ['navy', 'black', 'gray', 'white', 'beige'],
            'party': ['black', 'gold', 'silver', 'red'],
            'formal': ['black', 'navy', 'gray', 'white']
        }
    }
    
    preferred_colors = skin_tone_colors.get(body_color.lower(), {}).get(style_preference.lower(), [])
    
    for item in items:
        item_color = item.get('color', '').lower()
        if item_color in preferred_colors:
            score += 8
        elif item_color in ['black', 'white', 'gray', 'navy']:  # Universal colors
            score += 4
        else:
            score += 1
    
    return score / len(items)

def body_type_score(items, height, weight):
    if not items or not height or not weight:
        return 0
    
    # Convert height to meters if needed
    height_m = height / 100 if height > 3 else height
    bmi = weight / (height_m ** 2)
    
    score = 0
    for item in items:
        item_name = item.get('name', '').lower()
        item_style = item.get('style', '').lower()
        
        # Height-based scoring
        if height_m < 1.65:  # Shorter height
            if any(word in item_name for word in ['high-waisted', 'cropped', 'fitted']):
                score += 5
            elif any(word in item_name for word in ['long', 'oversized']):
                score -= 2
        elif height_m > 1.75:  # Taller height
            if any(word in item_name for word in ['long', 'maxi', 'wide-leg']):
                score += 5
        
        # BMI-based scoring
        if bmi > 25:  # Higher BMI
            if any(word in item_name for word in ['loose', 'flowy', 'a-line']):
                score += 4
            elif any(word in item_name for word in ['tight', 'bodycon']):
                score -= 3
        elif bmi < 18.5:  # Lower BMI
            if any(word in item_name for word in ['fitted', 'structured', 'tailored']):
                score += 4
        
        # Universal flattering items
        if any(word in item_name for word in ['wrap', 'v-neck', 'straight-leg']):
            score += 2
    
    return score / len(items)

def generate_outfit(payload):
    wardrobe = payload.get('wardrobe', [])
    style_preference = payload.get('style', 'Casual').lower()
    height = payload.get('height')
    weight = payload.get('weight')
    body_color = payload.get('body_color', 'Neutral').lower()
    gender = payload.get('gender', 'unisex').lower()

    # Re-categorize ALL wardrobe items using trained model
    for item in wardrobe:
        if model is not None and 'imageUrl' in item:
            try:
                response = requests.get(item['imageUrl'], timeout=5)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content)).convert('RGB').resize((224, 224))
                arr = preprocess_input(np.expand_dims(np.array(img), 0))
                preds = model.predict(arr)
                predicted_category = class_names[np.argmax(preds[0])]
                item['category'] = predicted_category
                # Add confidence score
                item['confidence'] = float(np.max(preds[0]))
            except Exception as e:
                item['category'] = 'Unknown'
                item['confidence'] = 0.0

    # Filter wardrobe by occasion and gender
    def gender_filter(items, gender):
        if gender == 'unisex':
            return items
        return [item for item in items if item.get('gender', 'unisex').lower() in [gender, 'unisex']]
    
    # Apply both occasion and gender filters with fallbacks
    all_tops = [item for item in wardrobe if item.get('category') == 'Tops']
    all_bottoms = [item for item in wardrobe if item.get('category') == 'Bottoms']
    all_shoes = [item for item in wardrobe if item.get('category') == 'Shoes']
    
    tops = gender_filter(occasion_filter(all_tops, style_preference), gender)
    bottoms = gender_filter(occasion_filter(all_bottoms, style_preference), gender)
    shoes = gender_filter(occasion_filter(all_shoes, style_preference), gender)
    
    # Fallback to all items if filters are too restrictive
    if not tops:
        tops = all_tops
    if not bottoms:
        bottoms = all_bottoms
    if not shoes:
        shoes = all_shoes

    candidates = []
    
    # Use only user's wardrobe items predicted by trained model
    sources_tops = tops if tops else []
    sources_bottoms = bottoms if bottoms else []
    sources_shoes = shoes if shoes else []

    for top in sources_tops:
        for bottom in sources_bottoms:
            for shoe in sources_shoes:
                current = [top, bottom, shoe]
                # Remove this filter to allow all combinations
                score = (style_score(current, style_preference) + 
                        color_score(current, body_color, style_preference) + 
                        body_type_score(current, height, weight))
                candidates.append((current, score))

    if not candidates:
        # Fallback: try without strict filtering
        all_tops = [item for item in wardrobe if item.get('category') == 'Tops']
        all_bottoms = [item for item in wardrobe if item.get('category') == 'Bottoms'] 
        all_shoes = [item for item in wardrobe if item.get('category') == 'Shoes']
        
        if all_tops and all_bottoms and all_shoes:
            # Pick first available items as fallback
            fallback_outfit = [all_tops[0], all_bottoms[0], all_shoes[0]]
            return {
                "userItems": fallback_outfit,
                "suggestedItems": [],
                "score": 1,
                "message": "Basic outfit suggestion from your wardrobe"
            }
        else:
            # Use Gemini as fallback when no items available
            from app.gemini_service import get_outfit_recommendations
            gemini_result = get_outfit_recommendations(
                wardrobe, style_preference, body_color, gender, style_preference
            )
            return {
                "userItems": [],
                "suggestedItems": [],
                "score": 0,
                "gemini_fallback": gemini_result,
                "message": f"No items found. Gemini suggestions provided. Missing: Tops({len(all_tops)}), Bottoms({len(all_bottoms)}), Shoes({len(all_shoes)})"
            }
    else:
        best_outfit, best_score = max(candidates, key=lambda x: x[1])
        user_items = best_outfit
        suggested_items = []
        
        # Add Gemini suggestions alongside our results
        from app.gemini_service import get_outfit_recommendations
        gemini_result = get_outfit_recommendations(
            wardrobe, style_preference, body_color, gender, style_preference
        )

    return {
        "userItems": user_items,
        "suggestedItems": suggested_items,
        "score": best_score,
        "gemini_suggestions": gemini_result if 'gemini_result' in locals() else None
    }
