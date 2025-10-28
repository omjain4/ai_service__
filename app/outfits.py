import os
import numpy as np
# amazonq-ignore-next-line
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
    # amazonq-ignore-next-line
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
    """AI-powered outfit generation using StyleZAP deep learning model"""
    from app.ai_model import ai_model
    
    # Process through AI model
    result = ai_model.predict_outfit_compatibility(payload)
    return result

def get_manual_outfits(style, height, weight, skin_tone, gender):
    """Generate manual outfit suggestions based on parameters"""
    
    # Weight categories
    if weight < 50:
        weight_cat = "underweight"
    elif weight <= 60:
        weight_cat = "light"
    elif weight <= 70:
        weight_cat = "medium"
    elif weight <= 80:
        weight_cat = "heavy"
    else:
        weight_cat = "plus"
    
    # Height categories
    if height < 150:
        height_cat = "petite"
    elif height <= 160:
        height_cat = "short"
    elif height <= 170:
        height_cat = "average"
    elif height <= 180:
        height_cat = "tall"
    else:
        height_cat = "very_tall"
    
    # Combined body type for outfit selection
    body_type = f"{weight_cat}_{height_cat}"
    
    # Color palettes by skin tone
    colors = {
        'warm': {
            'casual': ['coral', 'peach', 'olive', 'brown', 'cream'],
            'work': ['navy', 'burgundy', 'forest green', 'camel'],
            'party': ['gold', 'red', 'emerald', 'bronze'],
            'formal': ['charcoal', 'deep blue', 'burgundy']
        },
        'cool': {
            'casual': ['mint', 'lavender', 'gray', 'white', 'denim'],
            'work': ['navy', 'black', 'cool gray', 'white'],
            'party': ['silver', 'royal blue', 'purple', 'emerald'],
            'formal': ['black', 'navy', 'platinum', 'white']
        },
        'neutral': {
            'casual': ['beige', 'khaki', 'white', 'gray', 'denim'],
            'work': ['navy', 'black', 'gray', 'white'],
            'party': ['black', 'gold', 'red', 'navy'],
            'formal': ['black', 'navy', 'charcoal', 'white']
        }
    }
    
    style_colors = colors.get(skin_tone, colors['neutral']).get(style, ['black', 'white'])
    
    # Generate outfits based on gender and body type
    outfits = []
    
    print(f"DEBUG: Processing gender '{gender}' for style '{style}'")
    
    if gender in ['female', 'woman', 'girl', 'f']:
        print("DEBUG: Using female outfits")
        outfits = generate_female_outfits(style, body_type, height_cat, style_colors)
    elif gender in ['male', 'man', 'boy', 'm']:
        print("DEBUG: Using male outfits")
        outfits = generate_male_outfits(style, body_type, height_cat, style_colors)
    else:
        print(f"DEBUG: Unknown gender '{gender}', defaulting to female")
        outfits = generate_female_outfits(style, body_type, height_cat, style_colors)
    
    return outfits

def generate_female_outfits(style, body_type, height_cat, colors):
    weight_cat, height_cat = body_type.split('_')
    print(f"DEBUG: Generating female outfits for style: {style}, weight: {weight_cat}, height: {height_cat}")
    
    if style == 'casual':
        # Weight-based tops
        if weight_cat in ['underweight', 'light']:
            top_name = 'Fitted Crop Top'
        elif weight_cat in ['medium']:
            top_name = 'Regular T-Shirt'
        else:  # heavy, plus
            top_name = 'Flowy Tunic Top'
        
        # Height-based bottoms
        if height_cat in ['petite', 'short']:
            bottom_name = 'High-Waisted Skinny Jeans'
        elif height_cat == 'average':
            bottom_name = 'Straight Leg Jeans'
        else:  # tall, very_tall
            bottom_name = 'Bootcut Long Jeans'
        
        # Height-based shoes
        if height_cat in ['petite', 'short']:
            shoe_name = 'Platform Sneakers'
        else:
            shoe_name = 'Regular Sneakers'
        
        return [
            {
                '_id': f'f_casual_{weight_cat}_{height_cat}_1',
                'name': f'{colors[0].title()} {top_name}',
                'category': 'Tops',
                'color': colors[0],
                'style': 'Casual',
                'imageUrl': f'https://placehold.co/300x400/{colors[0]}/white?text={top_name.replace(" ", "+")}'
            },
            {
                '_id': f'f_casual_{weight_cat}_{height_cat}_2',
                'name': bottom_name,
                'category': 'Bottoms',
                'color': 'blue',
                'style': 'Casual',
                'imageUrl': f'https://placehold.co/300x400/4169E1/white?text={bottom_name.replace(" ", "+")}'
            },
            {
                '_id': f'f_casual_{weight_cat}_{height_cat}_3',
                'name': shoe_name,
                'category': 'Shoes',
                'color': 'white',
                'style': 'Casual',
                'imageUrl': f'https://placehold.co/300x400/white/black?text={shoe_name.replace(" ", "+")}'
            }
        ]
    elif style == 'work':
        return [
            {
                '_id': 'f_work_blazer_1',
                'name': 'Navy Business Blazer',
                'category': 'Tops',
                'color': 'navy',
                'style': 'Work',
                'imageUrl': 'https://placehold.co/300x400/000080/white?text=Business+Blazer'
            },
            {
                '_id': 'f_work_skirt_2',
                'name': 'Gray Pencil Skirt',
                'category': 'Bottoms',
                'color': 'gray',
                'style': 'Work',
                'imageUrl': 'https://placehold.co/300x400/808080/white?text=Pencil+Skirt'
            },
            {
                '_id': 'f_work_heels_3',
                'name': 'Black Pumps',
                'category': 'Shoes',
                'color': 'black',
                'style': 'Work',
                'imageUrl': 'https://placehold.co/300x400/000000/white?text=Pumps'
            }
        ]
    elif style == 'party':
        return [
            {
                '_id': 'f_party_top_1',
                'name': 'Gold Sequin Halter Top',
                'category': 'Tops',
                'color': 'gold',
                'style': 'Party',
                'imageUrl': 'https://placehold.co/300x400/FFD700/black?text=Sequin+Halter'
            },
            {
                '_id': 'f_party_skirt_2',
                'name': 'Red Leather Mini Skirt',
                'category': 'Bottoms',
                'color': 'red',
                'style': 'Party',
                'imageUrl': 'https://placehold.co/300x400/FF0000/white?text=Leather+Mini'
            },
            {
                '_id': 'f_party_heels_3',
                'name': 'Silver Stiletto Heels',
                'category': 'Shoes',
                'color': 'silver',
                'style': 'Party',
                'imageUrl': 'https://placehold.co/300x400/C0C0C0/black?text=Stiletto'
            }
        ]
    else:  # formal
        return [
            {
                '_id': 'f_formal_dress_1',
                'name': 'Black Evening Gown',
                'category': 'Tops',
                'color': 'black',
                'style': 'Formal',
                'imageUrl': 'https://placehold.co/300x400/000000/white?text=Evening+Gown'
            },
            {
                '_id': 'f_formal_pants_2',
                'name': 'Charcoal Dress Pants',
                'category': 'Bottoms',
                'color': 'charcoal',
                'style': 'Formal',
                'imageUrl': 'https://placehold.co/300x400/36454F/white?text=Dress+Pants'
            },
            {
                '_id': 'f_formal_shoes_3',
                'name': 'Patent Leather Heels',
                'category': 'Shoes',
                'color': 'black',
                'style': 'Formal',
                'imageUrl': 'https://placehold.co/300x400/000000/white?text=Patent+Heels'
            }
        ]

def generate_male_outfits(style, body_type, height_cat, colors):
    weight_cat, height_cat = body_type.split('_')
    print(f"DEBUG: Generating male outfits for style: {style}, weight: {weight_cat}, height: {height_cat}")
    
    if style == 'casual':
        # Weight-based tops
        if weight_cat in ['underweight', 'light']:
            top_name = 'Slim Fit T-Shirt'
        elif weight_cat in ['medium']:
            top_name = 'Regular Polo Shirt'
        else:  # heavy, plus
            top_name = 'Loose Fit Henley'
        
        # Height-based bottoms
        if height_cat in ['petite', 'short']:
            bottom_name = 'Regular Shorts'
        elif height_cat == 'average':
            bottom_name = 'Chino Pants'
        else:  # tall, very_tall
            bottom_name = 'Long Cargo Pants'
        
        return [
            {
                '_id': f'm_casual_{weight_cat}_{height_cat}_1',
                'name': f'{colors[0].title()} {top_name}',
                'category': 'Tops',
                'color': colors[0],
                'style': 'Casual',
                'imageUrl': f'https://placehold.co/300x400/{colors[0]}/white?text={top_name.replace(" ", "+")}'
            },
            {
                '_id': f'm_casual_{weight_cat}_{height_cat}_2',
                'name': bottom_name,
                'category': 'Bottoms',
                'color': 'khaki',
                'style': 'Casual',
                'imageUrl': f'https://placehold.co/300x400/F0E68C/black?text={bottom_name.replace(" ", "+")}'
            },
            {
                '_id': f'm_casual_{weight_cat}_{height_cat}_3',
                'name': 'Casual Sneakers',
                'category': 'Shoes',
                'color': 'white',
                'style': 'Casual',
                'imageUrl': 'https://placehold.co/300x400/white/black?text=Casual+Sneakers'
            }
        ]
    elif style == 'work':
        return [
            {
                '_id': 'm_work_shirt_1',
                'name': 'White Dress Shirt with Tie',
                'category': 'Tops',
                'color': 'white',
                'style': 'Work',
                'imageUrl': 'https://placehold.co/300x400/FFFFFF/000000?text=Dress+Shirt+Tie'
            },
            {
                '_id': 'm_work_pants_2',
                'name': 'Charcoal Suit Pants',
                'category': 'Bottoms',
                'color': 'charcoal',
                'style': 'Work',
                'imageUrl': 'https://placehold.co/300x400/36454F/white?text=Suit+Pants'
            },
            {
                '_id': 'm_work_shoes_3',
                'name': 'Black Oxford Shoes',
                'category': 'Shoes',
                'color': 'black',
                'style': 'Work',
                'imageUrl': 'https://placehold.co/300x400/000000/white?text=Oxford+Shoes'
            }
        ]
    elif style == 'party':
        return [
            {
                '_id': 'm_party_shirt_1',
                'name': 'Black Silk Shirt',
                'category': 'Tops',
                'color': 'black',
                'style': 'Party',
                'imageUrl': 'https://placehold.co/300x400/000000/white?text=Silk+Shirt'
            },
            {
                '_id': 'm_party_jeans_2',
                'name': 'Dark Wash Skinny Jeans',
                'category': 'Bottoms',
                'color': 'dark_blue',
                'style': 'Party',
                'imageUrl': 'https://placehold.co/300x400/191970/white?text=Skinny+Jeans'
            },
            {
                '_id': 'm_party_boots_3',
                'name': 'Brown Chelsea Boots',
                'category': 'Shoes',
                'color': 'brown',
                'style': 'Party',
                'imageUrl': 'https://placehold.co/300x400/8B4513/white?text=Chelsea+Boots'
            }
        ]
    else:  # formal
        return [
            {
                '_id': 'm_formal_tux_1',
                'name': 'Black Tuxedo with Bow Tie',
                'category': 'Tops',
                'color': 'black',
                'style': 'Formal',
                'imageUrl': 'https://placehold.co/300x400/000000/white?text=Tuxedo+Bow+Tie'
            },
            {
                '_id': 'm_formal_pants_2',
                'name': 'Black Tuxedo Pants',
                'category': 'Bottoms',
                'color': 'black',
                'style': 'Formal',
                'imageUrl': 'https://placehold.co/300x400/000000/white?text=Tuxedo+Pants'
            },
            {
                '_id': 'm_formal_shoes_3',
                'name': 'Black Patent Dress Shoes',
                'category': 'Shoes',
                'color': 'black',
                'style': 'Formal',
                'imageUrl': 'https://placehold.co/300x400/000000/white?text=Patent+Dress+Shoes'
            }
        ]

def generate_unisex_outfits(style, body_type, height_cat, colors):
    if style == 'casual':
        return generate_female_outfits(style, body_type, height_cat, colors)
    elif style == 'work':
        return generate_male_outfits(style, body_type, height_cat, colors)
    elif style == 'party':
        return generate_female_outfits(style, body_type, height_cat, colors)
    else:  # formal
        return generate_male_outfits(style, body_type, height_cat, colors)

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
                item['confidence'] = float(np.max(preds[0]))
                print(f"DEBUG: Predicted {item.get('name', 'Item')} as {predicted_category}")
            except Exception as e:
                print(f"DEBUG: Failed to predict category for {item.get('name', 'Item')}: {e}")
                item['category'] = 'Tops'  # Default fallback
                item['confidence'] = 0.0
        else:
            # Ensure every item has a category
            if 'category' not in item or not item['category']:
                item['category'] = 'Tops'  # Default fallback

    # Filter wardrobe by occasion and gender
    def gender_filter(items, gender):
        if gender == 'unisex':
            return items
        return [item for item in items if item.get('gender', 'unisex').lower() in [gender, 'unisex']]
    
    # Apply both occasion and gender filters with fallbacks
    all_tops = [item for item in wardrobe if item.get('category') == 'Tops']
    all_bottoms = [item for item in wardrobe if item.get('category') == 'Bottoms']
    all_shoes = [item for item in wardrobe if item.get('category') == 'Shoes']
    
    print(f"DEBUG: Found {len(all_tops)} tops, {len(all_bottoms)} bottoms, {len(all_shoes)} shoes")
    
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

    print(f"DEBUG: Generated {len(candidates)} outfit candidates")
    
    if not candidates:
        if all_tops and all_bottoms and all_shoes:
            # Pick first available items as fallback
            fallback_outfit = [all_tops[0], all_bottoms[0], all_shoes[0]]
            print(f"DEBUG: Using fallback outfit")
            return {
                "userItems": fallback_outfit,
                "suggestedItems": [],
                "score": 1,
                "message": "Basic outfit suggestion from your wardrobe"
            }
        else:
            # Use Gemini as fallback when no items available
            print(f"DEBUG: No items found, using Gemini fallback")
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
        print(f"DEBUG: Best outfit score: {best_score}")
        
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
