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
    
    # Calculate BMI category
    height_m = height / 100 if height > 3 else height
    bmi = weight / (height_m ** 2) if height_m > 0 else 22
    
    body_type = "slim" if bmi < 18.5 else "average" if bmi < 25 else "plus"
    height_cat = "short" if height < 160 else "tall" if height > 175 else "average"
    
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
    outfits = []
    
    if style == 'casual':
        outfits = [
            {
                '_id': 'f_casual_1',
                'name': f'{colors[0].title()} Blouse',
                'category': 'Tops',
                'color': colors[0],
                'style': 'Casual',
                'imageUrl': f'https://placehold.co/300x400/{colors[0]}/white?text=Blouse'
            },
            {
                '_id': 'f_casual_2', 
                'name': 'High-waisted Jeans' if height_cat == 'short' else 'Straight Jeans',
                'category': 'Bottoms',
                'color': 'denim',
                'style': 'Casual',
                'imageUrl': 'https://placehold.co/300x400/4169E1/white?text=Jeans'
            },
            {
                '_id': 'f_casual_3',
                'name': 'White Sneakers',
                'category': 'Shoes', 
                'color': 'white',
                'style': 'Casual',
                'imageUrl': 'https://placehold.co/300x400/white/black?text=Sneakers'
            }
        ]
    elif style == 'work':
        outfits = [
            {
                '_id': 'f_work_1',
                'name': f'{colors[0].title()} Blazer',
                'category': 'Tops',
                'color': colors[0],
                'style': 'Work',
                'imageUrl': f'https://placehold.co/300x400/{colors[0]}/white?text=Blazer'
            },
            {
                '_id': 'f_work_2',
                'name': 'A-line Skirt' if body_type == 'plus' else 'Pencil Skirt',
                'category': 'Bottoms',
                'color': colors[1] if len(colors) > 1 else 'black',
                'style': 'Work',
                'imageUrl': f'https://placehold.co/300x400/black/white?text=Skirt'
            },
            {
                '_id': 'f_work_3',
                'name': 'Block Heels',
                'category': 'Shoes',
                'color': 'black',
                'style': 'Work', 
                'imageUrl': 'https://placehold.co/300x400/black/white?text=Heels'
            }
        ]
    elif style == 'party':
        outfits = [
            {
                '_id': 'f_party_1',
                'name': f'{colors[0].title()} Sequin Top',
                'category': 'Tops',
                'color': colors[0],
                'style': 'Party',
                'imageUrl': f'https://placehold.co/300x400/{colors[0]}/white?text=Sequin+Top'
            },
            {
                '_id': 'f_party_2',
                'name': 'Mini Skirt' if body_type == 'slim' else 'Midi Skirt',
                'category': 'Bottoms',
                'color': 'black',
                'style': 'Party',
                'imageUrl': 'https://placehold.co/300x400/black/white?text=Skirt'
            },
            {
                '_id': 'f_party_3',
                'name': f'{colors[0].title()} Heels',
                'category': 'Shoes',
                'color': colors[0],
                'style': 'Party',
                'imageUrl': f'https://placehold.co/300x400/{colors[0]}/white?text=Heels'
            }
        ]
    else:  # formal
        outfits = [
            {
                '_id': 'f_formal_1',
                'name': f'{colors[0].title()} Dress Shirt',
                'category': 'Tops',
                'color': colors[0],
                'style': 'Formal',
                'imageUrl': f'https://placehold.co/300x400/{colors[0]}/white?text=Dress+Shirt'
            },
            {
                '_id': 'f_formal_2',
                'name': 'Tailored Trousers',
                'category': 'Bottoms',
                'color': 'black',
                'style': 'Formal',
                'imageUrl': 'https://placehold.co/300x400/black/white?text=Trousers'
            },
            {
                '_id': 'f_formal_3',
                'name': 'Oxford Shoes',
                'category': 'Shoes',
                'color': 'black',
                'style': 'Formal',
                'imageUrl': 'https://placehold.co/300x400/black/white?text=Oxfords'
            }
        ]
    
    return outfits

def generate_male_outfits(style, body_type, height_cat, colors):
    outfits = []
    
    if style == 'casual':
        outfits = [
            {
                '_id': 'm_casual_1',
                'name': f'{colors[0].title()} T-Shirt',
                'category': 'Tops',
                'color': colors[0],
                'style': 'Casual',
                'imageUrl': f'https://placehold.co/300x400/{colors[0]}/white?text=T-Shirt'
            },
            {
                '_id': 'm_casual_2',
                'name': 'Slim Jeans' if body_type == 'slim' else 'Regular Jeans',
                'category': 'Bottoms',
                'color': 'denim',
                'style': 'Casual',
                'imageUrl': 'https://placehold.co/300x400/4169E1/white?text=Jeans'
            },
            {
                '_id': 'm_casual_3',
                'name': 'Casual Sneakers',
                'category': 'Shoes',
                'color': 'white',
                'style': 'Casual',
                'imageUrl': 'https://placehold.co/300x400/white/black?text=Sneakers'
            }
        ]
    elif style == 'work':
        outfits = [
            {
                '_id': 'm_work_1',
                'name': f'{colors[0].title()} Dress Shirt',
                'category': 'Tops',
                'color': colors[0],
                'style': 'Work',
                'imageUrl': f'https://placehold.co/300x400/{colors[0]}/white?text=Dress+Shirt'
            },
            {
                '_id': 'm_work_2',
                'name': 'Chinos',
                'category': 'Bottoms',
                'color': colors[1] if len(colors) > 1 else 'khaki',
                'style': 'Work',
                'imageUrl': 'https://placehold.co/300x400/D2B48C/black?text=Chinos'
            },
            {
                '_id': 'm_work_3',
                'name': 'Loafers',
                'category': 'Shoes',
                'color': 'brown',
                'style': 'Work',
                'imageUrl': 'https://placehold.co/300x400/8B4513/white?text=Loafers'
            }
        ]
    elif style == 'party':
        outfits = [
            {
                '_id': 'm_party_1',
                'name': f'{colors[0].title()} Button Shirt',
                'category': 'Tops',
                'color': colors[0],
                'style': 'Party',
                'imageUrl': f'https://placehold.co/300x400/{colors[0]}/white?text=Button+Shirt'
            },
            {
                '_id': 'm_party_2',
                'name': 'Dark Jeans',
                'category': 'Bottoms',
                'color': 'dark_denim',
                'style': 'Party',
                'imageUrl': 'https://placehold.co/300x400/191970/white?text=Dark+Jeans'
            },
            {
                '_id': 'm_party_3',
                'name': 'Dress Shoes',
                'category': 'Shoes',
                'color': 'black',
                'style': 'Party',
                'imageUrl': 'https://placehold.co/300x400/black/white?text=Dress+Shoes'
            }
        ]
    else:  # formal
        outfits = [
            {
                '_id': 'm_formal_1',
                'name': f'{colors[0].title()} Suit Jacket',
                'category': 'Tops',
                'color': colors[0],
                'style': 'Formal',
                'imageUrl': f'https://placehold.co/300x400/{colors[0]}/white?text=Suit+Jacket'
            },
            {
                '_id': 'm_formal_2',
                'name': 'Dress Pants',
                'category': 'Bottoms',
                'color': colors[0],
                'style': 'Formal',
                'imageUrl': f'https://placehold.co/300x400/{colors[0]}/white?text=Dress+Pants'
            },
            {
                '_id': 'm_formal_3',
                'name': 'Oxford Shoes',
                'category': 'Shoes',
                'color': 'black',
                'style': 'Formal',
                'imageUrl': 'https://placehold.co/300x400/black/white?text=Oxfords'
            }
        ]
    
    return outfits

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
