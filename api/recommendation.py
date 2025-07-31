import random

# --- Feature Engineering & Color Theory ---

# A simplified color mapping to RGB and then to a vector space
COLOR_MAP = {
    'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1],
    'black': [0, 0, 0], 'white': [1, 1, 1], 'yellow': [1, 1, 0],
    'purple': [0.5, 0, 0.5], 'orange': [1, 0.5, 0], 'gray': [0.5, 0.5, 0.5],
    # Add more colors as your data grows
}

# A simplified style mapping
STYLE_MAP = {'casual': 0, 'work': 1, 'party': 2, 'formal': 3}

def get_color_vector(color_name):
    """Converts a color name to its vector representation."""
    return COLOR_MAP.get(str(color_name).lower(), [0.5, 0.5, 0.5]) # Default to gray

def calculate_color_harmony(color1, color2):
    """Calculates a score based on color distance. Lower distance = better harmony."""
    v1, v2 = get_color_vector(color1), get_color_vector(color2)
    # Euclidean distance - a simple measure of color difference
    distance = sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5
    # We want a high score for harmonious colors (low distance), so we invert it.
    # 1.5 is a magic number to scale the score nicely.
    return max(0, 1.5 - distance) * 5

# --- Main Recommendation Logic ---

def recommend_outfit(wardrobe, style, height, weight, body_color):
    """
    Recommends the best outfit by deterministically scoring all possible combinations.
    """
    tops = [item for item in wardrobe if item.get('category') == 'Tops']
    bottoms = [item for item in wardrobe if item.get('category') == 'Bottoms']
    
    if not tops or not bottoms:
        # If wardrobe is incomplete, you can use the VIRTUAL_CATALOG logic from previous steps.
        # For now, we'll focus on users who have at least one top and bottom.
        return []

    best_outfit = None
    best_score = -1

    # Deterministically iterate through all pairs to find the best score
    for top in tops:
        for bottom in bottoms:
            # --- Scoring ---
            score = 0
            
            # 1. Color Harmony Score (major factor)
            score += calculate_color_harmony(top.get('color'), bottom.get('color'))

            # 2. Style Consistency Score
            # In a real model, items would have style tags from the vision model.
            # We simulate it here to show how different filters produce different results.
            item_style_preference = STYLE_MAP.get(style.lower(), 0)
            # This logic will prefer items that are more likely to be 'casual', 'work', etc.
            # a more advanced model would have these tags per item.
            top_style_score = STYLE_MAP.get(top.get('style', 'casual').lower(), 0)
            bottom_style_score = STYLE_MAP.get(bottom.get('style', 'casual').lower(), 0)
            
            # Penalize outfits where styles are very different (e.g., formal top with casual bottom)
            style_difference = abs(top_style_score - bottom_style_score)
            score -= style_difference * 2

            # Reward outfits that match the user's preference
            if abs(top_style_score - item_style_preference) == 0: score += 5
            if abs(bottom_style_score - item_style_preference) == 0: score += 5

            # 3. Personalization Score (add back body type logic if desired)
            # score += body_type_score([top, bottom], height, weight)

            # --- Update Best Outfit ---
            if score > best_score:
                best_score = score
                best_outfit = [top, bottom]
    
    if best_outfit:
        return [item['_id'] for item in best_outfit]
        
    # Fallback if no outfit scores positively (highly unlikely)
    return [random.choice(tops)['_id'], random.choice(bottoms)['_id']]

