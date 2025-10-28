import google.generativeai as genai
import requests
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash-exp')

def analyze_face_and_tone(image_url):
    """Analyze face and skin tone using Gemini Vision"""
    try:
        response = requests.get(image_url, timeout=10)
        img = Image.open(io.BytesIO(response.content))
        
        prompt = """Analyze this person's face and determine:
1. Skin tone (warm/cool/neutral)
2. Gender (male/female/unisex)
3. Face shape (oval/round/square/heart/diamond)
4. Age range (young/middle/mature)

Return only JSON format:
{
  "skin_tone": "warm/cool/neutral",
  "gender": "male/female/unisex", 
  "face_shape": "oval/round/square/heart/diamond",
  "age_range": "young/middle/mature"
}"""
        
        result = model.generate_content([prompt, img])
        return result.text.strip()
    except Exception as e:
        return '{"skin_tone": "neutral", "gender": "unisex", "face_shape": "oval", "age_range": "young"}'

def get_outfit_recommendations(wardrobe, style, skin_tone, gender, occasion):
    """Get outfit recommendations using Gemini"""
    try:
        wardrobe_text = "\n".join([f"- {item.get('name', 'Item')} ({item.get('category', 'Unknown')}, {item.get('color', 'Unknown color')})" for item in wardrobe])
        
        prompt = f"""Based on this wardrobe:
{wardrobe_text}

Recommend 3 complete outfits for:
- Style: {style}
- Skin tone: {skin_tone}
- Gender: {gender}
- Occasion: {occasion}

Return JSON format:
{{
  "outfits": [
    {{
      "name": "Outfit 1",
      "items": ["item1", "item2", "item3"],
      "reason": "why this works"
    }}
  ]
}}"""
        
        result = model.generate_content(prompt)
        return result.text.strip()
    except Exception as e:
        return '{"outfits": []}'