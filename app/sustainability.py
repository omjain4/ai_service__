import json
import os

TIPS_PATH = os.path.join(os.path.dirname(__file__), '..', 'static', 'tips.json')
with open(TIPS_PATH, 'r') as f:
    TIP_DATA = json.load(f)

def get_sustainability_tip(payload):
    item_category = payload.get("category", "").lower()
    material = payload.get("material", "").lower()
    matched = []
    for tip in TIP_DATA:
        tags = [t.lower() for t in tip['tags']]
        if (item_category and item_category in tags) or (material and material in tags):
            matched.append(tip['text'])
    if not matched:
        matched = [tip['text'] for tip in TIP_DATA if 'general' in [t.lower() for t in tip['tags']]]
    return {"tips": matched[:3]}
