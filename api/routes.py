from flask import Blueprint, request, jsonify
from .image_tagging import predict_img_tags
from .recommendation import recommend_outfit

api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400
    file = request.files['file']
    prediction = predict_img_tags(file)
    return jsonify(prediction)

@api_blueprint.route('/generate-outfit', methods=['POST'])
def generate():
    data = request.get_json()
    wardrobe = data.get('wardrobe', [])
    style = data.get('style')
    weather = data.get('weather')
    outfit = recommend_outfit(wardrobe, style, weather)
    return jsonify({'selected_outfit_ids': outfit})
