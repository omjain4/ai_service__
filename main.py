from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.predict import predict_category
from app.outfits import generate_outfit
from app.sustainability import get_sustainability_tip
from app.nlp import handle_user_question
from app.gemini_service import analyze_face_and_tone, get_outfit_recommendations
import uvicorn

app = FastAPI()

# CORS for integration with frontend/backend
origins = ["http://localhost:3000", "http://localhost:5001", "https://ai-wardrobe-backend-production.up.railway.app" , "https://stylezap-wardrobe-ai.vercel.app/"]  # Adjust to your ports
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/predict-category")
async def predict_category_endpoint(file: UploadFile = File(...)):
    try:
        return await predict_category(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-outfit")
async def generate_outfit_endpoint(payload: dict):
    try:
        return generate_outfit(payload)
    except Exception as e:
        # Return simple fallback if error
        return {
            "userItems": [],
            "suggestedItems": [],
            "score": 0,
            "error": str(e),
            "fallback_outfit": "White shirt + Black pants + Sneakers"
        }

@app.post("/sustainability-tip")
async def sustainability_tip_endpoint(payload: dict):
    try:
        return get_sustainability_tip(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-advisor")
async def nlp_endpoint(payload: dict):
    try:
        question = payload.get("question", "")
        return handle_user_question(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/personalize")
async def personalize_endpoint(payload: dict):
    try:
        # Use our manual outfit generation
        result = generate_outfit(payload)
        return result
    except Exception as e:
        # Simple fallback outfit
        return {
            "userItems": [],
            "suggestedItems": [
                {
                    "_id": "fallback_1",
                    "name": "White T-Shirt",
                    "category": "Tops",
                    "color": "white",
                    "style": "Casual",
                    "imageUrl": "https://placehold.co/300x400/white/black?text=T-Shirt"
                },
                {
                    "_id": "fallback_2",
                    "name": "Blue Jeans",
                    "category": "Bottoms",
                    "color": "blue",
                    "style": "Casual",
                    "imageUrl": "https://placehold.co/300x400/4169E1/white?text=Jeans"
                },
                {
                    "_id": "fallback_3",
                    "name": "White Sneakers",
                    "category": "Shoes",
                    "color": "white",
                    "style": "Casual",
                    "imageUrl": "https://placehold.co/300x400/white/black?text=Sneakers"
                }
            ],
            "score": 1,
            "message": "Fallback outfit generated",
            "error": str(e)
        }

@app.post("/analyze-face")
async def analyze_face_endpoint(payload: dict):
    try:
        image_url = payload.get('imageUrl')
        if not image_url:
            raise HTTPException(status_code=400, detail="imageUrl required")
        
        analysis = analyze_face_and_tone(image_url)
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
