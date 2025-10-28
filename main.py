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
        raise HTTPException(status_code=500, detail=str(e))

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
        # Use both our model and Gemini
        our_result = generate_outfit(payload)
        
        # Get Gemini recommendations as backup
        gemini_result = get_outfit_recommendations(
            payload.get('wardrobe', []),
            payload.get('style', 'casual'),
            payload.get('body_color', 'neutral'),
            payload.get('gender', 'unisex'),
            payload.get('style', 'casual')
        )
        
        return {
            "our_model": our_result,
            "gemini_suggestions": gemini_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
