from fastapi import FastAPI, Request
import joblib, requests, io
from functools import lru_cache

app = FastAPI()

@lru_cache(maxsize=10)
def load_model(model_url):
    print(f"📦 Loading model: {model_url}")
    content = requests.get(model_url).content
    return joblib.load(io.BytesIO(content))

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    model_url = data.get("model_url")
    features = data.get("features")

    if not model_url or not features:
        return {"error": "Eksik parametre (model_url veya features)"}

    try:
        model = load_model(model_url)
        prediction = model.predict_proba([features])[0][1]
        return {"success": True, "probability": round(float(prediction), 4)}
    except Exception as e:
        return {"error": str(e)}
