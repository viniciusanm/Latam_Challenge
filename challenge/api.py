import fastapi
from model import DelayModel

app = fastapi.FastAPI()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(features) -> dict:
    delay = DelayModel.predict(features)
    return
