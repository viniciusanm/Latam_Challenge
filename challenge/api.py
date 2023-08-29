import fastapi
from model import DelayModel
import uvicorn

import json


app = fastapi.FastAPI()

@app.get("/health", status_code=200)
def get_health():
    return {"status": "OK"}

@app.post("/predict", status_code=200)
async def post_predict(features) -> dict:

    features = json.loads(features)

    features, target = DelayModel.preprocess(features)

   
    delay = DelayModel.predict(features)
    return delay

if __name__ == '__main__':
    uvicorn.run(app)
