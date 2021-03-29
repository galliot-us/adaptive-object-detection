from PIL import Image
from io import BytesIO
import json

from inference_single_img import inference

from fastapi import FastAPI
from fastapi import UploadFile, File

app = FastAPI()

@app.get("/api/ping")
async def ping():
    return {"health":"healthy!"}

@app.post("/api/inference")
async def inference_image(file: UploadFile = File(...)):
    # read the file
    rec_file = await file.read()
    img = Image.open(BytesIO(rec_file))
    result = inference(img)
    print(result)
    return(json.dumps(result))
