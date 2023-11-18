from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
import numpy as np
from src.my_utils import encode_search_query, search_closest_image, compute_clip_features

app = FastAPI()

n_results_per_query = 1

class EmbeddingsRequest(BaseModel):
    embeddings: List[List[float]]
    num: int

@app.get("/get_closest_image/", response_class=JSONResponse)
async def get_closest_image(text: str):
    text_encoded = encode_search_query(text)
    return [search_closest_image(text_encoded, 1)['matches'][i]['id'] for i in range(1)]

@app.post("/get_closest_images/")
async def get_closest_images(data: EmbeddingsRequest):
    input_np_arr = np.array(data.embeddings)
    return [search_closest_image(input_np_arr, data.num)['matches'][i]['id'] for i in range(data.num)]
