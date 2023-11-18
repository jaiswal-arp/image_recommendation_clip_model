import torch
import clip
import pinecone
from PIL import Image

pinecone.init(api_key="4628f9ab-aef0-4667-8568-c6a6b29567e8", environment="gcp-starter")
index_name = "imageindex"
index = pinecone.Index(index_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


def encode_search_query(search_query):
    with torch.no_grad():
        text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    return text_encoded

def search_closest_image(text_encoded, num):
    return index.query(
        vector=text_encoded.tolist(),
        top_k=num,
        include_values=True
    )

def compute_clip_features(image):
    images_preprocessed = torch.stack((preprocess(image),)).to(device)

    with torch.no_grad():
        images_features = model.encode_image(images_preprocessed)
        images_features /= images_features.norm(dim=-1, keepdim=True)

    return images_features