import streamlit as st
import requests
import boto3
import os
from PIL import Image
from io import BytesIO
from src.my_utils import compute_clip_features
from dotenv import load_dotenv

st.title("Assignment 04 - Image Similarity")

load_dotenv()

def display_image_from_s3(text_path):
    s3 = boto3.client('s3', aws_access_key_id= os.environ.get("AWS_ACCESS_KEY"), aws_secret_access_key=os.environ.get("AWS_ACCESS_SECRET_KEY"))
    st.text("Closest Images")
    for i,j in enumerate(text_path):
        try:
            res = s3.get_object(Bucket = 'clip-model-images', Key = 'MEN/' + j + ".jpg")
            image_data = res['Body'].read()

            image = Image.open(BytesIO(image_data))
            st.image(image, use_column_width=True)
        except Exception as e:
            print(f"Error: {e}")

tab1, tab2 = st.tabs(['Text', 'Upload'])

with tab1:
    text = st.text_input("Enter a text to get closest similar Image")
    if text:
        response = requests.get(base_url + 'get_closest_image/', params={"text": text})
        if response.status_code == 200:
                st.write("Closest Image to the above text is:")
                display_image_from_s3(response.json())

with tab2:
    uploaded_file = st.file_uploader("Upload an image", type = ["jpg"])
    num = st.slider("Select number of images", 1, 5, 2)
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption = "Uploaded image", use_column_width = True)
        image_embeddings = compute_clip_features(image)
        image_embeddings_list = image_embeddings.tolist()
        try:
            response = requests.post(base_url + 'get_closest_images/', json= {"embeddings": image_embeddings_list, "num": num})
            display_image_from_s3(response.json())
        except requests.HTTPError as e:
            st.write(e.response.status_code)
            st.write(e.response.json())



