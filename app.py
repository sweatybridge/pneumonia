"""
Streamlit app
"""
import json
import requests

import streamlit as st
from PIL import Image

from utils_image import encode_image, decode_image

DATA_DIR = "test_images/"
RESULT_DIR = "assets/"


@st.cache
def recognize(image, url, token):
    encoded_img = encode_image(image)
    data = json.dumps({"encoded_image": encoded_img})

    headers = {"Content-Type": "application/json"}
    if token != "":
        headers.update({"X-Bedrock-Api-Token": token})

    response = requests.post(url, headers=headers, data=data)
    return response.json()


def show_results(prob, cam_image, gc_image, ig_image):
    st.subheader(f"Probability of having COVID-19 = `{prob:.2f}%`")
    st.header("Explainability")
    st.write("Methods: Grad-CAM, Guided Grad-CAM and Integrated Gradients")
    st.write("To visualise the regions of input that are 'important' for predictions from "
             "Convolutional Neural Network-based models.")
    st.image(cam_image, caption="Grad-CAM Image", width=300)
    st.image(gc_image, caption="Guided Grad-CAM Image", width=300)
    st.image(ig_image, caption="Integrated Gradients Image", width=300)


def image_recognize():
    st.title("Chest X-ray Image Classification Demo")

    select_mode = st.selectbox("Choose a mode.", ["Select a sample image", "Upload an image"])

    if select_mode == "Select a sample image":
        samples = {
            "ex1": {
                "raw_img": "covid-19-pneumonia-67.jpeg",
                "prob": 0.8808,
                "can_img": "ex1_cam_img.jpeg",
                "gc_image": "ex1_gc_img.jpeg",
                "ig_image": "ex1_ig_img.jpeg",
            },
            "ex2": {
                "raw_img": "covid-19-caso-82-1-8.png",
                "prob": 0.7102,
                "can_img": "ex2_cam_img.jpeg",
                "gc_image": "ex2_gc_img.jpeg",
                "ig_image": "ex2_ig_img.jpeg",
            },
            "ex3": {
                "raw_img": "41182_2020_203_Fig4_HTML.jpg",
                "prob": 0.9706,
                "can_img": "ex3_cam_img.jpeg",
                "gc_image": "ex3_gc_img.jpeg",
                "ig_image": "ex3_ig_img.jpeg",
            },
            "ex4": {
                "raw_img": "pneumococcal-pneumonia-day0.jpg",
                "prob": 0.2417,
                "can_img": "ex4_cam_img.jpeg",
                "gc_image": "ex4_gc_img.jpeg",
                "ig_image": "ex4_ig_img.jpeg",
            },
        }

        select_ex = st.selectbox("Select a sample image.", [""] + list(samples.keys()))
        if select_ex != "":
            sample = samples[select_ex]
            raw_img = Image.open(DATA_DIR + sample["raw_img"])
            st.image(raw_img, caption="Sample Image", width=400)

            prob = sample["prob"] * 100
            cam_image = Image.open(RESULT_DIR + sample["can_img"])
            gc_image = Image.open(RESULT_DIR + sample["gc_image"])
            ig_image = Image.open(RESULT_DIR + sample["ig_image"])
            show_results(prob, cam_image, gc_image, ig_image)
            
    elif select_mode == "Upload an image":
        url = st.text_input("Input API URL.")
        token = st.text_input("Input token.")
        uploaded_file = st.file_uploader("Upload an image.")

        if uploaded_file is not None and url != "":
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)

            response_json = recognize(image, url, token)
            prob = response_json["prob"] * 100
            cam_image = decode_image(response_json["cam_image"])
            gc_image = decode_image(response_json["gc_image"])
            ig_image = decode_image(response_json["ig_image"])
            show_results(prob, cam_image, gc_image, ig_image)


if __name__ == "__main__":
    image_recognize()
