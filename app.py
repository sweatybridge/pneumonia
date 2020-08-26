import json
import requests

import streamlit as st
from PIL import Image

from utils_image import encode_image, decode_image


@st.cache
def recognize(image, url, token):
    encoded_img = encode_image(image)
    data = json.dumps({"encoded_image": encoded_img})

    headers = {"Content-Type": "application/json"}
    if token != "":
        headers.update({"X-Bedrock-Api-Token": token})

    response = requests.post(url, headers=headers, data=data)
    return response.json()


def image_recognize():
    st.title("Chest X-ray Image Classification Demo")

    url = st.text_input("Input API URL.", "http://127.0.0.1:5000/")
    token = st.text_input("Input token.")

    select_mode = st.selectbox("Choose a mode.", ["Select a sample image", "Upload an image"])

    uploaded_file = None
    if select_mode == "Select a sample image":
        samples = {
            "ex1": "covid-19-pneumonia-67.jpeg",
            "ex2": "covid-19-caso-82-1-8.png",
            "ex3": "41182_2020_203_Fig4_HTML.jpg",
            "ex4": "pneumococcal-pneumonia-day0.jpg",
        }

        select_eg = st.selectbox("Select a sample image.", [""] + list(samples.keys()))
        if select_ex != "":
            uploaded_file = "test_images/" + samples[select_eg]
    elif select_mode == "Upload an image":
        uploaded_file = st.file_uploader("Upload an image.")

    if uploaded_file is not None and url != "":
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

        response_json = recognize(image, url, token)
        prob = response_json["prob"] * 100
        cam_image = decode_image(response_json["cam_image"])
        gc_image = decode_image(response_json["gc_image"])
        ig_image = decode_image(response_json["ig_image"])

        st.subheader(f"Probability of having COVID-19 = `{prob:.2f}%`")
        st.header("Explainability")
        st.write("Methods: Grad-CAM, Guided Grad-CAM and Integrated Gradients")
        st.write("To visualise the regions of input that are 'important' for predictions from "
                 "Convolutional Neural Network-based models.")
        st.image(cam_image, caption="Grad-CAM Image", width=300)
        st.image(gc_image, caption="Guided Grad-CAM Image", width=300)
        st.image(ig_image, caption="Integrated Gradients Image", width=300)

    # st.sidebar.info(
    #     "**Note**: When querying Bedrock endpoints, for\n"
    #     "> `ConnectionError: ('Connection aborted.', BrokenPipeError(32, 'Broken pipe'))`\n\n"
    #     "change **http** to **https** in the API URL.")


if __name__ == "__main__":
    image_recognize()

