import json
import requests

import streamlit as st
from PIL import Image

from utils_image import encode_image, decode_image

SAMPLES = {
    "ex1": "covid-19-pneumonia-67.jpeg",
    "ex2": "pneumococcal-pneumonia-day0.jpg",
}


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

    url = st.text_input("Input API URL.")
    token = st.text_input("Input token.")

    select = st.selectbox("Choose a mode.", ["", "Select a sample image", "Upload an image"])

    if select == "Select a sample image":
        select_eg = st.selectbox("Select a sample image.", list(SAMPLES.keys()))
        uploaded_file = "test_images/" + SAMPLES[select_eg]
    else:
        uploaded_file = st.file_uploader("Upload an image.")

    if uploaded_file is not None and url != "":
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)

        response_json = recognize(image, url, token)
        prob = response_json["prob"] * 100
        cam_image = decode_image(response_json["cam_image"])
        gc_image = decode_image(response_json["gc_image"])

        st.subheader(f"Probability of having COVID-19 = `{prob:.2f}%`")
        st.header("Explainability")
        st.subheader("[Grad-CAM and Guided Grad-CAM](http://gradcam.cloudcv.org/)")
        st.write("To visualise the regions of input that are 'important' for predictions from "
                 "Convolutional Neural Network (CNN)-based models.")
        st.image(cam_image, caption="Grad-CAM Image", width=300)
        st.image(gc_image, caption="Guided Grad-CAM Image", width=300)

    st.sidebar.info(
        "**Note**: When querying Bedrock endpoints, for\n"
        "> `ConnectionError: ('Connection aborted.', BrokenPipeError(32, 'Broken pipe'))`\n\n"
        "change **http** to **https** in the API URL.")


if __name__ == "__main__":
    image_recognize()

