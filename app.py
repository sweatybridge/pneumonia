import base64
import json
import requests
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image


def encode_image(image):
    """Encode an image to base64 encoded bytes.
    Args:
        image: PIL.PngImagePlugin.PngImageFile
    Returns:
        base64 encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format="png")
    base64_bytes = base64.b64encode(buffered.getvalue())
    return base64_bytes.decode("utf-8")


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

    st.sidebar.info(
        "**Note**: If you see the error message:\n"
        "> `ConnectionError: ('Connection aborted.', BrokenPipeError(32, 'Broken pipe'))`\n\n"
        "please change from **http** to **https** in the API URL.")
    
    url = st.text_input("Input API URL.")
    token = st.text_input("Input token.")

    uploaded_file = st.file_uploader("Upload an image.")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        response_json = recognize(image, url, token)
        prob = response_json["prob"] * 100
        st.subheader(f"Probability of having COVID-19 = `{prob:.2f}%`")

        st.image(image, caption="Uploaded Image", use_column_width=True)
        

if __name__ == "__main__":
    image_recognize()

