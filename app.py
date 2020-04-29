import base64
import json
import requests

import numpy as np
import streamlit as st
from PIL import Image


def encode_image(array, dtype=np.uint8):
    """Encode an array to base64 encoded string or bytes.
    Args:
        array: numpy.array
        dtype
    Returns:
        base64 encoded string
    """
    if array is None:
        return None
    return base64.b64encode(np.asarray(array, dtype=dtype)).decode("utf-8")

    
def main():
    max_width = 900 #st.sidebar.slider("Set page width", min_value=700, max_value=1500, value=1000, step=20)
    st.markdown(
        f"""
        <style>
        .reportview-container .main .block-container{{
            max-width: {max_width}px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("COVID-19 Chest X-ray Image Classification")
    
    token = st.text_input("Input token.")
    
    uploaded_file = st.file_uploader("Upload an image.")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        img = np.asarray(image.convert("RGB"))
        img_shape = img.shape
        encoded_img = encode_image(img.ravel())
        data = json.dumps({"encoded_image": encoded_img, "image_shape": img_shape})

        url = "http://localhost:8080"  # 'https://wild-mountain-4441.pub.playground.bdrk.ai'
        headers = {"Content-Type": "application/json"}
        if token:
            headers.update({"X-Bedrock-Api-Token": token})
        
        response = requests.post(url, headers=headers, data=data)
        prob = response.json()["prob"]
        st.subheader(f"Probability of having COVID-19 = {prob:.6f}")
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        

if __name__ == "__main__":
    main()
