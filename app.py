"""
Streamlit app
"""
import json
from os import getenv

import requests
import pandas as pd
import streamlit as st
from PIL import Image

from utils_image import encode_image, decode_image

DATA_DIR = "test_images/"
RESULT_DIR = "assets/"
API_TOKEN = getenv("API_TOKEN", "")


@st.cache
def load_samples():
    config = {
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

    for sample in config.values():
        sample["cam_image"] = Image.open(RESULT_DIR + sample["can_img"])
        sample["gc_image"] = Image.open(RESULT_DIR + sample["gc_image"])
        sample["ig_image"] = Image.open(RESULT_DIR + sample["ig_image"])

    return config


@st.cache
def recognize(image, url, token):
    encoded_img = encode_image(image)
    data = json.dumps({"encoded_image": encoded_img})

    headers = {"Content-Type": "application/json"}
    if token != "":
        headers.update({"X-Bedrock-Api-Token": token})

    response = requests.post(url, headers=headers, data=data)
    return response.json()


def show_results(panel, sample):
    # panel.subheader(f"`{prob:.2f}%`")
    panel.image(sample["cam_image"], caption="Grad-CAM Image", width=300)
    panel.image(sample["gc_image"], caption="Guided Grad-CAM Image", width=300)
    panel.image(sample["ig_image"], caption="Integrated Gradients Image", width=300)


@st.cache
def get_results(url, img):
    resp = requests.post(url, files={"image": img}, timeout=30)
    resp.raise_for_status()
    sample = resp.json()
    sample["cam_image"] = decode_image(sample["cam_image"])
    sample["gc_image"] = decode_image(sample["gc_image"])
    sample["ig_image"] = decode_image(sample["ig_image"])
    return sample


def get_endpoints():
    resp = requests.get(
        "https://api.bdrk.ai/v1/endpoint/",
        params={"project_id": "ihis-dev"},
        headers={"X-Bedrock-Access-Token": API_TOKEN},
    )
    return resp.json()["data"] if resp.ok else []


def image_recognize():
    # st.set_page_config(layout="wide")
    st.title("Chest X-ray Image Classification Demo")

    endpoints = [
        endpoint["fqdn"]
        for endpoint in get_endpoints()
        if ".pub." not in endpoint["fqdn"]
    ]

    samples = load_samples()
    select_ex = st.sidebar.selectbox("Select a sample image.", list(samples.keys()))
    uploaded_file = st.sidebar.file_uploader("Or upload an image.")
    select_ep = st.sidebar.multiselect("Choose model endpoints", endpoints, endpoints)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)
        # Parallelize over thread pool
        result = [get_results(f"https://{fqdn}", uploaded_file) for fqdn in select_ep]
    else:
        sample = samples[select_ex]
        left, right = st.beta_columns((3, 2))
        raw_img = Image.open(DATA_DIR + sample["raw_img"])
        left.image(raw_img, caption="Sample Image", width=400)

        right.subheader("Patient Attributes")
        df = pd.DataFrame(
            [("NRIC", "i****ljAlp6KR6x"), ("Gender", ""), ("Age", "-")],
            columns=["header", "Protected Data"],
        )
        df.set_index("header", inplace=True)
        right.table(df)

        result = [sample for _ in select_ep]

    # p_cols = st.beta_columns(len(endpoints) + 1)
    st.header("Probability of having COVID-19")
    p_cols = st.beta_columns(len(select_ep))
    for i, c in enumerate(p_cols):
        prob = result[i]["prob"] * 100
        c.subheader(f"{endpoints[i].split('.')[0]}: `{prob:.2f}%`")

    st.header("Explainability")
    st.write("Methods: Grad-CAM, Guided Grad-CAM and Integrated Gradients")
    st.write(
        "To visualise the regions of input that are 'important' for predictions from "
        "Convolutional Neural Network-based models."
    )
    e_cols = st.beta_columns(len(select_ep))
    for c in e_cols:
        show_results(c, result[i])


if __name__ == "__main__":
    image_recognize()
