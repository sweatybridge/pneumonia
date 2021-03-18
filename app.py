"""
Streamlit app
"""
import json
from os import getenv
from zlib import crc32
from concurrent.futures import ThreadPoolExecutor, wait

import requests
import pandas as pd
import streamlit as st
from PIL import Image

from utils_image import encode_image, decode_image

DATA_DIR = "test_images/"
RESULT_DIR = "assets/"
API_TOKEN = getenv("API_TOKEN", "")
EXECUTOR = ThreadPoolExecutor(max_workers=4)


def str_to_float(s):
    return float(crc32(s.encode()) & 0xFFFFFFFF) / 2 ** 32


def load_models():
    model_info = {}
    model_info["ihis"] = ["Normal", "COVID-19"]
    model_info["inhouse"] = ["Normal", "COVID-19"]
    model_info["chexnet"] = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    ]
    return model_info


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


def get_heatmap(fqdn, img, target=None):
    path = "/explain" if target is None else f"/explain/{target}"
    resp = requests.post(f"https://{fqdn}{path}", files={"image": img}, timeout=20)
    resp.raise_for_status()
    sample = resp.json()
    sample["cam_image"] = decode_image(sample["cam_image"])
    sample["gc_image"] = decode_image(sample["gc_image"])
    return sample


def get_results(fqdn, img):
    resp = requests.post(f"https://{fqdn}", files={"image": img}, timeout=10)
    resp.raise_for_status()
    return resp.json()


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
    model_info = load_models()
    select_ex = st.sidebar.selectbox("Select a sample image.", list(samples.keys()))
    uploaded_file = st.sidebar.file_uploader("Or upload an image.")
    select_ep = st.sidebar.multiselect("Choose model endpoints.", endpoints, endpoints)

    left, right = st.beta_columns((3, 2))
    if uploaded_file is not None:
        cache = f"/tmp/{uploaded_file.name}"
        with open(cache, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image = Image.open(cache)
        left.image(image, caption="Uploaded Image", width=400)

        # Parallelize over thread pool
        futures = [
            EXECUTOR.submit(get_results, fqdn, open(cache, "rb")) for fqdn in select_ep
        ]
        _ = wait(futures)
        result = [f.result() for f in futures]
    else:
        sample = samples[select_ex]
        raw_img = Image.open(DATA_DIR + sample["raw_img"])
        left.image(raw_img, caption="Sample Image", width=400)

        result = []
        for fqdn in select_ep:
            model_name = fqdn.split(".")[0]
            targets = model_info[model_name]
            result.append({k: str_to_float(k + str(select_ex)) for k in targets})

    # Draw patient metadata
    right.subheader("Patient Attributes")
    df = pd.DataFrame(
        [("NRIC", "i****ljAlp6KR6x"), ("Gender", ""), ("Age", "-")],
        columns=["header", "Protected Data"],
    )
    df.set_index("header", inplace=True)
    right.table(df)

    columns = set(k for r in result for k in r.keys())
    for i, r in enumerate(result):
        model_name = select_ep[i].split(".")[0]
        r["Model"] = model_name

    # Draw summary table
    st.header("Model Predictions")
    pred = pd.DataFrame(result)
    pred.set_index("Model", inplace=True)
    pred *= 100

    high_risk = pred[pred > 50].dropna(axis=1, how="all").max()
    for k, v in high_risk.sort_values(ascending=False)[:3].iteritems():
        st.markdown(f"The risk prediction score for `{k}` is {v:.1f}%.")

    table = st.dataframe(
        pred.style.set_precision(1)
        .set_na_rep("-")
        .highlight_max(axis=0)
        .applymap(lambda v: f"color: {'red' if v > 50 else 'black'}")
    )
    left, right = st.beta_columns(2)
    left.markdown("**Red**: model confidence > 50% (high risk)")
    right.markdown("**Yellow**: highest probability across models")

    # Draw heatmap on selection
    left, right = st.beta_columns((1, 3))
    left.header("Visualise Heatmap")
    select_target = right.selectbox("Target class:", [""] + list(columns))
    if select_target:
        if uploaded_file is not None:
            futures = [
                EXECUTOR.submit(get_heatmap, fqdn, open(cache, "rb"), select_target)
                for fqdn in select_ep
                if select_target in model_info[fqdn.split(".")[0]]
            ]
            _ = wait(futures)
            result = [f.result() for f in futures]
        else:
            result = [samples[select_ex] for _ in select_ep]

        p_cols = st.beta_columns(len(select_ep))
        for col, sample, fqdn in zip(p_cols, result, select_ep):
            model_name = fqdn.split(".")[0]
            prob = sample["prob"] * 100
            risk = "High Risk" if prob > 50 else "Low Risk"
            col.subheader(f"{model_name}: `{prob:.2f}%` ({risk})")

        e_cols = st.beta_columns(len(select_ep))
        for col, sample in zip(e_cols, result):
            col.image(sample["cam_image"], caption="Grad-CAM Image", width=300)
            col.image(sample["gc_image"], caption="Guided Grad-CAM Image", width=300)


if __name__ == "__main__":
    image_recognize()
