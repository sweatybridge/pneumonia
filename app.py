"""
Streamlit app
"""
import json
from os import getenv
from zlib import crc32
from concurrent.futures import ThreadPoolExecutor, wait
from urllib.parse import urlparse

import pydicom
import requests
import numpy as np
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
    scheme = "https" if "." in fqdn else "http"
    path = f"explain/{target}" if target else "explain"
    resp = requests.post(f"{scheme}://{fqdn}/{path}", files={"image": img}, timeout=20)
    resp.raise_for_status()
    sample = resp.json()
    sample["cam_image"] = decode_image(sample["cam_image"])
    sample["gc_image"] = decode_image(sample["gc_image"])
    sample["model"] = fqdn.split(".")[0]
    return sample


def get_results(fqdn, img):
    scheme = "https" if "." in fqdn else "http"
    resp = requests.post(f"{scheme}://{fqdn}", files={"image": img}, timeout=10)
    resp.raise_for_status()
    result = resp.json()
    result["model"] = fqdn.split(".")[0]
    return result


def get_endpoints():
    resp = requests.get(
        "https://api.bdrk.ai/v1/endpoint/",
        params={"project_id": "ihis-dev"},
        headers={"X-Bedrock-Access-Token": API_TOKEN},
    )
    localhost = [{"fqdn": "chexnet"}, {"fqdn": "inhouse"}]
    return resp.json()["data"] if resp.ok else localhost


def check_endpoint(url):
    try:
        # TODO: validate api compatibility
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        st.warning(exc)
        return False


def image_recognize():
    # st.set_page_config(layout="wide")
    st.title("Chest X-ray Image Classification Demo")

    # Load data
    endpoints = [
        endpoint["fqdn"]
        for endpoint in get_endpoints()
        if ".pub." not in endpoint["fqdn"]
    ]
    samples = load_samples()
    model_info = load_models()

    # Render sidebar
    select_ex = st.sidebar.selectbox("Select a sample image.", list(samples.keys()))
    uploaded_file = st.sidebar.file_uploader("Or upload an image.")
    # select_ep = st.sidebar.multiselect("Choose model endpoints.", endpoints, endpoints)
    exp_s = st.sidebar.beta_expander("Available Model Endpoints")
    external = exp_s.text_input("Enter an external URL.", max_chars=63)
    st.sidebar.markdown("###")  ## add margin
    check_ep = [st.sidebar.checkbox(fqdn, value=True) for fqdn in endpoints]

    if external:
        with exp_s:
            ok = check_endpoint(external)
        if ok:
            fqdn = urlparse(external).hostname
            endpoints.append(fqdn)
            check_ep.append(st.sidebar.checkbox(fqdn))

    left, right = st.beta_columns((3, 2))
    if uploaded_file is not None:
        cache = f"/tmp/{uploaded_file.name}"
        with open(cache, "wb") as f:
            f.write(uploaded_file.getbuffer())

        metadata = []
        if uploaded_file.name.lower().endswith(".dcm"):
            dcm = pydicom.dcmread(cache)
            attributes = ["PatientID", "PatientAge", "PatientSex", "StudyDescription", "Modality"]
            for tag in attributes:
                if tag not in dcm:
                    continue
                metadata.append((tag, getattr(dcm, tag)))
            image = dcm.pixel_array
            # .npy extension will be appended by numpy
            np.save(cache, image)
            cache += ".npy"
        else:
            image = Image.open(cache)

        left.image(image, caption="Uploaded Image", width=400)
        # Parallelize over thread pool
        futures = [
            EXECUTOR.submit(get_results, fqdn, open(cache, "rb"))
            for fqdn, chosen in zip(endpoints, check_ep)
            if chosen
        ]
        _ = wait(futures)
        result = [f.result() for f in futures]
    else:
        sample = samples[select_ex]
        raw_img = Image.open(DATA_DIR + sample["raw_img"])
        left.image(raw_img, caption="Sample Image", width=400)
        # Create dummy data
        metadata = [("NRIC", "i****ljAlp6KR6x"), ("Gender", ""), ("Age", "-")]
        result = []
        for fqdn, chosen in zip(endpoints, check_ep):
            model_name = fqdn.split(".")[0]
            if not chosen or model_name not in model_info:
                continue
            targets = model_info[model_name]
            r = {k: str_to_float(k + str(select_ex)) for k in targets}
            r["model"] = model_name
            result.append(r)

    # Render patient metadata
    right.subheader("Patient Attributes")
    df = pd.DataFrame(metadata, columns=["header", "Protected Data"])
    df.set_index("header", inplace=True)
    right.table(df)

    # Render summary text and table
    st.header("Model Prediction")
    pred = pd.DataFrame(result)
    pred.set_index("model", inplace=True)
    pred *= 100

    high_risk = pred[pred > 50].dropna(axis=1, how="all").max()
    for k, v in high_risk.sort_values(ascending=False)[:3].iteritems():
        st.markdown(f"The risk prediction score for `{k}` is {v:.1f}%.")

    exp = st.beta_expander("Confidence by model and target class")
    left, right = exp.beta_columns(2)
    left.markdown("**Red**: confidence > 50% (high risk)")
    right.markdown("**Yellow**: highest class probability")
    table = st.dataframe(
        pred.style.set_precision(1)
        .set_na_rep("-")
        .highlight_max(axis=0)
        .applymap(lambda v: f"color: {'red' if v > 50 else 'black'}")
    )

    # Render heatmap on selection
    st.text("")  # add margin
    left, right = st.beta_columns((1, 2))
    left.header("Heatmap Visualization")
    columns = set(k for r in result for k in r.keys() if k != "model")
    select_tg = right.selectbox(
        "Target class:",
        [None] + list(columns),
        format_func=lambda opt: opt or "Auto (top score of every model)",
    )

    if uploaded_file is not None:
        futures = [
            EXECUTOR.submit(get_heatmap, fqdn, open(cache, "rb"), select_tg)
            for fqdn, chosen in zip(endpoints, check_ep)
            if chosen and (not select_tg or select_tg in model_info[fqdn.split(".")[0]])
        ]
        _ = wait(futures)
        result = [f.result() for f in futures]
    else:
        result = []
        for fqdn, chosen in zip(endpoints, check_ep):
            if not chosen:
                continue
            model_name = fqdn.split(".")[0]
            if not select_tg:
                prob = pred[pred.index == model_name].max(axis=1).iloc[0] / 100
            elif select_tg in model_info[model_name]:
                prob = str_to_float(select_tg + str(select_ex))
            else:
                continue
            r = samples[select_ex].copy()
            r["prob"] = prob
            r["model"] = model_name
            result.append(r)

    # Layout images
    p_cols = st.beta_columns(sum(check_ep))
    for col, sample in zip(p_cols, result):
        model_name = sample["model"]
        prob = sample["prob"] * 100
        if select_tg:
            risk = "High Risk" if prob > 50 else "Low Risk"
        else:
            risk = pred[pred.index == model_name].idxmax(axis=1).iloc[0]
        col.subheader(f"{model_name}: `{prob:.1f}%` ({risk})")
        col.image(sample["cam_image"], caption="Grad-CAM Image", width=330)
        col.image(sample["gc_image"], caption="Guided Grad-CAM Image", width=330)


if __name__ == "__main__":
    image_recognize()
