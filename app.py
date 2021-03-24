"""
Streamlit app
"""
import json
from os import getenv
from zlib import crc32
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait
from urllib.parse import urlparse

import pydicom
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from inhouse.utils_image import decode_image

DATA_DIR = Path("test_images")
RESULT_DIR = Path("assets")
API_TOKEN = getenv("API_TOKEN", "")
EXECUTOR = ThreadPoolExecutor(max_workers=4)


def str_to_float(s):
    return float(crc32(s.encode()) & 0xFFFFFFFF) / 2 ** 32


def load_samples():
    config = {
        "ex1": {
            "raw_img": "covid-19-pneumonia-67.jpeg",
            "prob": 0.8808,
            "cam_img": "ex1_cam_img.jpeg",
            "gc_image": "ex1_gc_img.jpeg",
            "ig_image": "ex1_ig_img.jpeg",
        },
        "ex2": {
            "raw_img": "covid-19-caso-82-1-8.png",
            "prob": 0.7102,
            "cam_img": "ex2_cam_img.jpeg",
            "gc_image": "ex2_gc_img.jpeg",
            "ig_image": "ex2_ig_img.jpeg",
        },
        "ex3": {
            "raw_img": "41182_2020_203_Fig4_HTML.jpg",
            "prob": 0.9706,
            "cam_img": "ex3_cam_img.jpeg",
            "gc_image": "ex3_gc_img.jpeg",
            "ig_image": "ex3_ig_img.jpeg",
        },
        "ex4": {
            "raw_img": "pneumococcal-pneumonia-day0.jpg",
            "prob": 0.2417,
            "cam_img": "ex4_cam_img.jpeg",
            "gc_image": "ex4_gc_img.jpeg",
            "ig_image": "ex4_ig_img.jpeg",
        },
    }
    model_info = {
        "inhouse": ["Normal", "COVID-19"],
        "chexnet": [
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
        ],
    }
    return config, model_info


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
    localhost = [{"fqdn": "inhouse"}, {"fqdn": "chexnet"}, {"fqdn": "ihis"}]
    return resp.json()["data"] if resp.ok else localhost


def check_endpoint(url, panel=None):
    try:
        # TODO: validate api compatibility
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        if panel is not None:
            panel.warning(exc)
        return False


def image_recognize():
    # st.set_page_config(layout="wide")
    st.title("Chest X-ray Diagnosis Demo")

    # Load data
    endpoints = [
        endpoint["fqdn"]
        for endpoint in get_endpoints()
        if ".pub." not in endpoint["fqdn"]
    ]
    samples, model_info = load_samples()

    # Render sidebar
    select_ex = st.sidebar.selectbox("Select a sample image.", list(samples.keys()))
    uploaded_file = st.sidebar.file_uploader("Or upload an image.")
    exp_s = st.sidebar.beta_expander("Available Model Endpoints")
    external = exp_s.text_input("Enter an external URL.", max_chars=63)
    st.sidebar.markdown("###")  ## add margin
    check_ep = [
        st.sidebar.checkbox(
            fqdn,
            value=check_endpoint(
                f"{'https' if '.' in fqdn else 'http'}://{fqdn}/metrics"
            ),
        )
        for fqdn in endpoints
    ]

    if external:
        ok = check_endpoint(external, exp_s)
        if ok:
            fqdn = urlparse(external).hostname
            endpoints.append(fqdn)
            check_ep.append(st.sidebar.checkbox(fqdn))

    # Parse uploaded file or use sample image
    if uploaded_file is not None:
        cache_img = f"/tmp/{uploaded_file.name}"
        with open(cache_img, "wb") as f:
            f.write(uploaded_file.getbuffer())

        metadata = []
        if uploaded_file.name.lower().endswith(".dcm"):
            dcm = pydicom.dcmread(cache_img)
            attributes = [
                "PatientID",
                "PatientAge",
                "PatientSex",
                "StudyDescription",
                "Modality",
            ]
            for tag in attributes:
                if tag not in dcm:
                    continue
                metadata.append((tag, getattr(dcm, tag)))
            image = dcm.pixel_array
            # .npy extension will be appended by numpy
            np.save(cache_img, image)
            cache_img += ".npy"
        else:
            image = Image.open(cache_img)
        caption = "Uploaded Image"
    else:
        cache_img = DATA_DIR / samples[select_ex]["raw_img"]
        image = Image.open(cache_img)
        caption = "Sample Image"
        metadata = [("NRIC", "i****ljAlp6KR6x"), ("Gender", ""), ("Age", "-")]

    # Render patient metadata
    left, right = st.beta_columns((3, 2))
    left.image(image, caption=caption, width=400)
    right.subheader("Patient Attributes")
    df = pd.DataFrame(metadata, columns=["header", "Protected Data"])
    df.set_index("header", inplace=True)
    right.table(df)

    selected_ep = [fqdn for fqdn, chosen in zip(endpoints, check_ep) if chosen]
    if API_TOKEN:
        # Make parallel requests to endpoints
        futures = [
            EXECUTOR.submit(get_results, fqdn, open(cache_img, "rb"))
            for fqdn in selected_ep
        ]
        _ = wait(futures)
        result = [f.result() for f in futures if not f.exception()]
    else:
        # Create dummy data
        result = []
        for fqdn in selected_ep:
            model_name = fqdn.split(".")[0]
            if model_name not in model_info:
                continue
            targets = model_info[model_name]
            r = {k: str_to_float(k + str(select_ex)) for k in targets}
            r["model"] = model_name
            result.append(r)

    # Render summary text and table
    render_marketplace(result, cache_img, selected_ep, select_ex)


def render_marketplace(result, cache_img, selected_ep, select_ex):
    st.header("Model Marketplace Predictions")
    if not result:
        st.markdown("No predictions available.")
        return

    pred = pd.DataFrame(result)
    pred.set_index("model", inplace=True)
    pred.drop([c for c in pred.columns if c.lower() == "normal"], axis=1, inplace=True)
    pred *= 100

    exp = st.beta_expander("Confidence by model (row) and target class (column)")
    left, right = exp.beta_columns(2)
    left.markdown("**Red**: confidence > 50% (high risk)")
    right.markdown("**Yellow**: highest class probability")
    table = st.dataframe(
        pred.style.set_precision(1)
        .set_na_rep("-")
        .highlight_max(axis=0)
        .applymap(lambda v: f"color: {'red' if v > 50 else 'black'}")
    )

    for k, v in pred.idxmax(axis=1).iteritems():
        s = pred[v].loc[k]
        st.markdown(f"The risk prediction score by `{k}` model for `{v}` is {s:.1f}%.")
    # high_risk = pred[pred > 50].dropna(axis=1, how="all").max()
    # for k, v in high_risk.sort_values(ascending=False)[:3].iteritems():
    #     st.markdown(f"The risk prediction score for `{k}` is {v:.1f}%.")

    # Render heatmap on selection
    st.text("")  # add margin
    left, right = st.beta_columns((1, 2))
    left.header("Heatmap Visualization")
    select_tg = right.selectbox(
        "Target class:",
        [None] + list(pred.columns),
        format_func=lambda opt: opt or "Auto (top score of every model)",
    )

    if API_TOKEN:
        futures = [
            EXECUTOR.submit(get_heatmap, fqdn, open(cache_img, "rb"), select_tg)
            for fqdn in selected_ep
            if not select_tg
            or pred[pred.index == fqdn.split(".")[0]][select_tg].notna().all()
        ]
        _ = wait(futures)
        result = [f.result() for f in futures if not f.exception()]
    else:
        result = []
        samples, model_info = load_samples()
        for fqdn in selected_ep:
            model_name = fqdn.split(".")[0]
            if model_name not in model_info:
                continue
            if not select_tg:
                prob = pred[pred.index == model_name].max(axis=1).iloc[0] / 100
            elif select_tg in model_info[model_name]:
                prob = str_to_float(select_tg + str(select_ex))
            else:
                continue
            sample = samples[select_ex]
            result.append(
                {
                    "model": model_name,
                    "prob": prob,
                    "cam_image": Image.open(
                        RESULT_DIR / model_name / sample["cam_img"]
                    ),
                    "gc_image": Image.open(
                        RESULT_DIR / model_name / sample["gc_image"]
                    ),
                }
            )

    # Layout images
    p_cols = st.beta_columns(len(result))
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
