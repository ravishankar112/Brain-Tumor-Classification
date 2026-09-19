import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


# APP CONFIGURATION (edit these to change model / labels)

APP_TITLE = "NeuroVision AI"
APP_TAGLINE = "Brain MRI Image Classification using VGG19 Deep Learning"
DEVELOPER_NAME = "Ravi Shankar Kumar"

MODEL_PATH = "brain_tumor_vgg19_final.keras"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_ICONS = {"Glioma": "🔴", "Meningioma": "🟠", "No Tumor": "🟢", "Pituitary": "🔵"}
TEST_ACCURACY = 92.85

DATASET_STATS = {
    "Total Images": 10000,
    "Training Images": 8000,
    "Validation Images": 1000,
    "Testing Images": 1000,
}

# Confidence bands used to label how reliable a single prediction looks.
# (Purely a UX signal — not a statistical guarantee.)
CONFIDENCE_BANDS = [
    (90.0, "High Confidence", "#0f9d58"),
    (70.0, "Moderate Confidence", "#e8a33d"),
    (0.0, "Low Confidence", "#d64545"),
]

MAX_HISTORY_ITEMS = 5

FEEDBACK_FILE = "feedback.csv"


# PAGE SETUP

st.set_page_config(
    page_title=f"{APP_TITLE} | Brain MRI Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# STYLES (kept in one place, easy to scan / edit)

def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* ------------------------------------------------------------
           COLOR SYSTEM — change the whole app's look from right here.
           A fresh indigo / violet + amber palette on a clean neutral
           background, instead of the old all-blue/teal theme.
           ------------------------------------------------------------ */
        :root {
            --bg-1: #ffffff;
            --bg-2: #f4f9fb;
            --bg-3: #eef6f8;

            --primary-dark: #003f5c;
            --primary: #0077b6;
            --primary-light: #48cae4;
            --accent: #2a9d8f;
            --accent-2: #e63946;

            --text-main: #1b2b34;
            --text-muted: #56707d;
            --text-soft: #7a92a0;

            --surface: #ffffff;
            --surface-alt: #eaf6f8;
            --border: #d6e8ee;
            --shadow: rgba(0, 63, 92, 0.10);
        }

        /* ------------------------------------------------------------
           FORCE LIGHT APPEARANCE
           Fixes the "text turns white / invisible" issue that happens
           when the visitor's browser or OS is set to dark mode and
           Streamlit's default dark theme text (white) lands on top of
           our light card backgrounds. Every plain text element below
           gets an explicit dark color so it is never theme-dependent.
           ------------------------------------------------------------ */
        html, body { color-scheme: light only; }

        .stApp {
            background: linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 45%, var(--bg-3) 100%);
            color: var(--text-main) !important;
        }

        .stApp, .stApp p, .stApp span, .stApp li, .stApp div,
        .stMarkdown, .stMarkdown p, .stText, .stCaption,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMetricDelta"] {
            color: var(--text-main) !important;
        }

        /* Keep the deliberately-light text (hero banner, sidebar) as-is
           — these selectors are more specific, so they still win. */
        .hero-box, .hero-box *, .footer-box, .footer-box *,
        section[data-testid="stSidebar"], section[data-testid="stSidebar"] * {
            color: white !important;
        }

        .block-container { max-width: 1280px; padding-top: 1.3rem; padding-bottom: 2rem; }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--primary-dark) 0%, var(--primary) 60%, var(--accent) 100%);
        }
        section[data-testid="stSidebar"] * { color: white !important; }

        .sidebar-brand { text-align: center; padding: 12px 5px 20px 5px; }
        .sidebar-icon { font-size: 58px; margin-bottom: 5px; }
        .sidebar-title { font-size: 25px; font-weight: 900; }
        .sidebar-subtitle { color: #d9f4f7; font-size: 13px; font-weight: 600; }
        .sidebar-section {
            font-size: 13px; font-weight: 800; letter-spacing: 1px;
            margin-top: 18px; margin-bottom: 8px; color: #9fe8ee !important;
        }

        .hero-box {
            background: linear-gradient(120deg, var(--primary-dark) 0%, var(--primary) 55%, var(--accent) 100%);
            border-radius: 28px; padding: 48px 35px; text-align: center;
            box-shadow: 0 18px 45px var(--shadow); margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.18);
            position: relative; overflow: hidden;
        }
        .hero-icon { font-size: 68px; margin-bottom: 5px; }
        .hero-title { color: white !important; font-size: 48px; font-weight: 950; letter-spacing: 0.5px; }
        .hero-subtitle { color: #eafcff !important; font-size: 19px; font-weight: 650; margin-top: 8px; }
        .developer-badge {
            display: inline-block; margin-top: 20px; padding: 10px 22px; border-radius: 50px;
            background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.35);
            color: white !important; font-size: 14px; font-weight: 750;
        }

        .section-title {
            color: var(--primary-dark) !important; font-size: 28px; font-weight: 950;
            margin-top: 30px; margin-bottom: 17px; padding-left: 13px;
            border-left: 6px solid var(--accent);
        }

        .card {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: 20px; padding: 25px; box-shadow: 0 8px 25px var(--shadow);
            height: 100%;
        }
        .card-title { color: var(--primary-dark); font-size: 20px; font-weight: 900; margin-bottom: 10px; }
        .card-text { color: var(--text-muted); font-size: 15px; line-height: 1.75; font-weight: 550; }

        [data-testid="stMetric"] {
            background: linear-gradient(145deg, #ffffff, var(--surface-alt));
            border: 1px solid var(--border); border-radius: 19px; padding: 18px;
            box-shadow: 0 8px 22px var(--shadow);
        }
        [data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-weight: 800 !important; }
        [data-testid="stMetricValue"] { color: var(--primary) !important; font-weight: 950 !important; }

        .stButton > button {
            background: linear-gradient(90deg, var(--primary), var(--primary-light));
            color: white !important; border: none; border-radius: 13px;
            min-height: 48px; font-weight: 850; font-size: 15px;
            box-shadow: 0 6px 15px rgba(0, 119, 182, 0.28);
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, var(--primary-dark), var(--primary));
            color: white !important; transform: translateY(-1px);
            box-shadow: 0 10px 20px rgba(0, 119, 182, 0.32);
        }

        /* Primary "Analyze" button uses the medical-teal accent to stand out */
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, var(--accent), #21867a);
            box-shadow: 0 6px 15px rgba(42, 157, 143, 0.35);
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(90deg, #21867a, var(--accent));
        }

        [data-testid="stFileUploader"] {
            background: var(--surface); border: 2px dashed var(--primary-light);
            border-radius: 20px; padding: 20px; box-shadow: 0 7px 20px var(--shadow);
        }

        .result-box {
            background: linear-gradient(135deg, #e6f4f6, #eaf7f5);
            border: 2px solid var(--primary-light); border-radius: 24px; padding: 32px; text-align: center;
            box-shadow: 0 12px 28px var(--shadow);
        }
        .result-heading { color: var(--text-soft); font-size: 14px; font-weight: 900; letter-spacing: 1.5px; }
        .result-class { color: var(--primary-dark); font-size: 40px; font-weight: 950; margin-top: 5px; }
        .result-confidence { color: var(--accent); font-size: 22px; font-weight: 900; margin-top: 8px; }

        .workflow-card {
            background: linear-gradient(145deg, #ffffff, var(--surface-alt));
            border: 1px solid var(--border); border-radius: 19px; padding: 23px 12px;
            text-align: center; min-height: 145px; box-shadow: 0 7px 20px var(--shadow);
        }
        .workflow-number {
            color: white; background: linear-gradient(135deg, var(--primary), var(--accent));
            width: 35px; height: 35px; line-height: 35px; border-radius: 50%;
            margin: 0 auto 10px auto; font-weight: 900;
        }
        .workflow-icon { font-size: 30px; }
        .workflow-title { color: var(--primary-dark); font-size: 16px; font-weight: 900; margin-top: 7px; }
        .workflow-text { color: var(--text-muted); font-size: 12px; font-weight: 600; margin-top: 5px; }

        .category-card {
            background: var(--surface); border-radius: 19px; padding: 22px;
            border-left: 6px solid var(--accent); box-shadow: 0 7px 20px var(--shadow);
            min-height: 135px; margin-bottom: 16px;
        }
        .category-title { color: var(--primary-dark); font-size: 20px; font-weight: 900; }
        .category-description { color: var(--text-muted); font-size: 14px; line-height: 1.6; font-weight: 550; margin-top: 7px; }

        .notice {
            background: linear-gradient(135deg, #fff3e0, #fffaf0);
            border: 1px solid #ffcf94; border-radius: 17px; padding: 18px 22px;
            color: #7a4a12; font-weight: 650; line-height: 1.6;
        }

        .footer-box {
            background: linear-gradient(135deg, var(--primary-dark), var(--primary), var(--accent));
            border-radius: 24px; padding: 38px 25px; text-align: center; margin-top: 45px;
            box-shadow: 0 15px 35px var(--shadow);
        }
        .footer-title { color: white; font-size: 25px; font-weight: 950; }
        .footer-main { color: #eafcff; font-size: 15px; font-weight: 650; margin-top: 8px; }
        .footer-developer { color: white; font-size: 16px; font-weight: 900; margin-top: 18px; }
        .footer-warning { color: #ffe8b0; font-size: 13px; font-weight: 650; line-height: 1.7; margin-top: 18px; }
        .footer-line { width: 70%; height: 1px; background: rgba(255,255,255,0.25); margin: 22px auto; }

        button[data-baseweb="tab"] { color: var(--primary-dark) !important; font-weight: 850 !important; }
        label { color: var(--primary-dark) !important; font-weight: 800 !important; }
        hr { border-color: var(--border); }

        /* Scrollbar polish (cosmetic, harmless if unsupported) */
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: var(--bg-3); }
        ::-webkit-scrollbar-thumb { background: var(--primary-light); border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# MODEL LOADING


@st.cache_resource(show_spinner="Loading NeuroVision AI model...")
def load_model():
    """Load and cache the trained Keras model."""
    return tf.keras.models.load_model(MODEL_PATH)


def get_model():
    """Load the model, showing a friendly error if it fails."""
    try:
        return load_model()
    except Exception as error:
        st.error("The trained model could not be loaded.")
        st.info(
            f"Please make sure that **'{MODEL_PATH}'** is located "
            "in the same folder as this app file."
        )
        with st.expander("Technical error details"):
            st.code(str(error))
        st.stop()


# ============================================================
# IMAGE HELPERS
# ============================================================

def check_image_quality(image: Image.Image) -> bool:
    """
    Basic sanity check to flag obviously unusable images
    (near-blank, over/under-exposed). This is NOT an
    MRI-vs-non-MRI classifier.
    """
    image_array = np.array(image)
    gray = np.mean(image_array, axis=2)
    mean_value = float(np.mean(gray))
    std_value = float(np.std(gray))

    if std_value < 15:
        return False
    if mean_value > 245 or mean_value < 5:
        return False
    return True


# ============================================================
# ADDED MRI IMAGE CHECK
# ============================================================

def check_mri_like_image(image: Image.Image) -> bool:
    """
    Basic heuristic check to flag images that do not
    visually resemble a typical grayscale MRI scan.

    NOTE:
    This is only a basic UX filter.
    It is NOT a medical-grade MRI detector.
    """

    image_array = np.array(image).astype(np.float32)

    # Check RGB channel difference.
    # Normal MRI images are generally grayscale-like.
    r = image_array[:, :, 0]
    g = image_array[:, :, 1]
    b = image_array[:, :, 2]

    channel_difference = (
        np.mean(np.abs(r - g))
        + np.mean(np.abs(g - b))
        + np.mean(np.abs(r - b))
    ) / 3

    # Convert to grayscale
    gray = np.mean(image_array, axis=2)

    mean_value = float(np.mean(gray))
    std_value = float(np.std(gray))

    # Very bright / very dark area ratios
    bright_ratio = float(np.mean(gray > 245))
    dark_ratio = float(np.mean(gray < 10))

    # Reject strongly colored images
    if channel_difference > 35:
        return False

    # Reject very low-detail images
    if std_value < 18:
        return False

    # Reject completely bright / dark images
    if mean_value > 245 or mean_value < 5:
        return False

    # Reject mostly white / mostly black images
    if bright_ratio > 0.85 or dark_ratio > 0.85:
        return False

    return True


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize an image for model input."""
    resized = image.resize(IMAGE_SIZE)
    array = np.array(resized) / 255.0
    return np.expand_dims(array, axis=0)


def predict_image(model, image: Image.Image):
    """
    Run inference and return (predicted_class, confidence, probabilities).
    Raises a RuntimeError with a friendly message if inference fails,
    so the caller can show it instead of crashing the app.
    """
    try:
        processed = preprocess_image(image)
        predictions = model.predict(processed, verbose=0)[0]
    except Exception as error:
        raise RuntimeError(f"Model inference failed: {error}") from error

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index] * 100)
    probabilities = {
        name: float(predictions[i] * 100)
        for i, name in enumerate(CLASS_NAMES)
    }

    return predicted_class, confidence, probabilities


def get_confidence_band(confidence: float):
    """Map a confidence percentage to a (label, color) band for display."""
    for threshold, label, color in CONFIDENCE_BANDS:
        if confidence >= threshold:
            return label, color
    return CONFIDENCE_BANDS[-1][1], CONFIDENCE_BANDS[-1][2]


def sorted_probabilities(probabilities: dict) -> list:
    """Return (class_name, probability) pairs sorted highest first."""
    return sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True
    )


def push_to_history(result: dict) -> None:
    """Keep the most recent predictions in session state for quick reference."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    st.session_state.prediction_history.insert(
        0,
        {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "File": result["file_name"],
            "Prediction": result["predicted_class"],
            "Confidence": f"{result['confidence']:.2f}%",
        },
    )

    st.session_state.prediction_history = (
        st.session_state.prediction_history[:MAX_HISTORY_ITEMS]
    )


def build_report(
    file_name: str,
    predicted_class: str,
    confidence: float,
    probabilities: dict
) -> str:

    """Build a plain-text summary report for download."""
    lines = [
        "NEUROVISION AI",
        "BRAIN MRI CLASSIFICATION REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
        f"Image File: {file_name}",
        "Model Architecture: VGG19",
        "Model Input Size: 224 x 224 x 3",
        f"Test Accuracy: {TEST_ACCURACY:.2f}%",
        "",
        "PREDICTION RESULT",
        "-" * 30,
        f"Predicted Class: {predicted_class}",
        f"Confidence: {confidence:.2f}%",
        "",
        "CLASS PROBABILITIES",
        "-" * 30,
    ]

    lines += [
        f"{name}: {value:.2f}%"
        for name, value in probabilities.items()
    ]

    lines += [
        "",
        "IMPORTANT NOTICE",
        "This application is intended for educational and research purposes only.",
        "It is not a medical diagnostic system.",
        "Clinical interpretation should be performed by qualified medical professionals.",
    ]

    return "\n".join(lines)


# ============================================================
# UI SECTIONS
# ============================================================

def section_title(text: str) -> None:
    st.markdown(
        f'<div class="section-title">{text}</div>',
        unsafe_allow_html=True
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div class="sidebar-icon">🧠</div>
                <div class="sidebar-title">{APP_TITLE}</div>
                <div class="sidebar-subtitle">Brain MRI Classification System</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown(
            '<div class="sidebar-section">NAVIGATION</div>',
            unsafe_allow_html=True
        )

        page = st.radio(
            "Navigation",
            [
                "🏠 Dashboard",
                "🔬 MRI Classifier",
                "🤖 Model Information",
                "💬 Feedback"
            ],
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown(
            '<div class="sidebar-section">MODEL DETAILS</div>',
            unsafe_allow_html=True
        )

        st.write("**Architecture:** VGG19")
        st.write(f"**Classes:** {len(CLASS_NAMES)}")
        st.write(f"**Input Size:** {IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}")
        st.write(f"**Test Accuracy:** {TEST_ACCURACY:.2f}%")

        st.divider()

        st.markdown(
            '<div class="sidebar-section">SUPPORTED CLASSES</div>',
            unsafe_allow_html=True
        )

        for name in CLASS_NAMES:
            st.write(f"{CLASS_ICONS[name]} **{name}**")

        st.divider()

        st.markdown(
            '<div class="sidebar-section">DEVELOPER</div>',
            unsafe_allow_html=True
        )

        st.write(f"👨‍💻 **{DEVELOPER_NAME}**")
        st.caption("AI / Machine Learning Project")

    return page


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero-box">
            <div class="hero-icon">🧠</div>
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-subtitle">{APP_TAGLINE}</div>
            <div class="developer-badge">👨‍💻 Developed by {DEVELOPER_NAME}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        f"""
        <div class="footer-box">
            <div class="footer-title">🧠 {APP_TITLE}</div>
            <div class="footer-main">Brain MRI Image Classification System</div>
            <div class="footer-developer">👨‍💻 Developed by {DEVELOPER_NAME}</div>
            <div class="footer-main">AI / Machine Learning Project</div>
            <div class="footer-line"></div>
            <div class="footer-warning">
                ⚠️ <b>Important Medical Disclaimer</b><br><br>
                {APP_TITLE} is an educational and research prototype developed for
                demonstrating deep learning-based brain MRI image classification.<br>
                This application is <b>not a medical diagnostic system</b> and its
                predictions should not be used to make medical decisions.<br>
                Always consult a qualified healthcare professional for medical
                evaluation and clinical interpretation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_dashboard() -> None:
    section_title("🏠 Project Dashboard")

    st.info(
        f"""
        **Welcome to {APP_TITLE}**

        This application uses a trained VGG19 deep learning model to classify
        brain MRI scan images into four predefined categories: Glioma,
        Meningioma, No Tumor, and Pituitary.
        """
    )

    section_title("📊 Key Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Classification Classes", len(CLASS_NAMES))
    c2.metric("Total MRI Images", f"{DATASET_STATS['Total Images']:,}")
    c3.metric("Model Architecture", "VGG19")
    c4.metric("Test Accuracy", f"{TEST_ACCURACY:.2f}%")

    section_title("📁 Dataset Statistics")

    d1, d2, d3 = st.columns(3)

    d1.metric("Training Images", f"{DATASET_STATS['Training Images']:,}")
    d2.metric("Validation Images", f"{DATASET_STATS['Validation Images']:,}")
    d3.metric("Testing Images", f"{DATASET_STATS['Testing Images']:,}")

    section_title("🔬 About the Project")

    a1, a2 = st.columns(2)

    with a1:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">🎯 Project Objective</div>
                <div class="card-text">
                    NeuroVision AI demonstrates a deep learning-based system for
                    classifying brain MRI scan images. The project uses transfer
                    learning and fine-tuning with VGG19 to identify four
                    predefined image categories.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">🛠️ Technology Stack</div>
                <div class="card-text">
                    • Python<br>• TensorFlow / Keras<br>• VGG19<br>
                    • Streamlit<br>• NumPy<br>• Pandas<br>• Pillow
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_title("⚙️ System Workflow")

    workflow = [
        ("01", "📤", "Upload", "Upload MRI scan"),
        ("02", "🔍", "Validate", "Check image quality"),
        ("03", "📐", "Preprocess", "Resize and normalize"),
        ("04", "🤖", "Predict", "VGG19 inference"),
        ("05", "📊", "Result", "Probability scores"),
    ]

    columns = st.columns(5)

    for column, (number, icon, title, description) in zip(columns, workflow):
        with column:
            st.markdown(
                f"""
                <div class="workflow-card">
                    <div class="workflow-number">{number}</div>
                    <div class="workflow-icon">{icon}</div>
                    <div class="workflow-title">{title}</div>
                    <div class="workflow-text">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_title("🧠 Supported MRI Categories")

    categories = [
        ("Glioma", "A brain tumor category associated with glial cells."),
        ("No Tumor", "MRI images belonging to the no-tumor class."),
        ("Meningioma", "A tumor category associated with the meninges surrounding the brain."),
        ("Pituitary", "A tumor category associated with the pituitary region."),
    ]

    c1, c2 = st.columns(2)

    for i, (name, description) in enumerate(categories):
        target = c1 if i % 2 == 0 else c2

        with target:
            st.markdown(
                f"""
                <div class="category-card">
                    <div class="category-title">{CLASS_ICONS[name]} {name}</div>
                    <div class="category-description">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_uploaded_image_info(uploaded_file, image: Image.Image) -> None:
    section_title("🖼️ Uploaded MRI Image")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.image(
            image,
            caption="Brain MRI Scan",
            use_container_width=True
        )

    with col2:
        st.markdown(
            '<div class="card"><div class="card-title">📋 Image Information</div></div>',
            unsafe_allow_html=True,
        )

        st.write(f"**File Name:** {uploaded_file.name}")
        st.write(f"**Original Dimensions:** {image.size[0]} × {image.size[1]}")
        st.write("**Color Format:** RGB")
        st.write(f"**Model Input:** {IMAGE_SIZE[0]} × {IMAGE_SIZE[1]} × 3")
        st.write("**Normalization:** Pixel values / 255")

    st.divider()


def render_prediction_result(result: dict) -> None:
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]

    section_title("🎯 Classification Result")

    if predicted_class == "No Tumor":
        st.success("The model predicted the No Tumor class.")
    else:
        st.warning(f"The model predicted the {predicted_class} class.")

    icon = "✅" if predicted_class == "No Tumor" else "🧠"
    confidence_label, confidence_color = get_confidence_band(confidence)

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-heading">MODEL PREDICTION</div>
            <div class="result-class">{icon} {predicted_class}</div>
            <div class="result-confidence">Confidence: {confidence:.2f}%</div>
            <div style="margin-top:10px;">
                <span style="background:{confidence_color}; color:white; padding:6px 16px;
                border-radius:50px; font-weight:800; font-size:13px;">
                    {confidence_label}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if confidence_label == "Low Confidence":
        st.info(
            "ℹ️ The model's confidence for this image is relatively low. "
            "Consider uploading a clearer scan or treating this result with caution."
        )

    section_title("📊 Class Probability Scores")

    ranked = sorted_probabilities(probabilities)

    p1, p2 = st.columns(2)

    for i, (class_name, probability) in enumerate(ranked):
        target = p1 if i % 2 == 0 else p2

        with target:
            st.write(f"{CLASS_ICONS[class_name]} **{class_name}**")
            st.progress(min(probability / 100, 1.0))
            st.caption(f"Probability: {probability:.2f}%")

    with st.expander("📈 View as bar chart"):
        chart_df = pd.DataFrame(
            {
                "Class": [name for name, _ in ranked],
                "Probability (%)": [val for _, val in ranked]
            }
        ).set_index("Class")

        st.bar_chart(chart_df)

    section_title("📋 Analysis Summary")

    s1, s2, s3 = st.columns(3)

    s1.metric("Predicted Class", predicted_class)
    s2.metric("Confidence", f"{confidence:.2f}%")
    s3.metric("Model", "VGG19")

    report = build_report(
        result["file_name"],
        predicted_class,
        confidence,
        probabilities
    )

    st.download_button(
        "📄 Download Prediction Report",
        data=report,
        file_name="neurovision_prediction_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="notice">
            ⚠️ <b>Important:</b> The prediction and confidence score are generated
            by a machine learning model and should not be interpreted as a
            medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    if st.button("🔄 Start New Analysis", use_container_width=True):
        st.session_state.prediction_done = False
        st.session_state.prediction_data = None
        st.rerun()


def page_classifier(model) -> None:
    section_title("🔬 Brain MRI Classifier")

    st.info(
        "**Upload a clear brain MRI scan image** in JPG, JPEG, or PNG format "
        "and click **Analyze MRI Scan** to generate a model prediction."
    )

    uploaded_file = st.file_uploader(
        "Upload Brain MRI Scan",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file).convert("RGB")

    render_uploaded_image_info(uploaded_file, image)

    if not check_image_quality(image):
        st.error("❌ The uploaded image is not suitable for analysis.")

        st.warning(
            "Please upload a clear brain MRI scan image.\n\n"
            "The current validation is a basic image-quality screening step "
            "and is not a dedicated MRI-versus-non-MRI classifier."
        )

        return

    # ========================================================
    # ADDED CHECK FOR NON-MRI IMAGE
    # ========================================================

    if not check_mri_like_image(image):
        st.warning("⚠️ This image may not be a brain MRI scan.")

        st.info(
            "Please upload a valid brain MRI image. "
            "The uploaded image does not strongly resemble "
            "a typical grayscale MRI scan."
        )

        st.session_state.prediction_done = False
        st.session_state.prediction_data = None

        return

    analyze = st.button(
        "🔍 Analyze MRI Scan",
        type="primary",
        use_container_width=True
    )

    if analyze:
        with st.spinner("Analyzing the MRI scan..."):

            try:
                predicted_class, confidence, probabilities = predict_image(
                    model,
                    image
                )

            except RuntimeError as error:
                st.error(f"❌ {error}")

                st.session_state.prediction_done = False
                st.session_state.prediction_data = None

                return

        st.session_state.prediction_done = True

        st.session_state.prediction_data = {
            "file_name": uploaded_file.name,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
        }

        push_to_history(st.session_state.prediction_data)

    if st.session_state.prediction_done:
        render_prediction_result(
            st.session_state.prediction_data
        )

    history = st.session_state.get(
        "prediction_history",
        []
    )

    if history:
        with st.expander(
            f"🕒 Recent Predictions ({len(history)})"
        ):
            st.dataframe(
                pd.DataFrame(history),
                use_container_width=True,
                hide_index=True
            )


def page_model_info() -> None:
    section_title("🤖 Model Information")

    st.info(
        "**NeuroVision AI uses VGG19 transfer learning and fine-tuning for "
        "four-class brain MRI image classification.**"
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Architecture", "VGG19")
    m2.metric("Input Size", f"{IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}")
    m3.metric("Classes", len(CLASS_NAMES))
    m4.metric("Test Accuracy", f"{TEST_ACCURACY:.2f}%")

    section_title("⚙️ Model Processing Pipeline")

    pipeline = [
        "1️⃣ Input Brain MRI Image",
        "2️⃣ Resize Image to 224 × 224",
        "3️⃣ Normalize Pixel Values",
        "4️⃣ VGG19 Feature Extraction",
        "5️⃣ Fine-Tuned Convolutional Layers",
        "6️⃣ Dense Classification Layer",
        "7️⃣ Softmax Probability Output",
    ]

    for step in pipeline:
        st.info(f"**{step}**")

    section_title("📚 Dataset Information")

    t1, t2, t3, t4 = st.columns(4)

    t1.metric(
        "Total Images",
        f"{DATASET_STATS['Total Images']:,}"
    )

    t2.metric(
        "Training",
        f"{DATASET_STATS['Training Images']:,}"
    )

    t3.metric(
        "Validation",
        f"{DATASET_STATS['Validation Images']:,}"
    )

    t4.metric(
        "Testing",
        f"{DATASET_STATS['Testing Images']:,}"
    )

    section_title("⚠️ Model Limitations")

    st.warning(
        "The model was trained on a specific brain MRI dataset containing four "
        "predefined classes.\n\n"
        "Performance on MRI images from different scanners, hospitals, "
        "populations, imaging protocols, or datasets may differ from the "
        "reported test performance.\n\n"
        "The system is an educational and research prototype and is not "
        "intended for clinical diagnosis."
    )


def save_feedback(
    name: str,
    email: str,
    category: str,
    rating: int,
    feedback: str
) -> None:

    new_row = pd.DataFrame(
        [{
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Name": name,
            "Email": email,
            "Category": category,
            "Rating": rating,
            "Feedback": feedback,
        }]
    )

    if os.path.exists(FEEDBACK_FILE):
        existing = pd.read_csv(FEEDBACK_FILE)
        new_row = pd.concat(
            [existing, new_row],
            ignore_index=True
        )

    new_row.to_csv(
        FEEDBACK_FILE,
        index=False
    )


def page_feedback() -> None:
    section_title("💬 Feedback Center")

    st.info(
        f"**Your feedback helps improve {APP_TITLE}.**\n\n"
        "You can report bugs, describe prediction issues, suggest features, "
        "or share your experience with the application."
    )

    with st.form("feedback_form", clear_on_submit=True):

        st.subheader("📝 Submit Your Feedback")

        name = st.text_input(
            "Full Name",
            placeholder="Enter your full name"
        )

        email = st.text_input(
            "Email Address",
            placeholder="Enter your email address"
        )

        feedback_type = st.selectbox(
            "Feedback Category",
            [
                "General Feedback",
                "User Interface",
                "Prediction Issue",
                "Image Upload Issue",
                "Bug Report",
                "Feature Request",
                "Performance",
                "Other",
            ],
        )

        rating = st.slider(
            "Overall Experience Rating",
            min_value=1,
            max_value=5,
            value=5
        )

        feedback = st.text_area(
            "Your Feedback",
            placeholder="Describe your experience, problem, or suggestion...",
            height=170,
        )

        submitted = st.form_submit_button(
            "📨 Submit Feedback",
            use_container_width=True
        )

        if submitted:

            if feedback.strip() == "":
                st.error(
                    "Please enter your feedback before submitting."
                )

            else:
                save_feedback(
                    name,
                    email,
                    feedback_type,
                    rating,
                    feedback
                )

                st.success(
                    "Thank you! Your feedback has been submitted successfully."
                )

    section_title("💡 What Can You Report?")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">🎨 User Interface</div>
                <div class="card-text">Share suggestions about the design, colors,
                navigation, layout, accessibility, and usability.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f2:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">🤖 Prediction Experience</div>
                <div class="card-text">Report issues related to image uploading,
                prediction results, confidence scores, or analysis.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f3:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">🚀 Feature Requests</div>
                <div class="card-text">Suggest new features, reports,
                visualizations, tools, or improvements.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# MAIN APP


def main() -> None:
    inject_custom_css()

    if "prediction_done" not in st.session_state:
        st.session_state.prediction_done = False

    if "prediction_data" not in st.session_state:
        st.session_state.prediction_data = None

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    page = render_sidebar()

    render_hero()

    if page == "🏠 Dashboard":
        page_dashboard()

    elif page == "🔬 MRI Classifier":
        model = get_model()
        page_classifier(model)

    elif page == "🤖 Model Information":
        page_model_info()

    elif page == "💬 Feedback":
        page_feedback()

    render_footer()


if __name__ == "__main__":
    main()