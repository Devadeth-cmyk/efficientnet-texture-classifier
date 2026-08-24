import streamlit as st
from PIL import Image
import pandas as pd

from src.preprocessing import preprocess_image
from src.predictor import load_model, predict


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TextureAI | Surface Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

CUSTOM_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --background: #f5f7fb;
    --card: #ffffff;
    --primary: #4f46e5;
    --primary-dark: #3730a3;
    --secondary: #7c3aed;
    --text: #111827;
    --muted: #6b7280;
    --border: #e5e7eb;
    --success: #059669;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(99, 102, 241, 0.08),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(124, 58, 237, 0.07),
            transparent 30%
        ),
        var(--background);
}


/* ============================================================
   HEADER
   ============================================================ */

.hero {
    padding: 2rem 2.2rem;
    border-radius: 24px;
    margin-bottom: 1.8rem;

    background:
        linear-gradient(
            135deg,
            #312e81 0%,
            #4f46e5 50%,
            #7c3aed 100%
        );

    color: white;

    box-shadow:
        0 15px 40px rgba(79, 70, 229, 0.22);
}

.hero-badge {
    display: inline-block;

    padding: 0.35rem 0.8rem;

    border-radius: 999px;

    background: rgba(255,255,255,0.15);

    border: 1px solid rgba(255,255,255,0.25);

    font-size: 0.75rem;

    font-weight: 600;

    letter-spacing: 0.08em;

    text-transform: uppercase;

    margin-bottom: 0.8rem;
}

.hero-title {
    font-size: 2.6rem;

    font-weight: 800;

    margin: 0;

    line-height: 1.1;
}

.hero-subtitle {
    margin-top: 0.7rem;

    font-size: 1rem;

    color: rgba(255,255,255,0.82);

    max-width: 750px;
}


/* ============================================================
   SECTION HEADERS
   ============================================================ */

.section-title {
    font-size: 1.1rem;

    font-weight: 700;

    color: var(--text);

    margin-top: 1.5rem;

    margin-bottom: 0.8rem;
}

.section-caption {
    font-size: 0.85rem;

    color: var(--muted);

    margin-bottom: 1rem;
}


/* ============================================================
   STAT CARDS
   ============================================================ */

.stat-card {
    background: var(--card);

    border: 1px solid var(--border);

    border-radius: 16px;

    padding: 1rem 1.2rem;

    box-shadow:
        0 5px 18px rgba(17, 24, 39, 0.05);
}

.stat-label {
    font-size: 0.72rem;

    text-transform: uppercase;

    letter-spacing: 0.08em;

    color: var(--muted);

    font-weight: 600;
}

.stat-value {
    font-size: 1.25rem;

    font-weight: 700;

    color: var(--text);

    margin-top: 0.25rem;
}


/* ============================================================
   UPLOAD AREA
   ============================================================ */

[data-testid="stFileUploaderDropzone"] {

    background: rgba(255,255,255,0.85);

    border: 2px dashed #a5b4fc;

    border-radius: 18px;

    padding: 1rem;

    transition: all 0.2s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {

    border-color: var(--primary);

    background: #f8f8ff;
}


/* ============================================================
   CARDS
   ============================================================ */

.card {

    background: var(--card);

    border: 1px solid var(--border);

    border-radius: 20px;

    padding: 1.5rem;

    box-shadow:
        0 8px 25px rgba(17, 24, 39, 0.06);

}


/* ============================================================
   PREDICTION CARD
   ============================================================ */

.prediction-card {

    background:
        linear-gradient(
            145deg,
            #eef2ff,
            #ffffff
        );

    border: 1px solid #c7d2fe;

    border-radius: 20px;

    padding: 1.7rem;

    box-shadow:
        0 10px 30px rgba(79, 70, 229, 0.10);
}

.prediction-label {

    font-size: 0.72rem;

    text-transform: uppercase;

    letter-spacing: 0.12em;

    color: var(--primary);

    font-weight: 700;
}

.prediction-class {

    font-size: 2.1rem;

    font-weight: 800;

    color: var(--primary-dark);

    margin-top: 0.4rem;

    text-transform: capitalize;
}

.confidence-number {

    font-size: 2.8rem;

    font-weight: 800;

    color: var(--text);

    margin-top: 1rem;
}

.confidence-label {

    color: var(--muted);

    font-size: 0.8rem;

    text-transform: uppercase;

    letter-spacing: 0.08em;
}


/* ============================================================
   CONFIDENCE BAR
   ============================================================ */

.confidence-track {

    width: 100%;

    height: 10px;

    background: #e5e7eb;

    border-radius: 999px;

    overflow: hidden;

    margin-top: 0.8rem;
}

.confidence-fill {

    height: 100%;

    border-radius: 999px;

    background:
        linear-gradient(
            90deg,
            #4f46e5,
            #7c3aed
        );
}


/* ============================================================
   TOP PREDICTION ROW
   ============================================================ */

.prediction-row {

    display: flex;

    align-items: center;

    gap: 0.8rem;

    padding: 0.7rem 0;

    border-bottom: 1px solid var(--border);
}

.prediction-rank {

    width: 28px;

    height: 28px;

    border-radius: 50%;

    display: flex;

    align-items: center;

    justify-content: center;

    background: #eef2ff;

    color: var(--primary);

    font-size: 0.75rem;

    font-weight: 700;
}

.prediction-name {

    width: 130px;

    font-size: 0.9rem;

    font-weight: 600;

    color: var(--text);

    text-transform: capitalize;
}

.prediction-bar {

    flex: 1;

    height: 8px;

    background: #eef0f4;

    border-radius: 999px;

    overflow: hidden;
}

.prediction-fill {

    height: 100%;

    border-radius: 999px;

    background:
        linear-gradient(
            90deg,
            #6366f1,
            #8b5cf6
        );
}

.prediction-percent {

    width: 65px;

    text-align: right;

    font-size: 0.8rem;

    font-weight: 600;

    color: var(--muted);
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {

    background: #111827;
}

section[data-testid="stSidebar"] * {

    color: #f9fafb;
}

.sidebar-title {

    font-size: 1.3rem;

    font-weight: 800;

    margin-bottom: 1.2rem;
}

.sidebar-item {

    padding: 0.8rem;

    border-radius: 10px;

    background: rgba(255,255,255,0.06);

    margin-bottom: 0.5rem;
}

.sidebar-label {

    font-size: 0.7rem;

    color: #9ca3af !important;

    text-transform: uppercase;

    letter-spacing: 0.08em;
}

.sidebar-value {

    font-size: 0.9rem;

    font-weight: 600;
}


/* ============================================================
   BUTTON
   ============================================================ */

.stButton > button {

    border-radius: 10px;

    border: none;

    font-weight: 600;

    background: var(--primary);

    color: white;

    padding: 0.6rem 1.2rem;
}

.stButton > button:hover {

    background: var(--primary-dark);

    color: white;
}


/* ============================================================
   HIDE STREAMLIT DEFAULT ELEMENTS
   ============================================================ */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
"""

st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            AI-Powered Computer Vision
        </div>

        <div class="hero-title">
            🔬 TextureAI
        </div>

        <div class="hero-subtitle">
            Analyze surface textures using a fine-tuned
            EfficientNetB3 deep learning model.
            Upload an image and receive an instant
            47-class classification with confidence scores.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():

    return load_model()


try:

    with st.spinner("Initializing EfficientNetB3..."):

        model = get_model()

except Exception as e:

    st.error(
        "Unable to load the trained model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">⚙️ Model Details</div>',
        unsafe_allow_html=True
    )

    specs = [
        ("Architecture", "EfficientNetB3"),
        ("Input Size", "300 × 300"),
        ("Classes", "47"),
        ("Framework", "TensorFlow"),
        ("Preprocessing", "OpenCV"),
    ]

    for label, value in specs:

        st.markdown(
            f"""
            <div class="sidebar-item">

                <div class="sidebar-label">
                    {label}
                </div>

                <div class="sidebar-value">
                    {value}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown(
        """
        <div style="
            color:#9ca3af;
            font-size:0.75rem;
            line-height:1.5;
        ">
        The trained model analyzes uploaded images
        and returns the most probable texture class
        together with the top alternative predictions.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL STATISTICS
# ============================================================

st.markdown(
    '<div class="section-title">Model Overview</div>',
    unsafe_allow_html=True
)

stat1, stat2, stat3, stat4 = st.columns(4)

stats = [
    ("Architecture", "EfficientNetB3"),
    ("Input", "300 × 300"),
    ("Classes", "47"),
    ("Inference", "Real-time"),
]

for column, (label, value) in zip(
    [stat1, stat2, stat3, stat4],
    stats
):

    with column:

        st.markdown(
            f"""
            <div class="stat-card">

                <div class="stat-label">
                    {label}
                </div>

                <div class="stat-value">
                    {value}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Upload Texture Image</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-caption">'
    'Supported formats: JPG, JPEG, PNG, BMP, WEBP'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ],
    label_visibility="collapsed"
)


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="card"
             style="
                 text-align:center;
                 padding:3rem;
                 margin-top:1rem;
             ">

            <div style="font-size:3rem;">
                🖼️
            </div>

            <h3>
                Upload an image to begin
            </h3>

            <p style="color:#6b7280;">
                Our EfficientNetB3 model will analyze
                the image and identify its texture class.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# IMAGE
# ============================================================

image = Image.open(
    uploaded_file
)

# Make sure the image is RGB
image = image.convert("RGB")


# ============================================================
# PREDICTION
# ============================================================

with st.spinner(
    "Analyzing texture..."
):

    try:

        image_bytes = (
            uploaded_file.getvalue()
        )

        processed_image = (
            preprocess_image(
                image_bytes
            )
        )

        result = predict(
            model,
            processed_image
        )

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)

        st.stop()


# ============================================================
# RESULT HEADER
# ============================================================

st.markdown(
    '<div class="section-title">Analysis Result</div>',
    unsafe_allow_html=True
)


# ============================================================
# IMAGE + MAIN RESULT
# ============================================================

image_col, result_col = st.columns(
    [1.05, 1],
    gap="large"
)


# ------------------------------------------------------------
# IMAGE
# ------------------------------------------------------------

with image_col:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.image(
        image,
        use_container_width=True
    )

    st.markdown(
        f"""
        <div style="
            color:#6b7280;
            font-size:0.75rem;
            margin-top:0.6rem;
        ">
            📄 {uploaded_file.name}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------

with result_col:

    confidence = (
        result["confidence"] * 100
    )

    if confidence >= 80:

        status = "High confidence"
        status_color = "#059669"

    elif confidence >= 50:

        status = "Moderate confidence"
        status_color = "#d97706"

    else:

        status = "Low confidence"
        status_color = "#dc2626"

    st.markdown(
        f"""
        <div class="prediction-card">

            <div class="prediction-label">
                Predicted Texture
            </div>

            <div class="prediction-class">
                {result["class"]}
            </div>

            <div class="confidence-number">
                {confidence:.2f}%
            </div>

            <div class="confidence-label">
                Confidence
            </div>

            <div class="confidence-track">

                <div class="confidence-fill"
                     style="width:{confidence:.2f}%;">
                </div>

            </div>

            <div style="
                margin-top:0.9rem;
                color:{status_color};
                font-weight:700;
                font-size:0.85rem;
            ">
                ● {status}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TOP 5
# ============================================================

st.markdown(
    '<div class="section-title">Top 5 Predictions</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-caption">'
    'Probability distribution across the most likely classes'
    '</div>',
    unsafe_allow_html=True
)


for rank, item in enumerate(
    result["top_predictions"],
    start=1
):

    class_name = item["class"]

    probability = (
        item["confidence"] * 100
    )

    st.markdown(
        f"""
        <div class="prediction-row">

            <div class="prediction-rank">
                {rank}
            </div>

            <div class="prediction-name">
                {class_name}
            </div>

            <div class="prediction-bar">

                <div class="prediction-fill"
                     style="width:{probability:.2f}%;">
                </div>

            </div>

            <div class="prediction-percent">
                {probability:.2f}%
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# INTERACTIVE CHART
# ============================================================

st.markdown(
    '<div class="section-title">Prediction Distribution</div>',
    unsafe_allow_html=True
)

chart_data = pd.DataFrame(
    {
        "Texture": [
            item["class"]
            for item in result["top_predictions"]
        ],

        "Confidence": [
            item["confidence"] * 100
            for item in result["top_predictions"]
        ],
    }
)

st.bar_chart(
    chart_data.set_index("Texture"),
    y="Confidence",
    height=280
)


# ============================================================
# IMAGE INFORMATION
# ============================================================

with st.expander(
    "🔎 Image Information"
):

    info1, info2, info3 = st.columns(3)

    info1.metric(
        "Width",
        f"{image.width}px"
    )

    info2.metric(
        "Height",
        f"{image.height}px"
    )

    info3.metric(
        "Format",
        image.format or "Image"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#6b7280;
        font-size:0.75rem;
        padding:0.8rem;
    ">
        TextureAI · EfficientNetB3 · 47-Class Texture Recognition
    </div>
    """,
    unsafe_allow_html=True
)