import streamlit as st
from PIL import Image

from src.preprocessing import preprocess_image
from src.predictor import load_model, predict


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Surface & Texture Analysis",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# DESIGN SYSTEM (CSS)
# --------------------------------------------------
# Concept: a materials-lab specimen card. Paper-stone canvas,
# a pine/ink primary for structure, a copper accent reserved for
# the one thing that matters — the reading. Serif for identity,
# mono for every number, because numbers are data, not decoration.

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #EDEEE8;
    --surface: #FFFFFF;
    --ink: #1B2321;
    --ink-soft: #5B655F;
    --accent: #234C4A;
    --accent-copper: #B5622A;
    --line: #D9DBD1;
    --success-tint: #E8EFE6;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--bg);
}

/* ---------- Header / title block ---------- */

.spec-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--accent-copper);
    margin-bottom: 0.35rem;
}

.spec-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.4rem;
    line-height: 1.1;
    color: var(--ink);
    margin: 0 0 0.4rem 0;
}

.spec-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.98rem;
    color: var(--ink-soft);
    margin-bottom: 0.6rem;
}

.spec-rule {
    border: none;
    border-top: 1px solid var(--line);
    margin: 1.1rem 0 1.6rem 0;
}

/* ---------- Section labels ---------- */

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-soft);
    border-bottom: 1px solid var(--line);
    padding-bottom: 0.4rem;
    margin: 0.4rem 0 1rem 0;
}

/* ---------- Sidebar spec sheet ---------- */

section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--line);
}

.sidebar-block {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}

.spec-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px dashed var(--line);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}

.spec-row .k {
    color: var(--ink-soft);
}

.spec-row .v {
    color: var(--ink);
    font-weight: 500;
    text-align: right;
}

/* ---------- Upload / drop zone ---------- */

[data-testid="stFileUploaderDropzone"] {
    background: var(--surface);
    border: 1.5px dashed var(--ink-soft);
    border-radius: 6px;
}

/* ---------- Result card ---------- */

.result-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1.6rem 1.8rem;
}

.result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin-bottom: 0.5rem;
}

.result-class {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.9rem;
    color: var(--accent);
    margin: 0 0 1.1rem 0;
    text-transform: capitalize;
}

.reading {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
}

.reading-value {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 2.6rem;
    color: var(--ink);
}

.reading-unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    color: var(--ink-soft);
}

.reading-caption {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin-top: 0.3rem;
}

/* ---------- Ranked readout rows (top-5) ---------- */

.rank-row {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 0.55rem 0;
}

.rank-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--ink-soft);
    width: 1.4rem;
    flex-shrink: 0;
}

.rank-name {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.92rem;
    width: 11rem;
    flex-shrink: 0;
    text-transform: capitalize;
    color: var(--ink);
}

.rank-track {
    flex-grow: 1;
    height: 8px;
    background: var(--bg);
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid var(--line);
}

.rank-fill {
    height: 100%;
    background: var(--accent-copper);
    border-radius: 4px 0 0 4px;
}

.rank-fill.top {
    background: var(--accent);
}

.rank-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: var(--ink-soft);
    width: 3.4rem;
    text-align: right;
    flex-shrink: 0;
}

/* ---------- Misc ---------- */

.footnote {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--ink-soft);
    letter-spacing: 0.04em;
}

#MainMenu, footer {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="spec-eyebrow">Surface &amp; Texture Analysis · v1.0</div>
    <div class="spec-title">🧩 Texture Classification</div>
    <div class="spec-subtitle">
        Upload a surface sample to identify its texture class and view the model's confidence readout.
    </div>
    <hr class="spec-rule" />
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_model()


try:
    with st.spinner("Loading model…"):
        model = get_model()

except Exception as e:
    st.error("Failed to load the trained model.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# SIDEBAR — SPEC SHEET
# --------------------------------------------------

with st.sidebar:

    st.markdown('<div class="spec-eyebrow">Model Spec Sheet</div>', unsafe_allow_html=True)
    st.markdown('<div class="spec-title" style="font-size:1.3rem;">Configuration</div>', unsafe_allow_html=True)
    st.markdown('<hr class="spec-rule" style="margin:0.8rem 0 1.2rem 0;" />', unsafe_allow_html=True)

    specs = [
        ("Architecture", "EfficientNetB3"),
        ("Input size", "300 × 300"),
        ("Classes", "47"),
        ("Framework", "TensorFlow / Keras"),
        ("Preprocessing", "OpenCV"),
    ]

    rows_html = "".join(
        f'<div class="spec-row"><span class="k">{k}</span><span class="v">{v}</span></div>'
        for k, v in specs
    )
    st.markdown(f'<div class="sidebar-block">{rows_html}</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="footnote">Model loaded and cached for this session.</div>', unsafe_allow_html=True)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.markdown('<div class="section-label">01 · Sample Upload</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="footnote">
            Accepted formats: JPG, JPEG, PNG, BMP, WEBP. The image is analyzed locally within this session only.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 · Specimen &amp; Reading</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    # ---------------- DISPLAY IMAGE ----------------

    with col1:
        st.image(image, use_container_width=True)
        st.markdown(
            f'<div class="footnote">FILE · {uploaded_file.name}</div>',
            unsafe_allow_html=True,
        )

    # ---------------- RUN PREDICTION ----------------

    with st.spinner("Analyzing texture…"):

        try:
            image_bytes = uploaded_file.getvalue()
            processed_image = preprocess_image(image_bytes)
            result = predict(model, processed_image)

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)
            st.stop()

    # ---------------- MAIN RESULT ----------------

    with col2:
        confidence_pct = result["confidence"] * 100

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted Class</div>
                <div class="result-class">{result["class"]}</div>
                <div class="reading">
                    <span class="reading-value">{confidence_pct:.2f}</span>
                    <span class="reading-unit">%</span>
                </div>
                <div class="reading-caption">Confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------- TOP 5 ----------------

    st.markdown('<div style="height:1.8rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 · Ranked Readout — Top 5</div>', unsafe_allow_html=True)

    rows = []
    for i, item in enumerate(result["top_predictions"], start=1):
        class_name = item["class"]
        confidence = item["confidence"] * 100
        fill_class = "rank-fill top" if i == 1 else "rank-fill"

        rows.append(
            f'<div class="rank-row">'
            f'<span class="rank-badge">{i:02d}</span>'
            f'<span class="rank-name">{class_name}</span>'
            f'<div class="rank-track">'
            f'<div class="{fill_class}" style="width:{confidence:.2f}%;"></div>'
            f'</div>'
            f'<span class="rank-pct">{confidence:.2f}%</span>'
            f'</div>'
        )

    st.markdown(
        f'<div class="result-card">{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )