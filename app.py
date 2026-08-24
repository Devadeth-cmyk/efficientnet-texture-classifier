import textwrap

import plotly.graph_objects as go
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


def render_html(html: str) -> None:
    """st.markdown wrapper that dedents multi-line HTML first.

    Streamlit's markdown renderer treats 4+ leading spaces as a code
    block, so an indented triple-quoted HTML string gets shown as
    literal text instead of being rendered. Dedenting avoids that.
    """
    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)


# --------------------------------------------------
# DESIGN SYSTEM (CSS)
# --------------------------------------------------
# Concept: a materials lab, but the vivid version — swatch cards under
# gallery lighting. Violet-ink as the structural colour, gold for the
# hero reading, coral/mint as a live confidence signal (low -> high).

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #F4F1FA;
    --surface: #FFFFFF;
    --ink: #221933;
    --ink-soft: #6B6280;
    --primary: #5B4B9E;
    --primary-dark: #3F3270;
    --gold: #E3A857;
    --coral: #E4634F;
    --mint: #3FA796;
    --line: #E1DCEF;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, #EFE9FB 0%, transparent 45%),
        radial-gradient(circle at 100% 20%, #FBF1E3 0%, transparent 40%),
        var(--bg);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---------- Hero header ---------- */

.hero {
    background: linear-gradient(120deg, var(--primary) 0%, var(--primary-dark) 100%);
    border-radius: 18px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.8rem;
    color: #FFFFFF;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.5s ease-out;
}

.hero::after {
    content: "";
    position: absolute;
    top: -40%;
    right: -10%;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(227,168,87,0.35) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.5rem;
}

.hero-title {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.5rem;
    line-height: 1.1;
    margin: 0 0 0.5rem 0;
}

.hero-subtitle {
    font-size: 0.98rem;
    color: #E4DEF7;
    max-width: 40rem;
}

/* ---------- Section labels ---------- */

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--primary);
    border-bottom: 2px solid var(--line);
    padding-bottom: 0.4rem;
    margin: 0.4rem 0 1rem 0;
}

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--line);
}

.sidebar-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--gold);
}

.sidebar-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.35rem;
    color: var(--ink);
    margin: 0.2rem 0 1rem 0;
}

.spec-row {
    display: flex;
    justify-content: space-between;
    padding: 0.55rem 0;
    border-bottom: 1px dashed var(--line);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}

.spec-row .k { color: var(--ink-soft); }
.spec-row .v { color: var(--primary); font-weight: 600; text-align: right; }

/* ---------- Upload / drop zone ---------- */

[data-testid="stFileUploaderDropzone"] {
    background: var(--surface);
    border: 1.5px dashed var(--primary);
    border-radius: 10px;
}

/* ---------- Cards ---------- */

.card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 1px 2px rgba(34, 25, 51, 0.04);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    animation: fadeInUp 0.45s ease-out;
}

.card:hover {
    box-shadow: 0 12px 28px rgba(91, 75, 158, 0.14);
    transform: translateY(-2px);
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
    font-weight: 700;
    font-size: 2rem;
    color: var(--primary-dark);
    margin: 0 0 0.9rem 0;
    text-transform: capitalize;
}

/* ---------- Confidence badge ---------- */

.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 0.32rem 0.75rem;
    border-radius: 999px;
    margin-bottom: 0.3rem;
}

.badge-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

.badge.high { background: rgba(63, 167, 150, 0.12); color: var(--mint); }
.badge.high .badge-dot { background: var(--mint); }

.badge.medium { background: rgba(227, 168, 87, 0.16); color: #B67A2C; }
.badge.medium .badge-dot { background: var(--gold); }

.badge.low { background: rgba(228, 99, 79, 0.12); color: var(--coral); }
.badge.low .badge-dot { background: var(--coral); }

/* ---------- Ranked readout rows (top-5) ---------- */

.rank-row {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 0.6rem 0;
    animation: fadeInUp 0.4s ease-out;
}

.rank-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: #FFFFFF;
    background: var(--primary);
    width: 1.6rem;
    height: 1.6rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.rank-row.top .rank-badge { background: var(--gold); color: var(--ink); }

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
    height: 10px;
    background: var(--bg);
    border-radius: 5px;
    overflow: hidden;
    border: 1px solid var(--line);
}

.rank-fill {
    height: 100%;
    border-radius: 5px 0 0 5px;
    background: linear-gradient(90deg, var(--primary) 0%, #8776CE 100%);
    transition: width 0.6s ease-out;
}

.rank-row.top .rank-fill {
    background: linear-gradient(90deg, var(--gold) 0%, #F0C687 100%);
}

.rank-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: var(--ink-soft);
    width: 3.6rem;
    text-align: right;
    flex-shrink: 0;
}

/* ---------- Tabs ---------- */

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
    background: var(--surface);
    color: var(--primary) !important;
}

/* ---------- Misc ---------- */

.footnote {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--ink-soft);
    letter-spacing: 0.04em;
}

.stButton > button {
    background: var(--primary);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: background 0.2s ease;
}
.stButton > button:hover {
    background: var(--primary-dark);
    color: #FFFFFF;
}

#MainMenu, footer {visibility: hidden;}

/* ---------- Contrast fixes for native Streamlit widgets ---------- */
/* Streamlit's own dark theme leaks light-grey text into these widgets
   on our light background — force readable ink/soft-ink everywhere. */

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: var(--ink) !important;
}

[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"] {
    color: var(--ink-soft) !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--ink-soft) !important;
}

[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFileName"] {
    color: var(--ink) !important;
}

/* Alert boxes (st.warning / st.info / st.error / st.success) */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlertContentWarning"] p,
[data-testid="stAlertContentInfo"] p,
[data-testid="stAlertContentError"] p,
[data-testid="stAlertContentSuccess"] p {
    color: var(--ink) !important;
    font-weight: 500;
}

/* Tabs: inactive labels were washing out against the light card */
.stTabs [data-baseweb="tab"] p {
    color: var(--ink-soft) !important;
}
.stTabs [aria-selected="true"] p {
    color: var(--primary) !important;
    font-weight: 600;
}

/* st.json viewer in the "Raw data" tab */
[data-testid="stJson"] {
    background: var(--surface) !important;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.5rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------
# HERO HEADER
# --------------------------------------------------

render_html(
    """
    <div class="hero">
        <div class="hero-eyebrow">Surface &amp; Texture Analysis · v2.0</div>
        <div class="hero-title">🧩 Texture Classification</div>
        <div class="hero-subtitle">
            Upload a surface sample and get a live confidence readout across 47 texture classes,
            powered by EfficientNetB3.
        </div>
    </div>
    """
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

    render_html(
        """
        <div class="sidebar-eyebrow">Model Spec Sheet</div>
        <div class="sidebar-title">Configuration</div>
        """
    )

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
    render_html(f'<div>{rows_html}</div>')

    st.markdown("")
    confidence_threshold = st.slider(
        "Flag predictions below this confidence",
        min_value=0,
        max_value=100,
        value=60,
        format="%d%%",
        help="Predictions under this threshold are marked as low-confidence.",
    )

    st.markdown("")
    render_html('<div class="footnote">Model loaded and cached for this session.</div>')


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

render_html('<div class="section-label">01 · Sample Upload</div>')

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    render_html(
        """
        <div class="footnote">
            Accepted formats: JPG, JPEG, PNG, BMP, WEBP. The image is analyzed locally within this session only.
        </div>
        """
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.markdown("")
    render_html('<div class="section-label">02 · Specimen &amp; Reading</div>')

    col1, col2 = st.columns([1, 1], gap="large")

    # ---------------- DISPLAY IMAGE ----------------

    with col1:
        st.image(image, use_container_width=True)
        render_html(f'<div class="footnote">FILE · {uploaded_file.name}</div>')

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

    confidence_pct = result["confidence"] * 100

    if confidence_pct >= 90:
        tier, tier_label = "high", "High confidence"
    elif confidence_pct >= confidence_threshold:
        tier, tier_label = "medium", "Moderate confidence"
    else:
        tier, tier_label = "low", "Low confidence — verify manually"

    # ---------------- MAIN RESULT (gauge) ----------------

    with col2:
        gauge_colors = {"high": "#3FA796", "medium": "#E3A857", "low": "#E4634F"}

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence_pct,
                number={
                    "suffix": "%",
                    "font": {"size": 40, "family": "IBM Plex Mono", "color": "#221933"},
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#6B6280",
                        "tickfont": {"color": "#6B6280", "size": 11},
                    },
                    "bar": {"color": gauge_colors[tier]},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, confidence_threshold], "color": "#F6E9E7"},
                        {"range": [confidence_threshold, 90], "color": "#FBF1E3"},
                        {"range": [90, 100], "color": "#E9F5F2"},
                    ],
                },
            )
        )
        fig.update_layout(
            template="plotly_white",
            height=220,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter", "color": "#221933"},
        )

        render_html(
            f"""
            <div class="card">
                <div class="result-label">Predicted Class</div>
                <div class="result-class">{result["class"]}</div>
                <div class="badge {tier}"><span class="badge-dot"></span>{tier_label}</div>
            </div>
            """
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if tier == "high" and confidence_pct >= 98:
        st.balloons()
    elif tier == "low":
        st.warning(
            f"Confidence is below your {confidence_threshold}% threshold — consider reviewing "
            "the sample or checking the top-5 alternatives below."
        )

    # ---------------- TOP 5 (tabs: ranked view + chart + raw) ----------------

    st.markdown("")
    render_html('<div class="section-label">03 · Ranked Readout — Top 5</div>')

    tab_ranked, tab_chart, tab_raw = st.tabs(["Ranked list", "Chart", "Raw data"])

    with tab_ranked:
        rows = []
        for i, item in enumerate(result["top_predictions"], start=1):
            class_name = item["class"]
            confidence = item["confidence"] * 100
            top_class = "rank-row top" if i == 1 else "rank-row"

            rows.append(
                f'<div class="{top_class}">'
                f'<span class="rank-badge">{i}</span>'
                f'<span class="rank-name">{class_name}</span>'
                f'<div class="rank-track">'
                f'<div class="rank-fill" style="width:{confidence:.2f}%;"></div>'
                f'</div>'
                f'<span class="rank-pct">{confidence:.2f}%</span>'
                f'</div>'
            )

        render_html(f'<div class="card">{"".join(rows)}</div>')

    with tab_chart:
        names = [item["class"] for item in result["top_predictions"]][::-1]
        values = [item["confidence"] * 100 for item in result["top_predictions"]][::-1]
        bar_colors = ["#5B4B9E"] * (len(values) - 1) + ["#E3A857"]

        bar_fig = go.Figure(
            go.Bar(
                x=values,
                y=names,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.2f}%" for v in values],
                textposition="outside",
                textfont={"color": "#221933", "family": "IBM Plex Mono"},
                hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
            )
        )
        bar_fig.update_layout(
            template="plotly_white",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                range=[0, max(values) * 1.2],
                showgrid=False,
                ticksuffix="%",
                tickfont={"color": "#6B6280"},
            ),
            yaxis=dict(tickfont={"color": "#221933"}),
            font={"family": "Inter", "color": "#221933"},
        )
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

    with tab_raw:
        st.json(result)