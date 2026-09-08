import streamlit as st
from PIL import Image

from src.preprocessing import preprocess_image
from src.predictor import load_model, predict


# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Texture Classification",
    page_icon="🧩",
    layout="wide"
)


# -------------------------------
# TITLE
# -------------------------------

st.title("🧩 Texture Classification")

st.write(
    "Upload a surface image and the AI model will classify "
    "its texture."
)

st.divider()


# -------------------------------
# SIDEBAR
# -------------------------------

with st.sidebar:

    st.header("Model Information")

    st.write("**Architecture:** EfficientNetB3")
    st.write("**Input Size:** 300 × 300")
    st.write("**Classes:** 47")
    st.write("**Framework:** TensorFlow / Keras")

    st.divider()


# -------------------------------
# LOAD MODEL
# -------------------------------

@st.cache_resource
def get_model():
    return load_model()


try:

    model = get_model()

except Exception as e:

    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


# -------------------------------
# IMAGE UPLOAD
# -------------------------------

st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Choose a surface image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)


# -------------------------------
# PREDICTION
# -------------------------------

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    # IMAGE
    with col1:

        st.subheader("Input Image")

        st.image(
            image,
            use_container_width=True
        )


    # PREDICTION
    with col2:

        st.subheader("Prediction")

        with st.spinner("Analyzing image..."):

            try:

                image_bytes = uploaded_file.getvalue()

                processed_image = preprocess_image(
                    image_bytes
                )

                result = predict(
                    model,
                    processed_image
                )

            except Exception as e:

                st.error("Prediction failed.")
                st.exception(e)
                st.stop()


        # Main prediction
        predicted_class = result["class"]

        confidence = result["confidence"] * 100


        st.metric(
            "Predicted Texture",
            predicted_class
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        # Confidence progress bar
        st.progress(
            int(confidence)
        )


        # Confidence status
        if confidence >= 90:

            st.success(
                "High confidence prediction"
            )

        elif confidence >= 60:

            st.info(
                "Moderate confidence prediction"
            )

        else:

            st.warning(
                "Low confidence — manual verification recommended"
            )


# -------------------------------
# TOP 5 PREDICTIONS
# -------------------------------

if uploaded_file:

    st.divider()

    st.subheader("📊 Top 5 Predictions")

    for i, item in enumerate(
        result["top_predictions"],
        start=1
    ):

        class_name = item["class"]

        probability = item["confidence"] * 100

        st.write(
            f"**{i}. {class_name}** — "
            f"{probability:.2f}%"
        )

        st.progress(
            int(probability)
        )


# -------------------------------
# FOOTER
# -------------------------------

st.divider()

st.caption(
    "Surface & Texture Analysis • "
    "Powered by EfficientNetB3"
)
