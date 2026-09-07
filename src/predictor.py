import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# HUGGING FACE MODEL
# --------------------------------------------------

HF_REPO_ID = "DB03/best_efficientnetb3_finetuned.keras"

MODEL_FILENAME = "best_efficientnetb3_finetuned.keras"


# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

CLASS_NAMES = [
    "banded",
    "blotchy",
    "braided",
    "bubbly",
    "bumpy",
    "chequered",
    "cobwebbed",
    "cracked",
    "crosshatched",
    "crystalline",
    "dotted",
    "fibrous",
    "flecked",
    "freckled",
    "frilly",
    "gauzy",
    "grid",
    "grooved",
    "honeycombed",
    "interlaced",
    "knitted",
    "lacelike",
    "lined",
    "marbled",
    "matted",
    "meshed",
    "paisley",
    "perforated",
    "pitted",
    "pleated",
    "polka-dotted",
    "porous",
    "potholed",
    "scaly",
    "smeared",
    "spiralled",
    "sprinkled",
    "stained",
    "stratified",
    "striped",
    "studded",
    "swirly",
    "veined",
    "waffled",
    "woven",
    "wrinkled",
    "zigzagged"
]


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict(model, processed_image):

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    if len(predictions) != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {len(predictions)} classes, "
            f"but CLASS_NAMES contains {len(CLASS_NAMES)} classes."
        )

    predicted_index = int(np.argmax(predictions))

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(predictions[predicted_index])

    top_indices = np.argsort(predictions)[::-1][:5]

    top_predictions = []

    for index in top_indices:

        top_predictions.append({
            "class": CLASS_NAMES[int(index)],
            "confidence": float(predictions[index])
        })

    return {
        "class": predicted_class,
        "confidence": confidence,
        "top_predictions": top_predictions
    }