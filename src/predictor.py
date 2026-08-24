import os
import numpy as np
import tensorflow as tf


# --------------------------------------------------
# MODEL PATH
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(__file__)
    ),
    "model",
    "best_efficientnetb3.keras"
)


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

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict(model, processed_image):

    # Get predictions from the model
    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    # Find class with highest probability
    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    # Find top 5 predictions
    top_indices = np.argsort(
        predictions
    )[::-1][:5]

    top_predictions = []

    for index in top_indices:

        top_predictions.append({
            "class": CLASS_NAMES[index],
            "confidence": float(
                predictions[index]
            )
        })

    return {
        "class": predicted_class,
        "confidence": confidence,
        "top_predictions": top_predictions
    }