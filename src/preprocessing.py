import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input


IMAGE_SIZE = 300


def preprocess_image(image_bytes):
    """
    Convert an uploaded image into the format
    expected by the trained EfficientNetB3 model.
    """

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Could not read the uploaded image."
        )

    # BGR → RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize to training size
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Convert to float32
    image = image.astype(np.float32)

    # EfficientNet preprocessing
    image = preprocess_input(image)

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image