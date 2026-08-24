import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input


IMAGE_SIZE = 300


def preprocess_image(image_bytes):
    """
    Convert an uploaded image into the format
    expected by the trained EfficientNetB3 model.
    """

    # Convert uploaded bytes into NumPy array
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    # Decode image using OpenCV
    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Could not read the uploaded image."
        )

    # OpenCV uses BGR.
    # Convert BGR -> RGB.
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize to the same size used during training
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Convert to float32
    image = image.astype(
        np.float32
    )

    # EfficientNet preprocessing
    image = preprocess_input(image)

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image