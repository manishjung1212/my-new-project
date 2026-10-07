import os
import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# MiniFASNetV2 model
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "minifasnet_v2.onnx"
)

# Real face must have at least 0.50 probability
LIVENESS_THRESHOLD = 0.50


# ============================================================
# Load model
# ============================================================

_session = None


def _load_model():
    global _session

    if _session is None:

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"MiniFASNet model not found:\n{MODEL_PATH}"
            )

        _session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )

    return _session


# ============================================================
# Crop face
# ============================================================

def _crop_face(image, bbox):

    src_h, src_w = image.shape[:2]

    # Our bbox format:
    # [x1, y1, x2, y2]

    x1, y1, x2, y2 = [float(v) for v in bbox]

    # Convert to:
    # x, y, width, height

    x = x1
    y = y1
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return None

    # MiniFASNet uses 2.7x scale
    scale = 2.7

    # Keep the crop inside the image while
    # preserving the requested scale as much as possible.
    scale = min(
        (src_h - 1) / box_h,
        (src_w - 1) / box_w,
        scale
    )

    new_width = box_w * scale
    new_height = box_h * scale

    # Face center
    center_x = x + box_w / 2
    center_y = y + box_h / 2

    # Calculate expanded box
    left = center_x - new_width / 2
    top = center_y - new_height / 2
    right = center_x + new_width / 2
    bottom = center_y + new_height / 2

    # Move box instead of simply cutting it
    # when it touches image boundaries.

    if left < 0:
        right -= left
        left = 0

    if top < 0:
        bottom -= top
        top = 0

    if right > src_w - 1:
        left -= (right - src_w + 1)
        right = src_w - 1

    if bottom > src_h - 1:
        top -= (bottom - src_h + 1)
        bottom = src_h - 1

    # Final safety limits
    left = max(0, int(left))
    top = max(0, int(top))
    right = min(src_w, int(right))
    bottom = min(src_h, int(bottom))

    if right <= left or bottom <= top:
        return None

    face = image[
        top:bottom,
        left:right
    ]

    return face


# ============================================================
# Prepare image for MiniFASNet
# ============================================================

def _prepare_face(face):

    # Resize to MiniFASNet input size
    face = cv2.resize(
        face,
        (80, 80)
    )

    # Convert BGR image to float32
    # IMPORTANT:
    # Do NOT divide by 255 for this ONNX inference.
    face = face.astype(np.float32)

    # HWC -> CHW
    face = np.transpose(
        face,
        (2, 0, 1)
    )

    # CHW -> NCHW
    face = np.expand_dims(
        face,
        axis=0
    )

    return face


# ============================================================
# MiniFASNet liveness score
# ============================================================

def _minifasnet_score(face):

    session = _load_model()

    # Get model input name
    input_name = session.get_inputs()[0].name

    # Prepare image
    input_data = _prepare_face(face)

    # Run ONNX model
    output = session.run(
        None,
        {
            input_name: input_data
        }
    )[0][0]
    output = session.run(
    None,
    {
        input_name: input_data
    }
    )[0][0]

    print("RAW MODEL OUTPUT:", output)

    # Convert output to probabilities
    exp_output = np.exp(
        output - np.max(output)
    )

    probabilities = (
        exp_output / exp_output.sum()
    )

    # ========================================================
    # MiniFASNetV2 classes
    #
    # 0 = Fake
    # 1 = Real / Live
    # 2 = Fake
    # ========================================================

    fake_probability_0 = float(
        probabilities[0]
    )

    live_probability = float(
        probabilities[1]
    )

    fake_probability_2 = float(
        probabilities[2]
    )

    # Class 1 represents a real/live face
    score = live_probability

    # Show result in terminal
    print("\n------------------------------")

    print(
        f"Fake probability 0 : "
        f"{fake_probability_0:.3f}"
    )

    print(
        f"Live probability   : "
        f"{live_probability:.3f}"
    )

    print(
        f"Fake probability 2 : "
        f"{fake_probability_2:.3f}"
    )

    print(
        f"Liveness score     : "
        f"{score:.3f}"
    )

    print("------------------------------")

    return score


# ============================================================
# Engine name
# ============================================================

def engine_name():

    return "MiniFASNetV2 (ONNX)"


# ============================================================
# Main liveness function
# ============================================================

def is_live_face(
    bgr_image,
    bbox,
    threshold=None
):

    if threshold is None:
        threshold = LIVENESS_THRESHOLD

    # Crop detected face
    face = _crop_face(
        bgr_image,
        bbox
    )

    if face is None or face.size == 0:

        return (
            False,
            0.0,
            engine_name()
        )

    try:

        # Get liveness score
        score = _minifasnet_score(
            face
        )

    except Exception as e:

        print(
            "Liveness model error:",
            e
        )

        return (
            False,
            0.0,
            engine_name()
        )

    # Decide whether face is live
    is_live = score >= threshold

    return (
        is_live,
        score,
        engine_name()
    )