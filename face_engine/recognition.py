"""
Face detection + recognition using InsightFace (ArcFace embeddings).

InsightFace is lazy-imported (only when actually used) so the rest of the
Flask app can start and run even on a machine where `insightface` hasn't
been installed yet / has no internet to fetch its models -- only the
face-related routes will fail, with a clear error, instead of the whole
server refusing to boot.

On first real use, InsightFace downloads its "buffalo_l" model pack
(detection + recognition, ~300MB) automatically to ~/.insightface/models.
That download needs an internet connection on whatever machine actually
runs the app -- it is not something that can be bundled in the project
zip.
"""

import numpy as np

_app = None

# Cosine similarity threshold for "is this the same person". ArcFace
# embeddings from the buffalo_l pack are well separated; 0.45 is a
# reasonably strict default -- raise it (e.g. 0.55) for stricter matching,
# lower it (e.g. 0.35) if legitimate logins are being rejected too often.
MATCH_THRESHOLD = 0.45


class FaceEngineUnavailable(Exception):
    """Raised when insightface/onnxruntime aren't installed or fail to load."""
    pass


def get_app():
    """Lazily creates (once) and returns the InsightFace FaceAnalysis app."""
    global _app
    if _app is None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise FaceEngineUnavailable(
                "insightface is not installed. Run: pip install -r requirements.txt "
                "(this needs an internet connection the first time, to download the "
                "ArcFace model pack)."
            ) from e

        _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(320, 320))
    return _app


def detect_faces(bgr_image):
    """Returns a list of InsightFace Face objects (.bbox, .det_score, .embedding, .normed_embedding)."""
    return get_app().get(bgr_image)


def get_largest_face(bgr_image):
    """Returns the largest detected face (by bounding-box area), or None."""
    faces = detect_faces(bgr_image)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def embeddings_match(a, b, threshold=None):
    return cosine_similarity(a, b) >= (threshold if threshold is not None else MATCH_THRESHOLD)
