"""
Adapter between Abhilash's CastingPredictor (src/predict.py) and the
interface the API (src/serving/app.py) expects.

Nothing in src/predict.py is modified for this to work -- it's used exactly
as he wrote it. This file bridges two small differences:

1. CastingPredictor.predict() takes raw image bytes. The API works with a
   PIL Image (so it can validate the upload before prediction). This
   adapter converts back to bytes right before calling his code.
2. CastingPredictor's output uses different field names ("prediction",
   no raw probability) than the response shape the API already returns
   and has been tested against ("label", "raw_probability", etc.). This
   adapter translates field names -- the underlying prediction itself is
   untouched.
"""

import io

from PIL import Image

from src.predict import CastingPredictor


class DefectPredictorAdapter:
    def __init__(self):
        self._predictor = CastingPredictor()

    def predict(self, image: Image.Image) -> dict:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        raw_bytes = buffer.getvalue()

        result = self._predictor.predict(raw_bytes)

        confidence = result["confidence"]
        is_defective = result["is_defective"]
        raw_probability = confidence if is_defective else 1 - confidence

        return {
            "label": result["prediction"],
            "is_defective": is_defective,
            "confidence": confidence,
            "raw_probability": raw_probability,
            "model_used": "resnet18_transfer",
        }
