"""
Week 3 (M4) -- serving API.

Run locally with:
    uvicorn src.serving.app:app --reload

Then open http://127.0.0.1:8000/docs for an interactive test page (upload a
file straight from the browser, no command line needed).
"""

import io

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.serving.logger import log_prediction

# ---------------------------------------------------------------------------
# Swap this one import to change which predictor the API uses.
# Everything else in this file stays exactly the same -- .predict() has the
# identical interface either way.
from src.serving.predictor_adapter import DefectPredictorAdapter as Predictor
# from src.serving.placeholder_predictor import PlaceholderPredictor as Predictor
# ---------------------------------------------------------------------------

app = FastAPI(title="Casting Defect Classifier API")

# Loaded once, when the server starts -- not per request, which would be slow.
predictor = Predictor()


@app.get("/health")
def health_check():
    """Quick check that the server is up and the model loaded successfully."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    if not contents:
        return JSONResponse(
            status_code=400,
            content={"error": "Uploaded file is empty."},
        )

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents))  # re-open after verify()
    except UnidentifiedImageError:
        return JSONResponse(
            status_code=400,
            content={"error": "File is not a readable image. Please upload a JPEG or PNG."},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"error": f"Could not process the uploaded file: {exc}"},
        )

    result = predictor.predict(image)
    log_prediction(file.filename or "unknown", result)
    return result
