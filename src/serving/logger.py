"""
Prediction logging for monitoring (Week 4 / M5).

Every call to log_prediction() appends one row to a CSV file. This is the
raw material week4_monitoring.ipynb reads from -- keeping logging and
analysis as two separate, independently testable pieces.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

LOG_PATH = Path(__file__).resolve().parents[2] / "monitoring" / "prediction_log.csv"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_FIELDNAMES = [
    "timestamp", "filename", "label", "is_defective",
    "confidence", "raw_probability", "model_used",
]
_lock = Lock()  # guards against two near-simultaneous requests corrupting a row


def log_prediction(filename: str, result: dict) -> None:
    """Append one prediction to the log. Called from the API right after a
    prediction is made, before the response is sent back."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "label": result["label"],
        "is_defective": result["is_defective"],
        "confidence": result["confidence"],
        "raw_probability": result["raw_probability"],
        "model_used": result["model_used"],
    }

    with _lock:
        file_exists = LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
