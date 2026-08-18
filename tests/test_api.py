from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from src.serving.app import app


client = TestClient(app)


def make_test_image():
    """Create a small valid JPEG image entirely in memory."""
    image = Image.new("RGB", (100, 100), color="white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_image():
    image = make_test_image()

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", image, "image/jpeg")},
    )

    assert response.status_code == 200

    data = response.json()

    assert "label" in data
    assert "is_defective" in data
    assert "confidence" in data
    assert "raw_probability" in data
    assert "model_used" in data

    assert isinstance(data["is_defective"], bool)
    assert 0 <= data["confidence"] <= 1
    assert 0 <= data["raw_probability"] <= 1


def test_empty_upload():
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "Uploaded file is empty."


def test_corrupted_image():
    response = client.post(
        "/predict",
        files={
            "file": (
                "broken.jpg",
                b"this is not a real image",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400
    assert "not a readable image" in response.json()["error"]


def test_non_image_file():
    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"hello, this is not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert "not a readable image" in response.json()["error"]