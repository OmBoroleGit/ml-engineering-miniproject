from src.predict import CastingPredictor

# 1. Initialize your model package (loads the weights)
predictor = CastingPredictor()

# 2. Open your manually disturbed image file as raw bytes
# Replace this with the actual name of your disturbed photo
image_path = "cast_stress_test4.jpg" 
with open(image_path, "rb") as file:
    raw_image_bytes = file.read()

# 3. Ask the model for a prediction
result = predictor.predict(raw_image_bytes)

# 4. View the results!
print(f"--- Results for {image_path} ---")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence'] * 100}%")
print(f"Is Defective: {result['is_defective']}")