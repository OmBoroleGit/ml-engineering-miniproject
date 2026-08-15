import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class CastingPredictor:
    def __init__(self, model_path: str = "models/resnet18_transfer_best.pth"):
        """
        Step 1: Wake up the model.
        This runs once when the API starts so it is fast.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Rebuild the exact ResNet18 architecture from Week 2
        self.model = models.resnet18(weights=None)
        num_features = self.model.fc.in_features
        
        # Based on week2_transfer_learning.ipynb, the final layer outputs 1 value via Sigmoid
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid() 
        )
        
        # Load the saved weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        # Set to evaluation mode (crucial for inference!)
        self.model.to(self.device)
        self.model.eval()

        # Step 2: Set up the Image Translator
        # Matches Week 1, but we MUST include Resize(224, 224) because 
        # users might upload images of any size to our web API.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Takes raw uploaded bytes and turns them into a PyTorch Tensor."""
        # Convert to RGB as done in week2_transfer_learning.ipynb
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image)
        # Models expect batches, so we add a dummy batch dimension: [1, 3, 224, 224]
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image_bytes: bytes) -> dict:
        """
        Step 3: Translate the answer.
        Takes an image, runs the model, and returns a clean dictionary.
        """
        tensor = self.preprocess_image(image_bytes)

        with torch.no_grad(): # Don't track gradients (saves memory & runs faster)
            # The model outputs a single probability between 0.0 and 1.0
            probability = self.model(tensor).item()

        # Based on Week 1 CLASS_MAP: 1 = defective, 0 = ok
        is_defective = probability >= 0.5
        
        if is_defective:
            prediction_label = "def_front"
            confidence_score = probability # Closer to 1.0 is higher confidence of defect
        else:
            prediction_label = "ok_front"
            confidence_score = 1.0 - probability # Closer to 0.0 is higher confidence of ok

        return {
            "prediction": prediction_label,
            "confidence": round(confidence_score, 4),
            "is_defective": is_defective
        }

# --- Quick Test Block ---
if __name__ == "__main__":
    print("Testing Model Packaging...")
    try:
        predictor = CastingPredictor()
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")