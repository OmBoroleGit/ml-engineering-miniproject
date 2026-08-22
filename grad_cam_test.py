import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class_score):
        self.model.zero_grad()
        target_class_score.backward(retain_graph=True)

        # Global average pooling on the gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the channels by corresponding gradients
        activations = self.activations[0]
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]

        # Average the channels to get a single 2D heatmap
        heatmap = torch.mean(activations, dim=0).squeeze()
        heatmap = F.relu(heatmap)  # Keep only positive influence

        # Normalize heatmap between 0 and 1
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)

        return heatmap.cpu().detach().numpy()


def run_gradcam(image_path="cast_stress_test6.jpg", model_path="models/resnet18_transfer_best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1),
        nn.Sigmoid()
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Target the last convolutional block of ResNet-18
    target_layer = model.layer4[-1].conv2
    cam = GradCAM(model, target_layer)

    # 3. Load and Preprocess Image
    raw_img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    # 4. Forward Pass
    prob = model(input_tensor)
    prob_val = prob.item()
    is_defective = prob_val >= 0.5

    # Explain the winning class
    if is_defective:
        predicted_label = "def_front"
        confidence = prob_val
        target_score = prob
    else:
        predicted_label = "ok_front"
        confidence = 1.0 - prob_val
        target_score = 1.0 - prob  # Score for being normal

    # 5. Generate Heatmap
    heatmap = cam.generate_heatmap(input_tensor, target_score)

    # 6. Resize Heatmap to original image dimensions
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(raw_img.size, Image.Resampling.BILINEAR)
    heatmap_resized = np.array(heatmap_resized) / 255.0

    # 7. Plot and Save the Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(raw_img)
    axes[0].set_title(f"Input Image\nPrediction: {predicted_label} ({confidence*100:.1f}%)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Activation Heatmap", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(raw_img)
    axes[2].imshow(heatmap_resized, cmap="jet", alpha=0.5)
    axes[2].set_title("Superimposed Focus Map", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    output_path = "gradcam_result.png"
    plt.savefig(output_path, dpi=200)
    print(f"✓ Activation map generated and saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    run_gradcam()