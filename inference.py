import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
from models import build_model


class CrowdCounter:
    def __init__(self, weight_path):

        # -----------------------------
        # Device
        # -----------------------------

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")

        # -----------------------------
        # Model arguments
        # -----------------------------

        class Args:
            backbone = "vgg16_bn"
            row = 2
            line = 2

        args = Args()

        # -----------------------------
        # Build model
        # -----------------------------

        self.model = build_model(args)

        self.model.to(self.device)

        # -----------------------------
        # Load checkpoint
        # -----------------------------

        checkpoint = torch.load(weight_path, map_location="cpu")

        self.model.load_state_dict(checkpoint["model"])

        self.model.eval()

        # -----------------------------
        # Image preprocessing
        # -----------------------------

        self.transform = standard_transforms.Compose(
            [
                standard_transforms.ToTensor(),
                standard_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, image):

        # Convert to RGB
        image = image.convert("RGB")

        # Original image size
        original_width, original_height = image.size

        # P2PNet expects dimensions divisible by 128
        new_width = (original_width // 128) * 128

        new_height = (original_height // 128) * 128

        # Avoid zero dimension
        new_width = max(new_width, 128)
        new_height = max(new_height, 128)

        # Resize image
        resized_image = image.resize((new_width, new_height))

        # Transform image
        image_tensor = self.transform(resized_image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Move to device
        image_tensor = image_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Classification scores
        outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[
            :, :, 1
        ][0]

        # Point coordinates
        outputs_points = outputs["pred_points"][0]

        # Confidence threshold
        threshold = 0.5

        # Filter points
        keep = outputs_scores > threshold

        points = outputs_points[keep].detach().cpu().numpy()

        scores = outputs_scores[keep].detach().cpu().numpy()

        # Count
        count = len(points)

        return {
            "count": count,
            "points": points,
            "scores": scores,
            "image": resized_image,
        }

    # Note: interpolation helper removed; use `util.misc.interpolate` if needed.
