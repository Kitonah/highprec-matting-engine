import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModelForImageSegmentation

class HighResolutionDISPipeline:
    def __init__(self, model_id: str = "ZhengPeng7/BiRefNet"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[Segmentation] Initializing High-Precision BiRefNet on {self.device}...")
        
        # Load the official Bilateral Reference Network
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_id, 
            trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()

        # BiRefNet native input dimension
        self.input_size = (1024, 1024)
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        print("[Segmentation] BiRefNet loaded successfully.")

    @torch.inference_mode()
    def predict_mask(self, image: Image.Image) -> np.ndarray:
        w, h = image.size
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        
        # BiRefNet forward pass
        preds = self.model(tensor)
        
        # Extract prediction tensor
        if isinstance(preds, (list, tuple)):
            pred_tensor = preds[-1]  # BiRefNet yields the final reconstructed mask at -1
            while isinstance(pred_tensor, (list, tuple)):
                pred_tensor = pred_tensor[0]
        else:
            pred_tensor = preds

        # Normalize via sigmoid
        pred_tensor = torch.sigmoid(pred_tensor)
        mask = pred_tensor.squeeze().detach().cpu().numpy().astype(np.float32)

        # Scale to native image dimensions
        mask_uint8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode="L").resize((w, h), Image.Resampling.BILINEAR)
        
        return np.array(mask_pil).astype(np.float32) / 255.0