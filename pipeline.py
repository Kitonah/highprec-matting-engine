import numpy as np
from PIL import Image
import cv2
from core.segmentation import HighResolutionDISPipeline
from core.frequency import FrequencyFeatureExtractor

class HighPrecisionExtractionEngine:
    def __init__(self):
        print("[Pipeline] Booting production extraction subsystems...")
        self.seg_engine = HighResolutionDISPipeline()
        self.freq_engine = FrequencyFeatureExtractor()
        print("[Pipeline] Engine ready.")

    def execute(self, image: Image.Image) -> Image.Image:
        rgb_image = image.convert("RGB")
        w, h = rgb_image.size
        
        rgb_raw = np.array(rgb_image).astype(np.float32) / 255.0
        gray_raw = cv2.cvtColor((rgb_raw * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0

        # --- 1. BiRefNet Dense Segmentation ---
        raw_mask = self.seg_engine.predict_mask(rgb_image)
        if raw_mask.shape != (h, w):
            raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # --- 2. Low-Contrast Camouflage Edge Recovery (FFT Phase) ---
        # Fourier phase exposes high-frequency structural boundaries where luminance fails
        phase_map = self.freq_engine.extract_fourier_phase_map(gray_raw)
        if phase_map.shape != (h, w):
            phase_map = cv2.resize(phase_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Modulate uncertain transition zones with physical phase edges
        uncertain_zone = (raw_mask > 0.15) & (raw_mask < 0.85)
        raw_mask[uncertain_zone] = np.clip(
            raw_mask[uncertain_zone] * 0.7 + phase_map[uncertain_zone] * 0.3, 0.0, 1.0
        )

        # --- 3. Clean Edge Disentanglement ---
        # White products on white backdrops require high-fidelity edge clamping
        # to eliminate haloing and prevent foreground dropouts
        alpha = np.zeros_like(raw_mask)
        alpha[raw_mask > 0.50] = 1.0
        
        # Soften sub-pixel margin strictly within a 3px radius (no blurry feathering)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate((alpha * 255).astype(np.uint8), kernel)
        eroded = cv2.erode((alpha * 255).astype(np.uint8), kernel)
        
        edge_band = (dilated > 0) & (eroded == 0)
        alpha[edge_band] = raw_mask[edge_band]
        
        # Bilateral edge preservation along the jar perimeter
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        smooth_alpha = cv2.bilateralFilter(alpha_uint8, d=5, sigmaColor=30, sigmaSpace=30)
        
        # --- 4. Final RGBA Composite ---
        rgb_np = np.array(rgb_image)
        rgba = np.dstack([rgb_np, smooth_alpha])
        return Image.fromarray(rgba, mode="RGBA")