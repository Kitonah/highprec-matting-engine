import cv2
import numpy as np

class ColorDecontamination:
    @staticmethod
    def unmix_backdrop(rgb_img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Fast production despill: downsamples background estimation 
        to avoid CPU freezing on high-res images, preserving sub-pixel speeds.
        """
        clean_rgb = rgb_img.copy()
        
        # Identify edge transition zone (semi-transparent pixels)
        translucent_band = (alpha > 0.03) & (alpha < 0.97)
        if not np.any(translucent_band):
            return clean_rgb

        h, w = alpha.shape
        
        # Downsample for fast background color estimation (100x faster than full-res inpaint)
        scale = 0.25
        small_w, small_h = max(int(w * scale), 64), max(int(h * scale), 64)
        
        small_img = cv2.resize(rgb_img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small_alpha = cv2.resize(alpha, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        # Mask of known background
        small_bg_mask = (small_alpha < 0.1).astype(np.uint8) * 255
        
        # Fast inpaint on low-res proxy
        est_bg_small = cv2.inpaint((small_img * 255).astype(np.uint8), 255 - small_bg_mask, 5, cv2.INPAINT_TELEA)
        est_bg = cv2.resize(est_bg_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        
        # Unmix foreground color from estimated background
        alpha_3d = np.repeat(alpha[..., np.newaxis], 3, axis=2)
        numerator = rgb_img - (1.0 - alpha_3d) * est_bg
        unmixed = np.clip(numerator / np.maximum(alpha_3d, 0.2), 0.0, 1.0)
        
        clean_rgb[translucent_band] = unmixed[translucent_band]
        return clean_rgb