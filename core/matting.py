import cv2
import numpy as np

class BoundaryAdaptiveMatting:
    @staticmethod
    def generate_entropy_trimap(binary_mask: np.ndarray, rgb_img: np.ndarray) -> np.ndarray:
        h, w = binary_mask.shape
        
        # Fast proxy calculation on downsampled preview to keep response snappy
        small_gray = cv2.resize(
            cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
            (512, 512), 
            interpolation=cv2.INTER_NEAREST
        )
        
        _, std = cv2.meanStdDev(small_gray)
        entropy_factor = float(std[0][0]) / 128.0
        
        # Clamp kernel between 3 and 15 pixels max
        k_radius = int(np.clip(entropy_factor * 12, 3, 15))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_radius, k_radius))
        
        binary_uint8 = (binary_mask * 255).astype(np.uint8)
        dilated = cv2.dilate(binary_uint8, kernel, iterations=1)
        eroded = cv2.erode(binary_uint8, kernel, iterations=1)
        
        trimap = np.zeros((h, w), dtype=np.uint8)
        trimap[dilated > 0] = 128
        trimap[eroded == 255] = 255
        return trimap

    @staticmethod
    def refine_alpha_guided(rgb_img: np.ndarray, coarse_alpha: np.ndarray, radius: int = 9, eps: float = 1e-4) -> np.ndarray:
        src = (coarse_alpha * 255).astype(np.uint8)
        # Fast bilateral edge smoothing
        refined = cv2.bilateralFilter(src, d=radius, sigmaColor=50, sigmaSpace=50)
        return np.clip(refined.astype(np.float32) / 255.0, 0.0, 1.0)