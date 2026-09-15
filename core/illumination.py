"""
Module: Illumination Invariant Modeling & Shadow Cleaving
Author: Ekantika Kumari
"""

import cv2
import numpy as np

class IlluminationInvariantProcessor:
    def __init__(self, target_angle: float = 0.58):
        self.theta = target_angle

    def extract_invariant_scalar(self, rgb_img: np.ndarray) -> np.ndarray:
        eps = 1e-6
        img = np.maximum(rgb_img, eps)
        
        geo_mean = np.cbrt(img[..., 0] * img[..., 1] * img[..., 2])
        chi_r = np.log(img[..., 0] / geo_mean)
        chi_g = np.log(img[..., 1] / geo_mean)
        chi_b = np.log(img[..., 2] / geo_mean)
        
        p1 = chi_r / np.sqrt(2) - chi_g / np.sqrt(2)
        p2 = chi_r / np.sqrt(6) + chi_g / np.sqrt(6) - 2.0 * chi_b / np.sqrt(6)
        
        i_theta = p1 * np.cos(self.theta) + p2 * np.sin(self.theta)
        norm = (i_theta - np.min(i_theta)) / (np.ptp(i_theta) + eps)
        return norm.astype(np.float32)

    def compute_shadow_edge_mask(self, rgb_img: np.ndarray) -> np.ndarray:
        gray_rgb = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        inv_scalar = (self.extract_invariant_scalar(rgb_img) * 255).astype(np.uint8)
        
        grad_rgb = cv2.Sobel(gray_rgb, cv2.CV_32F, 1, 1, ksize=3)
        grad_inv = cv2.Sobel(inv_scalar, cv2.CV_32F, 1, 1, ksize=3)
        
        diff = np.abs(grad_rgb) - np.abs(grad_inv)
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        
        _, shadow_edges = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        return shadow_edges