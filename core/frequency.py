"""
Module: Multi-Band Fourier Phase & Wavelet Feature Extraction
Author: Ekantika Kumari
"""

import numpy as np
import pywt

class FrequencyFeatureExtractor:
    @staticmethod
    def extract_fourier_phase_map(gray_img: np.ndarray) -> np.ndarray:
        f = np.fft.fft2(gray_img)
        fshift = np.fft.fftshift(f)
        phase = np.angle(fshift)
        
        recon_complex = np.exp(1j * phase)
        inv_shift = np.fft.ifftshift(recon_complex)
        phase_map = np.abs(np.fft.ifft2(inv_shift))
        
        phase_map = (phase_map - np.min(phase_map)) / (np.ptp(phase_map) + 1e-6)
        return phase_map.astype(np.float32)

    @staticmethod
    def extract_wavelet_subbands(gray_img: np.ndarray) -> np.ndarray:
        coeffs2 = pywt.dwt2(gray_img, 'haar')
        LL, (LH, HL, HH) = coeffs2
        
        high_pass = np.sqrt(LH**2 + HL**2 + HH**2)
        high_pass = (high_pass - np.min(high_pass)) / (np.ptp(high_pass) + 1e-6)
        return high_pass.astype(np.float32)