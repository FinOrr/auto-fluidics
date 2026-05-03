"""
Channel Detection for Microfluidic Images

Detects the microfluidic channel region to constrain particle detection
and eliminate false positives from watermarks and artifacts.

Uses conventional computer vision techniques - no ML required.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict


class ChannelDetector:
    """
    Detects microfluidic channel boundaries in microscopy images.

    Supports multiple detection strategies with automatic fallback.
    """

    def __init__(self,
                 min_aspect_ratio: float = 8.0,
                 min_area_ratio: float = 0.15,
                 margin_pixels: int = 10):
        """
        Initialize channel detector.

        Args:
            min_aspect_ratio: Minimum width/height ratio for valid channel
            min_area_ratio: Minimum fraction of image area channel must occupy
            margin_pixels: Safety margin to add around detected bounds
        """
        self.min_aspect_ratio = min_aspect_ratio
        self.min_area_ratio = min_area_ratio
        self.margin_pixels = margin_pixels

        self.channel_bounds = None  # (x_min, y_min, x_max, y_max)
        self.detection_method = None
        self.confidence = 0.0

    def detect_channel(self, image: np.ndarray, method: str = 'auto') -> Optional[Tuple[int, int, int, int]]:
        """
        Detect channel boundaries in image.

        Args:
            image: Input image (color or grayscale)
            method: 'auto', 'morphological', 'hough', or 'profile'

        Returns:
            Tuple of (x_min, y_min, x_max, y_max) or None if detection failed
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape

        if method == 'auto':
            # Try methods in order of robustness
            bounds = self._detect_morphological(gray)
            if bounds:
                self.detection_method = 'morphological'
                self.channel_bounds = bounds
                return bounds

            bounds = self._detect_profile(gray)
            if bounds:
                self.detection_method = 'profile'
                self.channel_bounds = bounds
                return bounds

            bounds = self._detect_hough(gray)
            if bounds:
                self.detection_method = 'hough'
                self.channel_bounds = bounds
                return bounds

            # Fallback: use middle 60% of image
            self.detection_method = 'fallback'
            margin_y = int(h * 0.2)
            bounds = (0, margin_y, w, h - margin_y)
            self.channel_bounds = bounds
            self.confidence = 0.3
            return bounds

        elif method == 'morphological':
            return self._detect_morphological(gray)
        elif method == 'hough':
            return self._detect_hough(gray)
        elif method == 'profile':
            return self._detect_profile(gray)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _detect_morphological(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect channel using morphological operations.

        Strategy:
        1. Threshold image
        2. Morphological closing to fill channel
        3. Find largest horizontal rectangle
        """
        h, w = gray.shape

        # Otsu threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Try both normal and inverted (channel could be dark or light)
        for img in [binary, 255 - binary]:
            # Morphological closing to fill channel
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 3))
            closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_h)

            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Get bounding rectangle
                x, y, cw, ch = cv2.boundingRect(contour)

                # Check criteria
                aspect_ratio = cw / ch if ch > 0 else 0
                area_ratio = (cw * ch) / (w * h)

                if aspect_ratio > self.min_aspect_ratio and area_ratio > self.min_area_ratio:
                    # Add margin
                    x_min = max(0, x - self.margin_pixels)
                    y_min = max(0, y - self.margin_pixels)
                    x_max = min(w, x + cw + self.margin_pixels)
                    y_max = min(h, y + ch + self.margin_pixels)

                    self.confidence = min(1.0, area_ratio / 0.5)  # Higher area = higher confidence
                    return (x_min, y_min, x_max, y_max)

        return None

    def _detect_hough(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect channel using Hough line detection.

        Strategy:
        1. Edge detection
        2. Hough lines - find horizontal lines
        3. Group parallel lines as top/bottom boundaries
        """
        h, w = gray.shape

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough line detection
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                                minLineLength=w//3, maxLineGap=50)

        if lines is None:
            return None

        # Find horizontal lines (nearly 0 or 180 degrees)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Nearly horizontal (within 10 degrees)
            if angle < 10 or angle > 170:
                y_avg = (y1 + y2) / 2
                horizontal_lines.append(y_avg)

        if len(horizontal_lines) < 2:
            return None

        # Find top and bottom boundaries (cluster lines)
        horizontal_lines.sort()

        # Top boundary: cluster of lines in upper region
        # Bottom boundary: cluster of lines in lower region
        top_y = int(np.median(horizontal_lines[:len(horizontal_lines)//2]))
        bottom_y = int(np.median(horizontal_lines[len(horizontal_lines)//2:]))

        channel_height = bottom_y - top_y

        # Validate
        if channel_height < h * 0.1:  # Too small
            return None

        aspect_ratio = w / channel_height
        if aspect_ratio < self.min_aspect_ratio:
            return None

        # Add margin
        y_min = max(0, top_y - self.margin_pixels)
        y_max = min(h, bottom_y + self.margin_pixels)

        self.confidence = 0.8
        return (0, y_min, w, y_max)

    def _detect_profile(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect channel using horizontal intensity profile analysis.

        Strategy:
        1. Sum pixel intensities along each row
        2. Find stable plateau region (channel)
        3. Channel = region with consistent high/low activity

        Most robust to blur and artifacts.
        """
        h, w = gray.shape

        # Create horizontal profile (sum each row)
        profile = np.sum(gray, axis=1) / w

        # Smooth profile
        kernel_size = max(3, h // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        profile_smooth = cv2.GaussianBlur(profile.reshape(-1, 1), (kernel_size, 1), 0).flatten()

        # Find region with variance (channel has content, background doesn't)
        window_size = h // 10
        variance_profile = []
        for i in range(h):
            start = max(0, i - window_size // 2)
            end = min(h, i + window_size // 2)
            variance_profile.append(np.var(profile_smooth[start:end]))

        variance_profile = np.array(variance_profile)

        # Channel region has high variance (changing content)
        threshold = np.percentile(variance_profile, 50)
        channel_mask = variance_profile > threshold

        # Find largest contiguous region
        num_labels, labels = cv2.connectedComponents(channel_mask.astype(np.uint8))

        if num_labels < 2:
            return None

        # Find largest component (excluding background)
        largest_label = 0
        largest_size = 0
        for label in range(1, num_labels):
            size = np.sum(labels == label)
            if size > largest_size:
                largest_size = size
                largest_label = label

        # Get Y bounds
        y_indices = np.where(labels == largest_label)[0]
        if len(y_indices) == 0:
            return None

        y_min = max(0, y_indices[0] - self.margin_pixels)
        y_max = min(h, y_indices[-1] + self.margin_pixels)

        channel_height = y_max - y_min

        # Validate
        if channel_height < h * 0.1:
            return None

        area_ratio = channel_height / h
        if area_ratio < 0.2:
            return None

        self.confidence = 0.7
        return (0, y_min, w, y_max)

    def get_roi_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create binary mask for region of interest (channel).

        Args:
            image_shape: (height, width) of image

        Returns:
            Binary mask (255 = inside channel, 0 = outside)
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if self.channel_bounds:
            x_min, y_min, x_max, y_max = self.channel_bounds
            mask[y_min:y_max, x_min:x_max] = 255
        else:
            # No channel detected, use full image
            mask[:, :] = 255

        return mask

    def get_info(self) -> Dict:
        """Get detection info for debugging."""
        if self.channel_bounds is None:
            return {
                'detected': False,
                'method': None,
                'confidence': 0.0
            }

        x_min, y_min, x_max, y_max = self.channel_bounds
        width = x_max - x_min
        height = y_max - y_min

        return {
            'detected': True,
            'method': self.detection_method,
            'confidence': self.confidence,
            'bounds': self.channel_bounds,
            'width': width,
            'height': height,
            'aspect_ratio': width / height if height > 0 else 0
        }
