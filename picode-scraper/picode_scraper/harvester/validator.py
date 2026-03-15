"""Pair validation using SIFT feature matching."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from picode_scraper.config import ValidationConfig


@dataclass
class ValidationResult:
    """Result of pair validation.

    Attributes:
        valid: Whether the pair passed validation.
        reason: Reason for rejection (if invalid).
        corners: Detected corner coordinates [[x, y], ...].
        corner_confidence: Confidence in corner detection (0-1).
        coverage: Fraction of reference image covered by detected region.
        similarity: Visual similarity score (0-1).
        quality_score: Overall quality score combining all metrics.
    """

    valid: bool
    reason: str | None = None
    corners: list[list[float]] = field(default_factory=list)
    corner_confidence: float = 0.0
    coverage: float = 0.0
    similarity: float = 0.0
    quality_score: float = 0.0


class PairValidator:
    """Validates image pairs per data_scraper.md spec.

    Uses proxy resolution for initial SIFT matching to improve
    performance when checking many combinations, then validates at full
    resolution for final corner detection.
    """

    PROXY_SIZE = 800  # Max dimension for initial SIFT check

    def __init__(self, config: ValidationConfig) -> None:
        """Initialize validator with config.

        Args:
            config: Validation configuration with thresholds.
        """
        self.min_size = config.min_image_size
        self.min_confidence = config.min_corner_confidence
        self.min_coverage = config.min_coverage
        self.min_similarity = config.min_similarity
        self.sift = cv2.SIFT_create()  # type: ignore[attr-defined]
        self.matcher = cv2.BFMatcher()

    def validate(self, original: np.ndarray, capture: np.ndarray) -> ValidationResult:
        """Validate that capture contains original.

        Args:
            original: Original image (BGR, HWC)
            capture: Capture image that may contain original

        Returns:
            ValidationResult with validity and metrics
        """
        # 1. Size check
        if min(original.shape[:2]) < self.min_size:
            return ValidationResult(valid=False, reason="original too small")

        # 2. Quick check at proxy resolution first
        orig_proxy = self._to_proxy(original)
        cap_proxy = self._to_proxy(capture)

        corners_proxy, confidence = self._find_corners(orig_proxy, cap_proxy)
        if confidence < self.min_confidence:
            return ValidationResult(valid=False, reason="poor feature matching")

        # 3. Refine at full resolution
        corners, confidence = self._find_corners(original, capture)
        if confidence < self.min_confidence:
            return ValidationResult(valid=False, reason="poor feature matching at full res")

        # 4. Check coverage
        coverage = self._compute_coverage(corners, capture.shape)
        if coverage < self.min_coverage:
            return ValidationResult(valid=False, reason="original too small in capture")

        # 5. Perceptual similarity
        rectified = self._rectify(capture, corners)
        similarity = self._compute_similarity(original, rectified)
        if similarity < self.min_similarity:
            return ValidationResult(valid=False, reason="poor content match")

        quality_score = (confidence + coverage + similarity) / 3

        return ValidationResult(
            valid=True,
            corners=corners.tolist(),
            corner_confidence=confidence,
            coverage=coverage,
            similarity=similarity,
            quality_score=quality_score,
        )

    def _to_proxy(self, image: np.ndarray) -> np.ndarray:
        """Resize to proxy resolution for fast initial matching.

        Args:
            image: Input image (HWC)

        Returns:
            Resized image if larger than PROXY_SIZE, otherwise original
        """
        h, w = image.shape[:2]
        if max(h, w) <= self.PROXY_SIZE:
            return image
        scale = self.PROXY_SIZE / max(h, w)
        return cv2.resize(image, None, fx=scale, fy=scale)

    def _find_corners(
        self, original: np.ndarray, capture: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """SIFT + RANSAC homography estimation.

        Args:
            original: Original image to find
            capture: Capture image to search in

        Returns:
            Tuple of (corner coordinates as 4x2 array, inlier ratio confidence)
        """
        kp1, desc1 = self.sift.detectAndCompute(original, None)
        kp2, desc2 = self.sift.detectAndCompute(capture, None)

        if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
            return np.zeros((4, 2)), 0.0

        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Lowe's ratio test
        good = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 10:
            return np.zeros((4, 2)), 0.0

        src_pts = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float32)
        dst_pts = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float32)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return np.zeros((4, 2)), 0.0

        inlier_ratio = float(mask.sum()) / len(mask)

        h, w = original.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H)

        return transformed.reshape(4, 2), inlier_ratio

    def _compute_coverage(self, corners: np.ndarray, capture_shape: tuple) -> float:
        """Compute what fraction of capture the original occupies.

        Args:
            corners: Corner coordinates of detected region (4x2)
            capture_shape: Shape of capture image (H, W, C)

        Returns:
            Coverage ratio (0-1)
        """
        quad_area = cv2.contourArea(corners.astype(np.float32))
        capture_area = capture_shape[0] * capture_shape[1]
        return quad_area / capture_area if capture_area > 0 else 0.0

    def _rectify(self, capture: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Rectify capture to match original perspective.

        Args:
            capture: Capture image
            corners: Corner coordinates of detected region (4x2)

        Returns:
            Rectified image at standard comparison size
        """
        h, w = 200, 200  # Standard output size for comparison
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        return cv2.warpPerspective(capture, M, (w, h))

    def _compute_similarity(self, original: np.ndarray, rectified: np.ndarray) -> float:
        """Compute perceptual similarity between images.

        Args:
            original: Original image
            rectified: Rectified capture image

        Returns:
            Similarity score (0-1)
        """
        original_resized = cv2.resize(original, (rectified.shape[1], rectified.shape[0]))

        original_norm = original_resized.astype(np.float32) / 255.0
        rectified_norm = rectified.astype(np.float32) / 255.0

        # Flatten and compute correlation
        corr = np.corrcoef(original_norm.flatten(), rectified_norm.flatten())[0, 1]

        return max(0.0, float(corr)) if not np.isnan(corr) else 0.0
