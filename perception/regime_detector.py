"""
Regime Detection Module for Microfluidics Control

This module classifies flow regimes (NO_FLOW, DRIPPING, JETTING, CO_FLOW) based on
particle detection metrics. This is critical for safe parameter space exploration
and control loop operation.

Flow Regimes:
- NO_FLOW: Insufficient pressure, no droplets detected
- DRIPPING: Stable discrete droplet formation (DESIRED for control)
- JETTING: Continuous unstable stream (AVOID - damages experiments)
- CO_FLOW: Co-flowing streams without droplet formation

Author: Auto-Fluidics Project
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Tuple
from collections import deque


class FlowRegime(Enum):
    """Flow regime classifications"""
    NO_FLOW = 0
    DRIPPING = 1
    JETTING = 2
    CO_FLOW = 3
    UNKNOWN = 4


class RegimeDetector:
    """
    Detects and classifies microfluidic flow regimes based on particle metrics.

    Uses multiple indicators:
    - Aspect ratio (elongated particles indicate jetting)
    - Size variability (CV - high variance indicates instability)
    - Count stability (temporal variance in particle count)
    - Particle count (no particles = no flow)
    """

    def __init__(self,
                 aspect_ratio_threshold: float = 2.5,
                 size_cv_threshold: float = 0.3,
                 count_cv_threshold: float = 0.5,
                 min_particles: int = 3,
                 history_length: int = 30):
        """
        Initialize regime detector with tunable thresholds.

        Args:
            aspect_ratio_threshold: Mean aspect ratio above this indicates jetting (default 2.5)
            size_cv_threshold: Size CV above this indicates instability (default 0.3)
            count_cv_threshold: Count CV above this indicates irregular generation (default 0.5)
            min_particles: Minimum particles to classify as flowing (default 3)
            history_length: Number of frames to keep for temporal analysis (default 30)
        """
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.size_cv_threshold = size_cv_threshold
        self.count_cv_threshold = count_cv_threshold
        self.min_particles = min_particles

        # Temporal tracking
        self.history_length = history_length
        self.size_history = deque(maxlen=history_length)
        self.count_history = deque(maxlen=history_length)
        self.aspect_ratio_history = deque(maxlen=history_length)
        self.regime_history = deque(maxlen=history_length)

    def detect_regime(self, particle_metrics: Dict) -> Tuple[FlowRegime, float, Dict]:
        """
        Detect flow regime from particle metrics.

        Args:
            particle_metrics: Dictionary containing particle detection metrics
                Must include: num_particles, mean_aspect_ratio, coefficient_of_variation

        Returns:
            Tuple of (regime, confidence, indicators)
                regime: FlowRegime enum value
                confidence: Confidence score 0-1
                indicators: Dict of regime indicators used for classification
        """
        # Extract metrics
        num_particles = particle_metrics.get('num_particles', 0)
        mean_aspect_ratio = particle_metrics.get('mean_aspect_ratio', 1.0)
        size_cv = particle_metrics.get('coefficient_of_variation', 0.0)
        mean_size = particle_metrics.get('mean_particle_size', 0)

        # Update history
        if num_particles > 0:
            self.size_history.append(mean_size)
            self.aspect_ratio_history.append(mean_aspect_ratio)
        self.count_history.append(num_particles)

        # Calculate temporal metrics
        temporal_size_cv = self._calculate_temporal_cv(self.size_history)
        temporal_count_cv = self._calculate_temporal_cv(self.count_history)

        # Regime indicators
        indicators = {
            'num_particles': num_particles,
            'mean_aspect_ratio': mean_aspect_ratio,
            'spatial_size_cv': size_cv,
            'temporal_size_cv': temporal_size_cv,
            'temporal_count_cv': temporal_count_cv,
            'aspect_ratio_threshold': self.aspect_ratio_threshold,
            'size_cv_threshold': self.size_cv_threshold,
            'count_cv_threshold': self.count_cv_threshold
        }

        # Classification logic
        regime, confidence = self._classify_regime(indicators)

        # Store in history
        self.regime_history.append(regime)

        return regime, confidence, indicators

    def _classify_regime(self, indicators: Dict) -> Tuple[FlowRegime, float]:
        """
        Classify regime based on indicators.

        Returns:
            Tuple of (regime, confidence)
        """
        num_particles = indicators['num_particles']
        mean_aspect_ratio = indicators['mean_aspect_ratio']
        spatial_cv = indicators['spatial_size_cv']
        temporal_size_cv = indicators['temporal_size_cv']
        temporal_count_cv = indicators['temporal_count_cv']

        # NO_FLOW: Insufficient particles
        if num_particles < self.min_particles:
            return FlowRegime.NO_FLOW, 1.0

        # JETTING indicators (any one is sufficient)
        jetting_indicators = []

        # 1. High aspect ratio (elongated particles/jets)
        if mean_aspect_ratio > self.aspect_ratio_threshold:
            jetting_indicators.append('high_aspect_ratio')

        # 2. High spatial size variability
        if spatial_cv > self.size_cv_threshold:
            jetting_indicators.append('high_spatial_cv')

        # 3. High temporal size variability (if we have enough history)
        if len(self.size_history) >= 10 and temporal_size_cv > self.size_cv_threshold:
            jetting_indicators.append('high_temporal_cv')

        # 4. High temporal count variability
        if len(self.count_history) >= 10 and temporal_count_cv > self.count_cv_threshold:
            jetting_indicators.append('irregular_generation')

        # If any jetting indicators present, classify as JETTING
        if len(jetting_indicators) > 0:
            # Confidence based on number of indicators
            confidence = min(1.0, len(jetting_indicators) * 0.4)
            return FlowRegime.JETTING, confidence

        # DRIPPING: Sufficient particles, low variance, circular shapes
        # Confidence based on how far we are from thresholds
        aspect_margin = (self.aspect_ratio_threshold - mean_aspect_ratio) / self.aspect_ratio_threshold
        cv_margin = (self.size_cv_threshold - spatial_cv) / self.size_cv_threshold

        confidence = min(1.0, (aspect_margin + cv_margin) / 2.0)
        confidence = max(0.5, confidence)  # At least 0.5 if we reached here

        return FlowRegime.DRIPPING, confidence

    def _calculate_temporal_cv(self, history: deque) -> float:
        """
        Calculate coefficient of variation from temporal history.

        Args:
            history: Deque of historical values

        Returns:
            CV (std/mean) or 0.0 if insufficient data
        """
        if len(history) < 3:
            return 0.0

        values = np.array(list(history))
        mean = np.mean(values)

        if mean == 0:
            return 0.0

        std = np.std(values)
        return std / mean

    def get_regime_stability(self, window_size: int = 10) -> float:
        """
        Calculate regime stability over recent history.

        Args:
            window_size: Number of recent frames to consider

        Returns:
            Stability score 0-1 (1 = perfectly stable, same regime)
        """
        if len(self.regime_history) < window_size:
            return 0.5  # Insufficient data

        recent = list(self.regime_history)[-window_size:]
        most_common = max(set(recent), key=recent.count)
        stability = recent.count(most_common) / len(recent)

        return stability

    def reset_history(self):
        """Clear all temporal history."""
        self.size_history.clear()
        self.count_history.clear()
        self.aspect_ratio_history.clear()
        self.regime_history.clear()


def regime_to_string(regime: FlowRegime) -> str:
    """Convert FlowRegime enum to human-readable string."""
    return regime.name.replace('_', ' ')


def is_safe_regime(regime: FlowRegime) -> bool:
    """Check if regime is safe for control loop operation."""
    return regime == FlowRegime.DRIPPING
