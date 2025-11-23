"""
Perception Layer for Auto-Fluidics Control

This module contains computer vision and detection components for
analyzing microfluidic experiments in real-time.
"""

from .regime_detector import RegimeDetector, FlowRegime, regime_to_string, is_safe_regime

__all__ = [
    'RegimeDetector',
    'FlowRegime',
    'regime_to_string',
    'is_safe_regime'
]
