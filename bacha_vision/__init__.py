"""
bacha-vision: Microfluidic droplet detection and flow regime classification.
"""

from bacha_vision.detection.droplet_detector import ParticleImageProcessor
from bacha_vision.detection.channel_detector import ChannelDetector
from bacha_vision.detection.encapsulation_detector import EncapsulationDetector
from bacha_vision.classification.regime_detector import RegimeDetector, FlowRegime

__all__ = [
    "ParticleImageProcessor",
    "ChannelDetector",
    "EncapsulationDetector",
    "RegimeDetector",
    "FlowRegime",
]
