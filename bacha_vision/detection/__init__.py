"""
Detection subpackage: droplet detection, channel extraction, encapsulation analysis.
"""

from bacha_vision.detection.droplet_detector import ParticleImageProcessor
from bacha_vision.detection.channel_detector import ChannelDetector
from bacha_vision.detection.encapsulation_detector import EncapsulationDetector

__all__ = [
    "ParticleImageProcessor",
    "ChannelDetector",
    "EncapsulationDetector",
]
