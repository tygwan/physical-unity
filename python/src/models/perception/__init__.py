"""
Perception Module - Simplified perception for Planning focus

Components:
- GroundTruthExtractor: Extract GT from simulation
- PretrainedWrapper: Wrap pre-trained detection models
- BEVEncoder: Bird's Eye View encoding
"""

from .interface import DetectedObject, PerceptionOutput

__all__ = ["DetectedObject", "PerceptionOutput"]
