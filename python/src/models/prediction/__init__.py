"""
Prediction Module - Trajectory prediction for Planning

Components:
- ConstantVelocity: Simple CV baseline
- NuPlanBaseline: Wrapper for nuPlan baseline predictors
"""

from .interface import PredictedTrajectory, PredictionOutput

__all__ = ["PredictedTrajectory", "PredictionOutput"]
