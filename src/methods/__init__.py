"""
CoT Vector methods package.

Available methods based on Variational CoT Vectors framework:

- ExtractedCoTVector: Extract vectors from activation differences (Eq. 4-5)
  Statistical aggregation to approximate the posterior distribution.

- LearnableCoTVector: Learn vectors via teacher-student framework (Eq. 6)
  Gradient optimization for learning global reasoning patterns.

- UACoTVector: Uncertainty-Aware vectors with Bayesian shrinkage
  MAP estimation with structured prior for adaptive gating.
"""

from .base import BaseCoTVectorMethod
from .extracted import ExtractedCoTVector
from .learnable import LearnableCoTVector
from .ua_vector import UACoTVector

__all__ = [
    "BaseCoTVectorMethod",
    "ExtractedCoTVector",
    "LearnableCoTVector",
    "UACoTVector",
]
