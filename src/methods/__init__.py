"""
CoT Vector methods package.

Available methods based on Variational CoT Vectors framework:

- ExtractedCoTVector: Extract vectors from activation differences (Eq. 4-5)
  Statistical aggregation to approximate the posterior distribution.

- LearnableCoTVector: Learn vectors via teacher-student framework (Eq. 6)
  Gradient optimization for learning global reasoning patterns.

- UACoTVector: Uncertainty-Aware vectors with Bayesian shrinkage
  MAP estimation with structured prior for adaptive gating.

- ABCCoTVector: Adaptive Bayesian CoT Vector with variational inference
  Prior-posterior networks for dynamic, sample-specific vector injection.

- HierarchicalABCCoTVector: Two-layer hierarchical variational CoT Vector
  Global + instance latent variables with projection-based injection (no gate).
"""

from .base import BaseCoTVectorMethod
from .extracted import ExtractedCoTVector
from .learnable import LearnableCoTVector
from .ua_vector import UACoTVector
from .abc_vector import ABCCoTVector
from .habc_vector import HierarchicalABCCoTVector

__all__ = [
    "BaseCoTVectorMethod",
    "ExtractedCoTVector",
    "LearnableCoTVector",
    "UACoTVector",
    "ABCCoTVector",
    "HierarchicalABCCoTVector",
]