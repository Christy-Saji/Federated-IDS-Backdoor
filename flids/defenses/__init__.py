"""Phase 2 detection baselines (M2 track).

Neural Cleanse and Activation Clustering, reimplemented to match their papers
and - crucially - calibrated on clean models rather than trusting a paper's
default threshold.
"""

from .activation_clustering import (activation_clustering, calibrate_ac_threshold,
                                    exclusionary_reclassification)
from .neural_cleanse import nc_anomaly_index, neural_cleanse, reverse_engineer

__all__ = [
    "activation_clustering", "calibrate_ac_threshold",
    "exclusionary_reclassification",
    "neural_cleanse", "reverse_engineer", "nc_anomaly_index",
]
