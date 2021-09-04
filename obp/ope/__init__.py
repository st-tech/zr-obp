from obp.ope.estimators import BaseOffPolicyEstimator
from obp.ope.estimators import DirectMethod
from obp.ope.estimators import DoublyRobust
from obp.ope.estimators import DoublyRobustWithShrinkage
from obp.ope.estimators import InverseProbabilityWeighting
from obp.ope.estimators import ReplayMethod
from obp.ope.estimators import SelfNormalizedDoublyRobust
from obp.ope.estimators import SelfNormalizedInverseProbabilityWeighting
from obp.ope.estimators import SwitchDoublyRobust
from obp.ope.estimators_continuous import (
    KernelizedSelfNormalizedInverseProbabilityWeighting,
)
from obp.ope.estimators_continuous import BaseContinuousOffPolicyEstimator
from obp.ope.estimators_continuous import cosine_kernel
from obp.ope.estimators_continuous import epanechnikov_kernel
from obp.ope.estimators_continuous import gaussian_kernel
from obp.ope.estimators_continuous import KernelizedDoublyRobust
from obp.ope.estimators_continuous import KernelizedInverseProbabilityWeighting
from obp.ope.estimators_continuous import triangular_kernel
from obp.ope.estimators_slate import SlateIndependentIPS
from obp.ope.estimators_slate import SlateRewardInteractionIPS
from obp.ope.estimators_slate import SlateStandardIPS
from obp.ope.estimators_tuning import DoublyRobustTuning
from obp.ope.estimators_tuning import DoublyRobustWithShrinkageTuning
from obp.ope.estimators_tuning import InverseProbabilityWeightingTuning
from obp.ope.estimators_tuning import SwitchDoublyRobustTuning
from obp.ope.meta import OffPolicyEvaluation
from obp.ope.meta_continuous import ContinuousOffPolicyEvaluation
from obp.ope.meta_slate import SlateOffPolicyEvaluation
from obp.ope.regression_model import RegressionModel


__all__ = [
    "BaseOffPolicyEstimator",
    "ReplayMethod",
    "InverseProbabilityWeighting",
    "SelfNormalizedInverseProbabilityWeighting",
    "DirectMethod",
    "DoublyRobust",
    "SelfNormalizedDoublyRobust",
    "SwitchDoublyRobust",
    "DoublyRobustWithShrinkage",
    "InverseProbabilityWeightingTuning",
    "DoublyRobustTuning",
    "SwitchDoublyRobustTuning",
    "DoublyRobustWithShrinkageTuning",
    "OffPolicyEvaluation",
    "SlateOffPolicyEvaluation",
    "ContinuousOffPolicyEvaluation",
    "RegressionModel",
    "SlateStandardIPS",
    "SlateIndependentIPS",
    "SlateRewardInteractionIPS",
    "BaseContinuousOffPolicyEstimator",
    "KernelizedInverseProbabilityWeighting",
    "KernelizedSelfNormalizedInverseProbabilityWeighting",
    "KernelizedDoublyRobust",
    "triangular_kernel",
    "gaussian_kernel",
    "epanechnikov_kernel",
    "cosine_kernel",
]

__all_estimators__ = [
    "ReplayMethod",
    "InverseProbabilityWeighting",
    "SelfNormalizedInverseProbabilityWeighting",
    "DirectMethod",
    "DoublyRobust",
    "DoublyRobustWithShrinkage",
    "SwitchDoublyRobust",
    "SelfNormalizedDoublyRobust",
]


__all_estimators_tuning__ = [
    "InverseProbabilityWeightingTuning",
    "DoublyRobustTuning",
    "SwitchDoublyRobustTuning",
    "DoublyRobustWithShrinkageTuning",
]
