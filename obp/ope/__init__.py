from obp.ope.classification_model import ImportanceWeightEstimator
from obp.ope.classification_model import PropensityScoreEstimator
from obp.ope.estimators import BalancedInverseProbabilityWeighting
from obp.ope.estimators import BaseOffPolicyEstimator
from obp.ope.estimators import DirectMethod
from obp.ope.estimators import DoublyRobust
from obp.ope.estimators import DoublyRobustWithShrinkage
from obp.ope.estimators import InverseProbabilityWeighting
from obp.ope.estimators import ReplayMethod
from obp.ope.estimators import SelfNormalizedDoublyRobust
from obp.ope.estimators import SelfNormalizedInverseProbabilityWeighting
from obp.ope.estimators import SubGaussianDoublyRobust
from obp.ope.estimators import SubGaussianInverseProbabilityWeighting
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
from obp.ope.estimators_embed import (
    SelfNormalizedMarginalizedInverseProbabilityWeighting,
)
from obp.ope.estimators_embed import MarginalizedInverseProbabilityWeighting
from obp.ope.estimators_multi import BaseMultiLoggersOffPolicyEstimator
from obp.ope.estimators_multi import MultiLoggersBalancedDoublyRobust
from obp.ope.estimators_multi import MultiLoggersBalancedInverseProbabilityWeighting
from obp.ope.estimators_multi import MultiLoggersNaiveDoublyRobust
from obp.ope.estimators_multi import MultiLoggersNaiveInverseProbabilityWeighting
from obp.ope.estimators_multi import MultiLoggersWeightedDoublyRobust
from obp.ope.estimators_multi import MultiLoggersWeightedInverseProbabilityWeighting
from obp.ope.estimators_slate import SelfNormalizedSlateIndependentIPS
from obp.ope.estimators_slate import SelfNormalizedSlateRewardInteractionIPS
from obp.ope.estimators_slate import SelfNormalizedSlateStandardIPS
from obp.ope.estimators_slate import SlateCascadeDoublyRobust
from obp.ope.estimators_slate import SlateIndependentIPS
from obp.ope.estimators_slate import SlateRewardInteractionIPS
from obp.ope.estimators_slate import SlateStandardIPS
from obp.ope.estimators_tuning import DoublyRobustTuning
from obp.ope.estimators_tuning import DoublyRobustWithShrinkageTuning
from obp.ope.estimators_tuning import InverseProbabilityWeightingTuning
from obp.ope.estimators_tuning import SubGaussianDoublyRobustTuning
from obp.ope.estimators_tuning import SubGaussianInverseProbabilityWeightingTuning
from obp.ope.estimators_tuning import SwitchDoublyRobustTuning
from obp.ope.meta import OffPolicyEvaluation
from obp.ope.meta_continuous import ContinuousOffPolicyEvaluation
from obp.ope.meta_multi import MultiLoggersOffPolicyEvaluation
from obp.ope.meta_slate import SlateOffPolicyEvaluation
from obp.ope.regression_model import RegressionModel
from obp.ope.regression_model_slate import SlateRegressionModel


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
    "SubGaussianInverseProbabilityWeighting",
    "SubGaussianDoublyRobust",
    "InverseProbabilityWeightingTuning",
    "DoublyRobustTuning",
    "SwitchDoublyRobustTuning",
    "DoublyRobustWithShrinkageTuning",
    "SubGaussianInverseProbabilityWeightingTuning",
    "SubGaussianDoublyRobustTuning",
    "MarginalizedInverseProbabilityWeighting",
    "SelfNormalizedMarginalizedInverseProbabilityWeighting",
    "BaseMultiLoggersOffPolicyEstimator",
    "MultiLoggersNaiveInverseProbabilityWeighting",
    "MultiLoggersWeightedInverseProbabilityWeighting",
    "MultiLoggersBalancedInverseProbabilityWeighting",
    "MultiLoggersNaiveDoublyRobust",
    "MultiLoggersBalancedDoublyRobust",
    "MultiLoggersWeightedDoublyRobust",
    "OffPolicyEvaluation",
    "SlateOffPolicyEvaluation",
    "ContinuousOffPolicyEvaluation",
    "MultiLoggersOffPolicyEvaluation",
    "RegressionModel",
    "SlateRegressionModel",
    "SlateStandardIPS",
    "SlateIndependentIPS",
    "SlateRewardInteractionIPS",
    "SlateCascadeDoublyRobust",
    "SelfNormalizedSlateRewardInteractionIPS",
    "SelfNormalizedSlateIndependentIPS",
    "SelfNormalizedSlateStandardIPS",
    "BalancedInverseProbabilityWeighting",
    "ImportanceWeightEstimator",
    "PropensityScoreEstimator",
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
    "SubGaussianInverseProbabilityWeighting",
    "SubGaussianDoublyRobust",
    "BalancedInverseProbabilityWeighting",
]


__all_estimators_tuning__ = [
    "InverseProbabilityWeightingTuning",
    "DoublyRobustTuning",
    "SwitchDoublyRobustTuning",
    "DoublyRobustWithShrinkageTuning",
]


__all_estimators_tuning_sg__ = [
    "SubGaussianInverseProbabilityWeightingTuning",
    "SubGaussianDoublyRobustTuning",
]
