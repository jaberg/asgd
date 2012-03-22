"""
BinaryASGD - this symbol refers to the fastest available implementation of the
BinaryASGD algorithm, as defined in naive_asgd.

OVAASGD - this symbol refers to the fastest available implementation of the
OVAASGD algorithm, as defined in naive_asgd.
"""

# naive_asgd defines reference implementations

from naive_asgd import NaiveBinaryASGD
from naive_asgd import NaiveOVAASGD
from naive_asgd import NaiveRankASGD
from naive_asgd import SparseUpdateRankASGD

from experimental_asgd import ExperimentalBinaryASGD

from linsvm import LinearSVM
