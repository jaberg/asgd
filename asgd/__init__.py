"""
BinaryASGD - this symbol refers to the fastest available implementation of the
BinaryASGD algorithm, as defined in naive_asgd.

OVAASGD - this symbol refers to the fastest available implementation of the
OVAASGD algorithm, as defined in naive_asgd.
"""

# naive_asgd defines reference implementations

from .naive_asgd import BinaryASGD
from .naive_asgd import OneVsAllASGD
from .naive_asgd import NaiveRankASGD
from .naive_asgd import SparseUpdateRankASGD

from .base import BinaryClassifier
from .base import MultiClassifier
from .base import classifier

from .linsvm import LinearSVM
