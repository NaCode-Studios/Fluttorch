"""VoltaCast's model and the feature pipeline that feeds it.

Vendored verbatim from https://github.com/tonytonycoder11/voltacast, MIT
licensed, copyright 2026 Antonio Sarno. The licence sits beside this directory.

Copied rather than depended on because that repository is private and this one
is public: an example nobody can reproduce is not an example. The only edits are
the import paths, which changed because voltacast nests these under two packages
and here they are one.
"""

from .features import FUTURE_FEATURES, PAST_FEATURES, build_features
from .forecaster import VoltaCastTransformer
from .heads import monotone_quantiles
from .scaling import Scaler

__all__ = [
    "FUTURE_FEATURES",
    "PAST_FEATURES",
    "Scaler",
    "VoltaCastTransformer",
    "build_features",
    "monotone_quantiles",
]
