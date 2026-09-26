"""Evaluation harness shared by every pipeline: metric, bootstrap, experiment registry."""
from .bootstrap import bootstrap_mean, paired_delta  # noqa: F401
from .metrics import (best_threshold, entity_f05, entity_table, error_type, loss_decomposition,  # noqa: F401
                      summary, threshold_curve)
from .registry import Registry, read_json, write_json  # noqa: F401
