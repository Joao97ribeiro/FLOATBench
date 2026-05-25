"""
Initializes the split module.
"""

from .selectors import select_train_ids, split_train_test, split_test_groups
from .by_ids import (
    DEFAULT_TRAIN_WS_IDS,
    DEFAULT_TRAIN_HS_IDS,
    DEFAULT_TRAIN_TP_IDS,
    is_train_mask,
    split_train_test_by_ids,
    split_with_regimes,
)
