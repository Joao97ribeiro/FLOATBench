"""Design-oriented leaderboard metrics.

Two quantities that matter for a fatigue check and that pointwise error
metrics do not show:

* **Non-conservative share**: fraction of predictions that under-predict
  damage by more than a tolerance (default 10 %), restricted to a mask
  (by default extrapolated conditions at wind speeds where damage is
  large).
* **Lifetime-weighted damage error**: per tower section, the lifetime
  damage (occurrence-weighted sum over the operating envelope) computed
  with the predicted test damage, relative to the true lifetime damage.
  Training rows enter with their known labels, so the error measures the
  surrogate on the conditions it replaces.
"""

import numpy as np
import pandas as pd


def nonconservative_share(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          mask: np.ndarray | None = None,
                          tol: float = 0.1) -> float:
    """Fraction of rows with ``y_pred < (1 - tol) * y_true``.

    Args:
        y_true: True damage per row.
        y_pred: Predicted damage per row.
        mask: Optional boolean mask of the rows to evaluate.
        tol: Relative under-prediction tolerance.

    Returns:
        The share in [0, 1], or NaN if the mask selects no row.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_pred < (1.0 - tol) * y_true))


def lifetime_damage(damage: np.ndarray, weights: np.ndarray,
                    sections: np.ndarray) -> pd.Series:
    """Occurrence-weighted damage summed per section.

    Args:
        damage: Damage per row.
        weights: Occurrence weight per row (``damage_weight``).
        sections: Section identifier per row.

    Returns:
        Series of lifetime damage indexed by section.
    """
    prod = np.asarray(damage, float) * np.asarray(weights, float)
    return pd.Series(prod).groupby(np.asarray(sections)).sum()


def lifetime_error(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   weights: np.ndarray,
                   sections: np.ndarray,
                   train_lifetime: pd.Series | None = None) -> dict:
    """Relative lifetime-damage error per section, summarised.

    Args:
        y_true: True test damage per row.
        y_pred: Predicted test damage per row (negative values are
            clipped to zero).
        weights: Occurrence weight per test row.
        sections: Section identifier per test row.
        train_lifetime: Optional lifetime damage of the training rows per
            section (known labels); added to both true and predicted.

    Returns:
        Dict with ``lifetime_err_crit`` (signed error at the section with
        the largest true lifetime damage, the critical section),
        ``lifetime_err_maxabs`` (largest absolute error over sections)
        and ``lifetime_crit_section``.
    """
    life_true = lifetime_damage(y_true, weights, sections)
    life_hat = lifetime_damage(np.clip(np.asarray(y_pred, float), 0, None),
                               weights, sections)
    if train_lifetime is not None:
        base = train_lifetime.reindex(life_true.index, fill_value=0.0)
        life_true = life_true + base
        life_hat = life_hat + base
    err = (life_hat - life_true) / life_true
    crit = life_true.idxmax()
    return {
        "lifetime_err_crit": float(err[crit]),
        "lifetime_err_maxabs": float(err.abs().max()),
        "lifetime_crit_section": crit,
    }
