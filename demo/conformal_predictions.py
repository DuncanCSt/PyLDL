"""Split conformal prediction (SCP) for label-distribution-learning models.

Pipeline (one call to :func:`fsc_score`):

1. Run the *fitted* model on the calibration set to get point predictions.
2. Compute a per-sample non-conformity score
   ``s_i = max_k |D_cal_{i,k} - D_pred_cal_{i,k}|``.
3. Take the conformal quantile ``q̂`` of those scores at the level
   ``ceil((n_cal + 1) · confidence) / n_cal`` (the +1-correction that
   guarantees marginal coverage ≥ ``confidence`` under exchangeability).
4. Construct symmetric ± ``q̂`` intervals around the model's predictions
   on the test set: ``[D_pred_test − q̂, D_pred_test + q̂]``.
5. Compute coverage per (sample, label), aggregate to:
   - **Per-label FSC** — coverage rate for each label index.
   - **Per-bin FSC** — coverage in equal-frequency bins of
     ``D_pred_test[:, k]`` per label.
   - **Worst-bin FSC** — the minimum across all (label, bin) cells; the
     headline conditional-coverage figure.

Designed to be model-agnostic. Any PyLDL neural-net model works
(``BEDL_LDL``, ``EDL_LDL``, ``SNEFY_LDL``, ``Duo_LDL``, ``AA_BP``); the
helper :func:`_predict_distribution` strips the variance/uncertainty
components from models whose ``predict`` returns a tuple.
"""
from typing import Dict

import numpy as np


def _predict_distribution(model, X) -> np.ndarray:
    """Return just the mean label distribution from ``model.predict(X)``.

    BEDL_LDL's ``predict`` defaults to ``return_uncertainty=True`` and
    returns a 3-tuple; EDL_LDL / SNEFY_LDL likewise when their flag is
    set. We unwrap to keep the conformal pipeline uniform.
    """
    out = model.predict(X, return_uncertainty=True)
    if isinstance(out, tuple):
        pred = out[0]
        var = out[1]
    return np.asarray(pred), np.asarray(var)


def fsc_score(model, X_cal, D_cal, X_test, D_test,
              confidence: float = 0.9, bin_count: int = 10) -> Dict:
    """Split conformal prediction on a fitted LDL model.

    Parameters
    ----------
    model
        A fitted PyLDL model (designed with ``BEDL_LDL`` in mind, but works
        for any model whose ``predict(X)`` returns an ``(n, K)`` array — or
        a tuple whose first element is that array).
    X_cal, D_cal
        Calibration features and *true* label distributions. Must NOT
        overlap with the model's training set.
    X_test, D_test
        Held-out test features and true label distributions.
    confidence
        Target marginal coverage in ``(0, 1)``. E.g. ``0.9`` requests the
        true distribution to fall inside the predicted interval at least
        90% of the time on average.
    bin_count
        Number of equal-frequency bins per label, used for the
        worst-bin FSC report. Higher = stricter conditional-coverage
        check, but bins get smaller and noisier.

    Returns
    -------
    dict
        ``q_hat``         : float — calibration quantile (interval half-width).
        ``D_pred_test``   : ndarray (N_test, K) — point predictions.
        ``lower``, ``upper`` : ndarray (N_test, K) — interval endpoints.
        ``covered``       : bool ndarray (N_test, K) — per-(sample, label) coverage.
        ``per_label_fsc`` : ndarray (K,) — coverage rate per label.
        ``per_bin_fsc``   : ndarray (K, bin_count) — coverage in each
            (label, bin); NaN where a bin is empty.
        ``worst_bin_fsc`` : float — minimum non-NaN value in
            ``per_bin_fsc`` (the worst-case conditional coverage).
        ``joint_fsc``     : float — fraction of test samples whose entire
            true distribution fell inside the predicted interval (all K
            labels simultaneously).
    """
    if not 0. < confidence < 1.:
        raise ValueError(f'confidence must be in (0, 1), got {confidence!r}')
    if bin_count < 1:
        raise ValueError(f'bin_count must be >= 1, got {bin_count!r}')

    D_cal  = np.asarray(D_cal,  dtype=float)
    D_test = np.asarray(D_test, dtype=float)

    # Confidence = 1-1/k^2
    k = np.ceil(1 / np.sqrt(1 - confidence))

    # ------------------------------------------------------------------ #
    # 1. Calibration scores — max absolute per-label deviation per sample.
    # ------------------------------------------------------------------ #
    D_pred_cal, var_cal = _predict_distribution(model, X_cal)
    cal_scores = np.abs(D_cal - D_pred_cal)/(k*np.sqrt(var_cal))         # (N_cal, K)

    # ------------------------------------------------------------------ #
    # 2. Conformal quantile q̂. The +1 correction is what gives the
    #    finite-sample coverage guarantee. For very small calibration
    #    sets the corrected level can exceed 1.0 — we cap it.
    # ------------------------------------------------------------------ #
    n_cal   = len(cal_scores)
    q_level = min(1.0, ((n_cal + 1) * confidence) / n_cal)
    q_hat = np.quantile(cal_scores, q_level, axis=0, method='higher')   # (K,)

    # ------------------------------------------------------------------ #
    # 3. Test predictions and ± q̂ intervals.
    # ------------------------------------------------------------------ #
    D_pred_test, var_test = _predict_distribution(model, X_test)              # (N_test, K)
    lower = D_pred_test - k*q_hat*np.sqrt(var_test)
    upper = D_pred_test + k*q_hat*np.sqrt(var_test)

    # ------------------------------------------------------------------ #
    # 4. Coverage per (sample, label) and aggregates.
    # ------------------------------------------------------------------ #
    covered = (D_test >= lower) & (D_test <= upper)                  # (N_test, K)
    per_label_fsc = covered.mean(axis=0)                             # (K,)
    joint_fsc     = float(covered.all(axis=1).mean())

    # ------------------------------------------------------------------ #
    # 5. Per-(label, bin) FSC. For each label k, partition the test
    #    samples into `bin_count` equal-frequency bins on D_pred_test[:, k]
    #    and compute coverage within each bin. The worst-bin FSC is the
    #    minimum across all (k, bin) cells.
    # ------------------------------------------------------------------ #
    K = D_test.shape[1]
    per_bin_fsc = np.full((K, bin_count), np.nan, dtype=float)
    for k in range(K):
        edges = np.quantile(D_pred_test[:, k],
                            np.linspace(0., 1., bin_count + 1))
        # np.digitize returns indices in [1, bin_count] for interior edges
        # — clipping puts boundary cases into the outer bins.
        bin_idx = np.clip(np.digitize(D_pred_test[:, k], edges[1:-1]),
                          0, bin_count - 1)
        for b in range(bin_count):
            mask = bin_idx == b
            if mask.any():
                per_bin_fsc[k, b] = covered[mask, k].mean()

    worst_bin_fsc = float(np.nanmin(per_bin_fsc))

    return {
        'q_hat':         q_hat,
        'D_pred_test':   D_pred_test,
        'lower':         lower,
        'upper':         upper,
        'covered':       covered,
        'per_label_fsc': per_label_fsc,
        'per_bin_fsc':   per_bin_fsc,
        'worst_bin_fsc': worst_bin_fsc,
        'joint_fsc':     joint_fsc,
    }
