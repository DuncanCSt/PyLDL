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


def marginal_calibration(model, X_test, D_test,
                         X_cal=None, D_cal=None, levels=None) -> Dict:
    r"""Marginal calibration check across all confidence levels.

    This reads the model's *own* predictive distribution. For each
    (sample, label) the method :meth:`coverage_level` returns the smallest
    central credible level whose interval already contains the true value.
    Empirical coverage at a target confidence ``gamma`` is then the fraction
    of those values ``<= gamma``. Sweeping ``gamma`` over a grid traces the
    full reliability curve.

    A perfectly calibrated model has empirical coverage equal to nominal
    confidence at *every* level (the reliability curve lies on the diagonal),
    equivalently the ``coverage_level`` values are uniform on ``[0, 1]``.

    **Recalibration (optional).** If a calibration set is supplied, the raw
    nominal levels are replaced by *conformal thresholds* learned from it: for
    target confidence ``gamma`` the threshold is the conformal quantile of the
    calibration ``coverage_level`` values at level ``(n_cal + 1)·gamma / n_cal``
    (the +1-correction giving finite-sample marginal coverage under
    exchangeability). Testing ``c_min_test <= threshold`` then yields coverage
    ``>= gamma`` even when the model's own credible levels are miscalibrated.
    Without a calibration set the threshold is just ``gamma`` itself, i.e. the
    model's raw, uncorrected calibration.

    Parameters
    ----------
    model
        A fitted PyLDL model exposing ``coverage_level(X, Y) -> (N, K)`` array
        (``EDL`` / ``EDL_BAYES`` / ``BEDL`` / ``BEDL_BAYES`` / ``BOOJUM`` /
        ``BOOJUM_BAYES``).
    X_test, D_test
        Held-out test features and true label distributions.
    X_cal, D_cal
        Optional calibration features and true label distributions, disjoint
        from both training and test sets. If either is ``None`` the check is
        run uncorrected. Used only to learn the conformal thresholds.
    levels
        1-D array of nominal confidence levels in ``[0, 1]`` at which to
        evaluate coverage. Defaults to ``np.linspace(0, 1, 21)``.

    Returns
    -------
    dict
        ``levels``             : ndarray (M,) — nominal confidence grid.
        ``calibrated``         : bool — whether conformal recalibration was applied.
        ``thresholds``         : ndarray (M,) — credible-level cut-off applied at
            each nominal level (equals ``levels`` when uncalibrated).
        ``coverage_level``     : ndarray (N, K) — per-(sample, label) minimal
            covering credible level on the test set.
        ``per_label_coverage`` : ndarray (M, K) — empirical coverage per label
            at each nominal level.
        ``marginal_coverage``  : ndarray (M,) — empirical coverage pooled over
            all (sample, label) pairs at each nominal level.
        ``joint_coverage``     : ndarray (M,) — fraction of test samples whose
            *all* K labels are covered simultaneously at each nominal level.
        ``per_label_ece``      : ndarray (K,) — mean ``|empirical - nominal|``
            over the grid, per label.
        ``ece``                : float — expected calibration error, the
            grid-averaged ``|marginal_coverage - levels|`` (headline figure).
        ``max_ce``             : float — worst (max) calibration gap over the
            grid; a stricter, worst-case calibration figure.
    """
    levels = (np.linspace(0., 1., 21) if levels is None
              else np.asarray(levels, dtype=float))
    if np.any((levels < 0.) | (levels > 1.)):
        raise ValueError('levels must all lie in [0, 1]')

    D_test = np.asarray(D_test, dtype=float)

    # ------------------------------------------------------------------ #
    # 1. Model-intrinsic coverage levels: c_min[i, k] is the smallest
    #    credible level whose interval already contains D_test[i, k].
    # ------------------------------------------------------------------ #
    c_min = np.asarray(model.coverage_level(X_test, D_test), dtype=float)  # (N, K)

    # ------------------------------------------------------------------ #
    # 2. Per-level thresholds. Uncalibrated: the threshold is the nominal
    #    level itself. Calibrated: the conformal quantile of the
    #    calibration coverage levels, with the +1 finite-sample correction.
    # ------------------------------------------------------------------ #
    calibrated = X_cal is not None and D_cal is not None
    if calibrated:
        cal_c = np.asarray(model.coverage_level(X_cal, np.asarray(D_cal, float)),
                            dtype=float).ravel()                      # (n_cal,)
        n_cal = cal_c.size
        q_levels = np.minimum(1.0, ((n_cal + 1) * levels) / n_cal)
        thresholds = np.quantile(cal_c, q_levels, method='higher')    # (M,)
    else:
        thresholds = levels

    # ------------------------------------------------------------------ #
    # 3. Empirical coverage at every nominal level. Broadcasting c_min
    #    (N, K, 1) against thresholds (M,) gives a (N, K, M) coverage mask.
    # ------------------------------------------------------------------ #
    covered = c_min[:, :, None] <= thresholds[None, None, :]          # (N, K, M)

    per_label_coverage = covered.mean(axis=0).T                       # (M, K)
    marginal_coverage  = covered.mean(axis=(0, 1))                    # (M,)
    joint_coverage     = covered.all(axis=1).mean(axis=0)             # (M,)

    # ------------------------------------------------------------------ #
    # 4. Calibration error: gap between empirical and nominal coverage.
    # ------------------------------------------------------------------ #
    per_label_ece = np.abs(per_label_coverage - levels[:, None]).mean(axis=0)  # (K,)
    gap           = np.abs(marginal_coverage - levels)                # (M,)
    ece           = float(gap.mean())
    max_ce        = float(gap.max())

    return {
        'levels':             levels,
        'calibrated':         calibrated,
        'thresholds':         thresholds,
        'coverage_level':     c_min,
        'per_label_coverage': per_label_coverage,
        'marginal_coverage':  marginal_coverage,
        'joint_coverage':     joint_coverage,
        'per_label_ece':      per_label_ece,
        'ece':                ece,
        'max_ce':             max_ce,
    }


def joint_calibration(model, X_test, D_test,
                      X_cal=None, D_cal=None, levels=None) -> Dict:
    r"""Joint calibration check across all confidence levels.

    The multivariate analogue of :func:`marginal_calibration`. The joint
    credible region is the box formed by the per-label central intervals, all
    at the same credible level ``c``. By the AND rule a sample is covered iff
    *every* label is covered, which happens iff ``c >= max_k c_min(i, k)``.
    The per-sample joint coverage level is therefore

        ``joint_c_min(i) = max_k coverage_level(i, k)``

    — the worst (widest-needing) label drags the joint region out. This 1-D
    score is then run through the same thresholding / conformal-recalibration
    machinery as :func:`marginal_calibration`.

    **Reference line.** Uncalibrated, empirical joint coverage at level ``c``
    sits *below* ``c`` (the box at per-label level ``c`` has joint mass below
    ``c``); that gap is the structural K-label effect, not miscalibration, so
    ``joint_ece`` is *not* meaningful uncalibrated. With a calibration set the
    conformal threshold forces empirical joint coverage onto the nominal
    level, so ``joint_ece`` (calibrated) *is* a valid joint calibration error.

    Parameters
    ----------
    model
        A fitted PyLDL model exposing ``coverage_level(X, Y) -> (N, K)`` array.
    X_test, D_test
        Held-out test features and true label distributions.
    X_cal, D_cal
        Optional calibration features and true label distributions, disjoint
        from training and test. If either is ``None`` the check is uncorrected.
    levels
        1-D array of nominal confidence levels in ``[0, 1]``. Defaults to
        ``np.linspace(0, 1, 21)``.

    Returns
    -------
    dict
        ``levels``         : ndarray (M,) — nominal confidence grid.
        ``calibrated``     : bool — whether conformal recalibration was applied.
        ``thresholds``     : ndarray (M,) — joint credible-level cut-off per
            nominal level (equals ``levels`` when uncalibrated).
        ``joint_c_min``    : ndarray (N,) — per-sample joint coverage level
            ``max_k coverage_level(i, k)``.
        ``joint_coverage`` : ndarray (M,) — empirical joint coverage (all K
            labels covered simultaneously) at each nominal level.
        ``joint_ece``      : float — grid-averaged ``|joint_coverage - levels|``
            (a valid joint calibration error only when ``calibrated`` is True).
        ``joint_max_ce``   : float — worst joint calibration gap over the grid.
    """
    levels = (np.linspace(0., 1., 21) if levels is None
              else np.asarray(levels, dtype=float))
    if np.any((levels < 0.) | (levels > 1.)):
        raise ValueError('levels must all lie in [0, 1]')

    D_test = np.asarray(D_test, dtype=float)

    # ------------------------------------------------------------------ #
    # 1. Per-sample joint coverage level: the worst per-label coverage
    #    level — the smallest box-level whose region covers every label.
    # ------------------------------------------------------------------ #
    c_min = np.asarray(model.coverage_level(X_test, D_test), dtype=float)  # (N, K)
    joint_c_min = c_min.max(axis=1)                                       # (N,)

    # ------------------------------------------------------------------ #
    # 2. Per-level thresholds — nominal level itself, or the conformal
    #    quantile of the calibration joint coverage levels.
    # ------------------------------------------------------------------ #
    calibrated = X_cal is not None and D_cal is not None
    if calibrated:
        cal_c = np.asarray(model.coverage_level(X_cal, np.asarray(D_cal, float)),
                            dtype=float).max(axis=1)                      # (n_cal,)
        n_cal = cal_c.size
        q_levels = np.minimum(1.0, ((n_cal + 1) * levels) / n_cal)
        thresholds = np.quantile(cal_c, q_levels, method='higher')        # (M,)
    else:
        thresholds = levels

    # ------------------------------------------------------------------ #
    # 3. Empirical joint coverage and calibration error.
    # ------------------------------------------------------------------ #
    joint_coverage = (joint_c_min[:, None] <= thresholds[None, :]).mean(axis=0)  # (M,)

    gap          = np.abs(joint_coverage - levels)                        # (M,)
    joint_ece    = float(gap.mean())
    joint_max_ce = float(gap.max())

    return {
        'levels':         levels,
        'calibrated':     calibrated,
        'thresholds':     thresholds,
        'joint_c_min':    joint_c_min,
        'joint_coverage': joint_coverage,
        'joint_ece':      joint_ece,
        'joint_max_ce':   joint_max_ce,
    }
