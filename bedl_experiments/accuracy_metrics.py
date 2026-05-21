import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit

from helpers import write_results, load_data_fold, fit_best_model
from pyldl.metrics import score
from conformal_predictions import fsc_score, marginal_calibration

METRICS = ['chebyshev', 'clark', 'canberra', 'kl_divergence', 'cosine', 'intersection']
CONFORMAL_SCALAR_KEYS = ['worst_bin_fsc', 'joint_fsc']
MC_SCALAR_KEYS = ['ece', 'max_ce']


def run_metrics(model_cls_name, dataset_name, fold,
                hyperparams=None,
                n_splits=5, val_size=0.2,
                random_state=42, confidence=0.9, bin_count=10):
    """Run accuracy and conformal-prediction metrics for a given model,
    dataset, and fold.

    Refits the best model `n_splits` times on ShuffleSplit splits of the fold's
    training data. For each trial: evaluates accuracy metrics on the
    held-out test set, and uses the ShuffleSplit validation split as the
    calibration set for split conformal prediction. Writes the per-metric
    mean and variance across trials.
    """
    X_train, D_train, X_test, D_test = load_data_fold(dataset_name, fold)

    kf = ShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=random_state)
    trial_scores = []
    trial_conformal = []
    trial_marginal = []
    trial_marginal_cal = []
    for kf_train_idx, kf_val_idx in kf.split(X_train):
        X_kf_train, D_kf_train = X_train[kf_train_idx], D_train[kf_train_idx]
        X_kf_val, D_kf_val = X_train[kf_val_idx], D_train[kf_val_idx]

        model, _ = fit_best_model(
            model_cls_name, dataset_name, fold,
            train_data={'X': X_kf_train, 'D': D_kf_train},
            valid_data={'X': X_kf_val, 'D': D_kf_val},
            hyperparams=hyperparams,
        )
        D_pred = model.predict(X_test)
        trial_scores.append(score(D_test, D_pred, metrics=METRICS, return_dict=True))

        trial_conformal.append(fsc_score(
            model, X_kf_val, D_kf_val, X_test, D_test,
            confidence=confidence, bin_count=bin_count,
        ))

        trial_marginal.append(marginal_calibration(model, X_test, D_test))
        trial_marginal_cal.append(marginal_calibration(
            model, X_test, D_test, X_cal=X_kf_val, D_cal=D_kf_val,
        ))

    results = {}
    for metric in METRICS:
        values = np.array([s[metric] for s in trial_scores], dtype=float)
        results[f'{metric}_mean'] = float(values.mean())
        results[f'{metric}_var'] = float(values.var(ddof=1))

    for key in CONFORMAL_SCALAR_KEYS:
        values = np.array([c[key] for c in trial_conformal], dtype=float)
        results[f'{key}_mean'] = float(values.mean())
        results[f'{key}_var'] = float(values.var(ddof=1))

    per_label = np.stack([c['per_label_fsc'] for c in trial_conformal], axis=0)
    results['per_label_fsc_mean'] = per_label.mean(axis=0).tolist()
    results['per_label_fsc_var'] = per_label.var(axis=0, ddof=1).tolist()

    per_bin = np.stack([c['per_bin_fsc'] for c in trial_conformal], axis=0)
    results['per_bin_fsc_mean'] = np.nanmean(per_bin, axis=0).tolist()
    results['per_bin_fsc_var'] = np.nanvar(per_bin, axis=0, ddof=1).tolist()

    q_hat = np.stack([c['q_hat'] for c in trial_conformal], axis=0)
    results['q_hat_mean'] = q_hat.mean(axis=0).tolist()
    results['q_hat_var'] = q_hat.var(axis=0, ddof=1).tolist()

    # Marginal calibration check. The uncalibrated trials measure the model's
    # raw calibration; the `_cal` trials apply conformal recalibration using
    # the ShuffleSplit validation split as the calibration set.
    results['mc_levels'] = trial_marginal[0]['levels'].tolist()

    for suffix, trials in [('', trial_marginal), ('_cal', trial_marginal_cal)]:
        for key in MC_SCALAR_KEYS:
            values = np.array([m[key] for m in trials], dtype=float)
            results[f'{key}{suffix}_mean'] = float(values.mean())
            results[f'{key}{suffix}_var'] = float(values.var(ddof=1))

        for key in ['marginal_coverage', 'joint_coverage', 'per_label_ece']:
            stacked = np.stack([m[key] for m in trials], axis=0)
            results[f'{key}{suffix}_mean'] = stacked.mean(axis=0).tolist()
            results[f'{key}{suffix}_var'] = stacked.var(axis=0, ddof=1).tolist()

    write_results(
        model=model_cls_name,
        dataset=dataset_name,
        fold=fold,
        results=results,
    )
