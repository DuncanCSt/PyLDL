"""Worker module for parallel CV in edl_full_experiment.ipynb.

Kept as a .py file (not a notebook cell) so loky subprocesses can import it
cleanly. TensorFlow is imported lazily inside `init_worker` and `run_one_fold`
so each worker pays its TF init cost once, *after* CUDA_VISIBLE_DEVICES has
been pinned.
"""
import os

METRICS = ['chebyshev', 'clark', 'canberra', 'kl_divergence', 'cosine', 'intersection']

MODEL_NAMES = [
    'EDL_LDL (loglikelihood)', 'EDL_LDL (bayes_mse)',
    'BEDL_LDL (loglikelihood)', 'BEDL_LDL (bayes_mse)',
    'Duo_LDL', 'AA_BP', 'SNEFY_LDL', 'SA_BFGS',
]


def detect_gpus():
    """Return list of GPU ids visible to nvidia-smi. Empty list if no GPUs.

    Uses the nvidia-smi shell tool rather than `tf.config` so the parent
    process never imports TensorFlow — that matters because once TF is
    imported in the parent, it grabs CUDA state that conflicts with the
    workers' own per-process pinning.
    """
    import subprocess
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return []
        return [int(line.strip()) for line in out.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.SubprocessError):
        return []


def auto_pool_config(min_gpus_for_gpu_mode=2, cpu_workers=None):
    """Pick pool config based on what hardware is available.

    GPU mode (>= `min_gpus_for_gpu_mode` GPUs found): one worker pinned per
    GPU, single TF thread per worker (the GPU is the parallelism unit).

    CPU mode (otherwise): a few fat workers (default ~`cpu_count // 4`), one
    TF thread per worker. Heavy oversubscription destroys throughput because
    every worker's TF op-graph fights every other worker's for cores.
    """
    import os
    gpus = detect_gpus()
    if len(gpus) >= min_gpus_for_gpu_mode:
        return {
            'mode': 'GPU',
            'gpu_ids': gpus,
            'n_workers': len(gpus),
            'intra_op_threads': 1,
            'inter_op_threads': 1,
        }
    n = cpu_workers if cpu_workers is not None else max(1, (os.cpu_count() or 2) // 4)
    return {
        'mode': 'CPU',
        'gpu_ids': [],
        'n_workers': n,
        'intra_op_threads': 1,
        'inter_op_threads': 1,
    }


def init_worker(gpu_queue, intra_op_threads=1, inter_op_threads=1):
    """Pin this worker to one GPU (or CPU-only) and configure TF.

    Called once per loky worker process. CUDA_VISIBLE_DEVICES must be set
    before TF imports anything CUDA-aware, which is why this happens here.
    """
    gpu_id = gpu_queue.get()
    gpu_queue.put(gpu_id)                    # refill (worker-replacement fix)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if gpu_id is None else str(gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # --- Stop TF from JIT-compiling kernels (cc1plus invocations) ---
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=false'
    os.environ['XLA_FLAGS']    = '--xla_hlo_profile=false'

    import tensorflow as tf
    if gpu_id is None:
        tf.config.set_visible_devices([], 'GPU')
    for g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
    except Exception:
        pass


def _build_specs():
    from pyldl.algorithms import (
        EDL_LDL, BEDL_LDL, Duo_LDL, AA_BP, SNEFY_LDL, SA_BFGS,
    )
    return {
        'EDL_LDL (loglikelihood)':  {'cls': EDL_LDL,   'init': {'n_hidden': 64},
            'fit_kwargs': {'loss_type': 'loglikelihood'}, 'uncertainty_kind': 'edl',  'uses_epochs': True},
        'EDL_LDL (bayes_mse)':      {'cls': EDL_LDL,   'init': {'n_hidden': 64},
            'fit_kwargs': {'loss_type': 'bayes_mse'},     'uncertainty_kind': 'edl',  'uses_epochs': True},
        'BEDL_LDL (loglikelihood)': {'cls': BEDL_LDL,  'init': {'n_hidden': 64},
            'fit_kwargs': {'loss_type': 'loglikelihood'}, 'uncertainty_kind': 'edl',  'uses_epochs': True},
        'BEDL_LDL (bayes_mse)':     {'cls': BEDL_LDL,  'init': {'n_hidden': 64},
            'fit_kwargs': {'loss_type': 'bayes_mse'},     'uncertainty_kind': 'edl',  'uses_epochs': True},
        'Duo_LDL':                  {'cls': Duo_LDL,   'init': {'n_hidden': 64},
            'fit_kwargs': {},                             'uncertainty_kind': None,   'uses_epochs': True},
        'AA_BP':                    {'cls': AA_BP,     'init': {'n_hidden': 64},
            'fit_kwargs': {},                             'uncertainty_kind': None,   'uses_epochs': True},
        'SNEFY_LDL':                {'cls': SNEFY_LDL, 'init': {'n_hidden': 64},
            'fit_kwargs': {},                             'uncertainty_kind': 'snefy','uses_epochs': True},
        'SA_BFGS':                  {'cls': SA_BFGS,   'init': {},
            'fit_kwargs': {},                             'uncertainty_kind': None,   'uses_epochs': False},
    }


def _predict_with_uncertainty(model, kind, X_test):
    import numpy as np
    if kind is None:
        return model.predict(X_test), None
    out = model.predict(X_test, return_uncertainty=True)
    if kind == 'edl':
        D_pred, _variance, evidence_uncertainty = out
        u = np.asarray(evidence_uncertainty).reshape(-1)
    elif kind == 'snefy':
        D_pred, variance = out
        u = np.asarray(variance).sum(axis=1)
    else:
        raise ValueError(f'unknown uncertainty kind: {kind}')
    return np.asarray(D_pred), u


def _uncertainty_calibration(D_test, D_pred, uncertainty):
    import numpy as np
    from scipy.stats import spearmanr
    from pyldl.algorithms.utils import kl_divergence
    per_sample_kl = kl_divergence(D_test, D_pred, reduction=None)
    rho, _ = spearmanr(uncertainty, per_sample_kl)
    return float(rho), float(np.mean(uncertainty))


def run_one_fold(dataset_name, model_name, fold_idx,
                 X_train, D_train, X_test, D_test, n_epochs):
    """Worker entrypoint. Returns dict with dataset, model, fold, scores."""
    from pyldl.metrics import score
    import gc, keras

    spec = _build_specs()[model_name]
    fit_kwargs = dict(spec['fit_kwargs'])
    if spec['uses_epochs']:
        fit_kwargs['epochs'] = n_epochs

    model = spec['cls'](**spec['init'])
    model.fit(X_train, D_train, **fit_kwargs)

    D_pred, uncertainty = _predict_with_uncertainty(model, spec['uncertainty_kind'], X_test)
    fold_scores = score(D_test, D_pred, metrics=METRICS, return_dict=True)
    if uncertainty is not None:
        rho, mean_u = _uncertainty_calibration(D_test, D_pred, uncertainty)
        fold_scores['mean_uncertainty'] = mean_u
        fold_scores['uncertainty_calibration'] = rho

    keras.backend.clear_session()
    del model
    gc.collect()

    return {
        'dataset': dataset_name,
        'model': model_name,
        'fold': fold_idx,
        'scores': fold_scores,
    }
