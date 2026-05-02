"""Worker module for parallel KerasTuner BO of any PyLDL neural-net model
that follows the `BaseAdam` fit interface (`EDL`, `EDL_BAYES`, `BEDL`,
`BEDL_BAYES`, `Duo_LDL`, `AA_BP`, `SNEFY_LDL`, ...).

Concurrency model: each (model, dataset) pair runs its own complete BO
search on one GPU. Multiple pairs run in parallel by giving each loky
worker a different GPU at init time.

Best hyperparameters per (model, dataset) are written to
`<hyperparams_dir>/<model>__<dataset>.txt` as JSON, so future runs can
read them back via `load_best_hyperparams`.

TF is imported lazily so each subprocess controls device visibility
before TF grabs CUDA state.
"""
import json
import os
from pathlib import Path
import multiprocessing as mp
from loky import get_reusable_executor
from pyldl.algorithms._bedl import EDL



def load_data_fold(dataset_name, fold):
    """Returns (X_train, D_train, X_test, D_test) for this fold of this dataset."""
    import scipy.io as sio


    path = Path(__file__).parent.parent / 'dataset_splits' / dataset_name
    train = path / f"{fold}_train_{dataset_name}.mat"
    test = path / f"{fold}_test_{dataset_name}.mat"
    data_train = sio.loadmat(train)
    data_test = sio.loadmat(test)
    return data_train['features'], data_train['labels'], data_test['features'], data_test['labels']

def read_results(model, dataset, fold):
    """Read results JSON for a given model, dataset, and fold."""
    path = Path(__file__).parent.parent / 'results' / model / dataset / f"fold_{fold}.txt"
    with open(path, 'r') as f:
        return json.load(f)
    
def write_results(model, dataset, fold, results):
    """Write results JSON for a given model, dataset, and fold.
    
    If results already exist, update/append keys from current results.
    """
    
    path = Path(__file__).parent.parent / 'results' / model / dataset
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"fold_{fold}.txt"
    
    existing = {}
    if file_path.exists():
        with open(file_path, 'r') as f:
            existing = json.load(f)
    
    existing.update(results)
    with open(file_path, 'w') as f:
        json.dump(existing, f, indent=2, sort_keys=True)

def detect_gpus():
    """Return list of GPU ids visible to nvidia-smi. Empty if none."""
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

def init_worker(gpu_queue, intra_op_threads=1, inter_op_threads=1):
    """Pin this loky worker to one GPU (or CPU-only) and cap TF threads.

    The queue refill keeps things consistent if loky replaces a crashed
    worker. TF threads are capped so multiple CPU workers don't fight each
    other for cores.
    """
    gpu_id = gpu_queue.get()
    gpu_queue.put(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if gpu_id is None else str(gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


def auto_pool_config(min_cpus_per_worker=4):
    """Pick a pool config based on visible hardware.

    GPU mode (any GPU visible): one worker per GPU; TF threads capped to 1
    each since the GPU is the parallelism unit.

    CPU mode (no GPUs): split available cores into chunks of ~`min_cpus_per_worker`
    cores per worker (1 worker on a 5-core box, 2 on 8 cores, 4 on 16, etc.)
    and give each worker that many intra-op threads. Heavy oversubscription
    destroys throughput because every worker's TF op-graph fights every other
    worker's for cores.
    """
    gpus = detect_gpus()
    if gpus:
        return {
            'mode': 'GPU',
            'slots': list(gpus),
            'intra_op_threads': 1,
            'inter_op_threads': 1,
        }
    n_cpus = os.cpu_count() or 2
    n_workers = max(1, n_cpus // min_cpus_per_worker)
    intra = max(1, n_cpus // n_workers)
    return {
        'mode': 'CPU',
        'slots': [None] * n_workers,
        'intra_op_threads': intra,
        'inter_op_threads': 1,
    }


def create_executer(config=None, verbose=True):
    """Build a loky pool sized to the available hardware.

    Pass a dict from `auto_pool_config()` (or build your own) to override the
    auto-detected config.
    """
    cfg = config or auto_pool_config()
    slots = cfg['slots']

    if verbose:
        print(f"mode    : {cfg['mode']}")
        print(f"workers : {len(slots)}  (slots={slots})")
        print(f"threads : intra={cfg['intra_op_threads']}, inter={cfg['inter_op_threads']}")

    mgr = mp.Manager()
    gpu_queue = mgr.Queue()
    for g in slots:
        gpu_queue.put(g)

    return get_reusable_executor(
        max_workers=len(slots),
        initializer=init_worker,
        initargs=(gpu_queue, cfg['intra_op_threads'], cfg['inter_op_threads']),
        reuse=False,
    )

def _resolve_model_cls(model_cls_name):
    import pyldl.algorithms
    try:
        return getattr(pyldl.algorithms, model_cls_name)
    except AttributeError as e:
        raise ValueError(
            f'unknown model {model_cls_name!r}; must be importable from pyldl.algorithms'
        ) from e

def load_best_model(model, dataset, fold):
    """Load the best hyperparameters for this (model, dataset) and return a model instance."""

    best_hps = read_results(model, dataset, fold)['hyperparameters']
    model_cls = _resolve_model_cls(model)
    return model_cls(n_hidden=best_hps['n_hidden'], n_latent=best_hps['n_latent'])

def fit_best_model(model, dataset, fold, train_data=None, valid_data=None, extra_fit_kwargs=None):
    """Load the best hyperparameters for this (model, dataset), fit a model instance on the full training set, and return it."""
    from sklearn.model_selection import train_test_split
    from keras.optimizers import AdamW
    from pyldl.utils import LDLEarlyStopping, LossHistory

    if train_data is not None and valid_data is not None:
        X_train, D_train = train_data['X'], train_data['D']
        X_val, D_val = valid_data['X'], valid_data['D']
    else:
        X_train, D_train, _, _ = load_data_fold(dataset, fold)
        X_train, X_val, D_train, D_val = train_test_split(X_train, D_train, test_size=0.2, random_state=42)

    hps = read_results(model, dataset, fold)['hyperparameters']
    model_instance = load_best_model(model, dataset, fold)
    history = LossHistory()
    model_instance.fit(
        X_train, D_train,
        X_val=X_val, D_val=D_val,
        epochs=hps['max_epochs'],
        batch_size=hps['batch_size'],
        optimizer=AdamW(
            learning_rate=hps['learning_rate'],
            weight_decay=hps['weight_decay'],
        ),
        dropout_rate=hps['dropout_rate'],
        callbacks=[LDLEarlyStopping(monitor='kl_divergence',
                                    patience=hps['patience'],
                                    minimum=hps['minimum']),
                    history
                    ],
        verbose=0,
        **(extra_fit_kwargs or {}),
    )
    return model_instance, history

def plot_history(history, title=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='train')
    ax2 = ax.twinx()
    ax2.plot(history.history['kl_divergence'], color = 'orange', label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (Train)')
    ax2.set_ylabel('KL Divergence (Val)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('Training History' if title is None else title)
    plt.legend()
    plt.show()