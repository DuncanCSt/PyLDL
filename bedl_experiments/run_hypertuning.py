"""Driver for SLURM array jobs.

Each SLURM array task picks one (model, dataset, fold) from JOBS via
`SLURM_ARRAY_TASK_ID` and runs a complete BO hyperparameter search.

Results are written to results/<model>/<dataset>/fold_<fold>.txt by
hypertuner.run_search.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hypertuner import run_search

# ['BOOJUM', 'BOOJUM_BAYES', 'EDL', 'EDL_BAYES', 'BEDL', 'BEDL_BAYES']
MODELS   = ['SNEFY_LDL', 'AA_BP']
DATASETS = ['SJAFFE']
FOLDS    = [1, 2, 3, 4, 5, 6, 7, 8, 9]   # 10-fold CV

MAX_TRIALS           = 15
EXECUTIONS_PER_TRIAL = 1

HYPERTUNING_SETTINGS = {
    'Movie': {
        'n_hidden': [4, 8, 16],
        'n_latent': 16,
        'learning_rate': [1e-3],
        'weight_decay': [1e-5, 1e-4, 1e-3, 1e-2],
        'dropout_rate': [0.0, 0.2, 0.4],
        'patience': 50,
        'minimum': 100,
        'batch_size': [16, 128],
        'max_epochs': 1500,
    },

    'SJAFFE': {
        'n_hidden': [8, 16, 32, 64],
        'n_latent': 16,
        'learning_rate': [1e-3],
        'weight_decay': [1e-5, 1e-4, 1e-3, 1e-2],
        'dropout_rate': [0.0, 0.2, 0.4],
        'patience': 50,
        'minimum': 100,
        'batch_size': [16, 128],
        'max_epochs': 1500,
    }
}

JOBS = [(m, d, f) for m in MODELS for d in DATASETS for f in FOLDS]


def main():
    if 'SLURM_ARRAY_TASK_ID' not in os.environ:
        raise RuntimeError('SLURM_ARRAY_TASK_ID not set — submit via `sbatch --array=...`')

    idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
    if not 0 <= idx < len(JOBS):
        raise IndexError(f'SLURM_ARRAY_TASK_ID={idx} out of range (have {len(JOBS)} jobs)')

    model, dataset, fold = JOBS[idx]
    print(f'[{idx}/{len(JOBS) - 1}] {model} × {dataset} × fold {fold}', flush=True)

    result = run_search(
        model_cls_name=model,
        dataset_name=dataset,
        fold=fold,
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTIONS_PER_TRIAL,
        hypertuning_settings=HYPERTUNING_SETTINGS
    )
    print(f'[{idx}] done: val_kl_divergence={result["val_kl_divergence"]:.4f}, '
          f'epochs_run={result["epochs_run"]}', flush=True)


if __name__ == '__main__':
    main()
