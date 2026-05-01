"""Driver for SLURM array jobs.

Each SLURM array task picks one (model, dataset, fold) from JOBS via
`SLURM_ARRAY_TASK_ID` and runs a complete BO hyperparameter search.

Submit with:

    sbatch --array=0-$(($(python -c "from run_one import JOBS; print(len(JOBS))") - 1)) ...

Results are written to results/<model>/<dataset>/fold_<fold>.txt by
hypertuner.run_search.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hypertuner import run_search


MODELS   = ['EDL', 'EDL_BAYES', 'BEDL', 'BEDL_BAYES']
DATASETS = ['Movie']
FOLDS    = list(range(10))   # 10-fold CV

MAX_TRIALS           = 25
EXECUTIONS_PER_TRIAL = 1


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
    )
    print(f'[{idx}] done: val_kl_divergence={result["val_kl_divergence"]:.4f}, '
          f'epochs_run={result["epochs_run"]}', flush=True)


if __name__ == '__main__':
    main()
