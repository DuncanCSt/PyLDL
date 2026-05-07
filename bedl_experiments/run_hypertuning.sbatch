#!/bin/bash
#SBATCH --job-name=tuning_jobs
#SBATCH --array=0-1
# #SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

# --- environment ---
module --force purge
module load StdEnv/2023
module load python/3.12.4

# source ~/envs/pyldl/bin/activate
export PYTHONPATH=/project/6004619/dcs01/PyLDL:$PYTHONPATH

# Thread caps so loky workers don't oversubscribe BLAS/OMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TF_CPP_MIN_LOG_LEVEL=3

cd "$SLURM_SUBMIT_DIR"

echo "Node      : $(hostname)"
echo "Array idx : $SLURM_ARRAY_TASK_ID"
echo "GPU       : $CUDA_VISIBLE_DEVICES"

python bedl_experiments/run_hypertuning.py