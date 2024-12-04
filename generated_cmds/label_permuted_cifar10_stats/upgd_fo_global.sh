#!/bin/bash
#SBATCH --signal=USR1@90
#SBATCH --job-name=upgd_fo_global			# single job name for the array
#SBATCH --mem=2G			# maximum memory 100M per job
#SBATCH --time=01:00:00			# maximum wall time per job in d-hh:mm or hh:mm:ss
#SBATCH --array=1-240
#SBATCH --account=def-ashique
cd ../../
FILE="$SCRATCH/upgd/generated_cmds/label_permuted_cifar10_stats/upgd_fo_global.txt"
SCRIPT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $FILE)
module load python/3.7.9
source $SCRATCH/upgd/.upgd/bin/activate
srun $SCRIPT
