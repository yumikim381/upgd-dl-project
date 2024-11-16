#!/bin/bash
#SBATCH --signal=USR1@90
#SBATCH --job-name=requirements			# single job name for the array
#SBATCH --mem=2G			# maximum memory 100M per job
#SBATCH --time=01:00:00			# maximum wall time per job in d-hh:mm or hh:mm:ss
#SBATCH --array=1-240
#SBATCH --account=def-ashique
cd ../../
FILE="$SCRATCH/upgd/generated_cmds/input_permuted_mnist/requirements.txt"
SCRIPT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $FILE)
module load python/3.8.20
source $SCRATCH/upgd/.upgd/bin/activate
srun $SCRIPT
