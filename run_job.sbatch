#!/bin/bash

#SBATCH --job-name=run_stats_job
#SBATCH --account=dl_jobs
#SBATCH --partition=standard # Partition/queue name
#SBATCH --output=nvcc.out # Standard output file
#SBATCH --error=run_stats_job.err # Standard error file
#SBATCH --time=01:00:00 # Maximum execution time (HH:MM:SS)

# Load any required modules (if applicable)
module load python/3.7.16                    # Example module, adjust based on your system

# Activate virtual environment (if applicable)
conda activate upgdnew

pip install .

# Execute the command
python3 core/run/run_stats.py \
  --task label_permuted_cifar10_stats \
  --learner UPGDScaledWeightNormNoiseLearner \
  --seed 19 \
  --lr 0.01 \
  --beta_utility 0.999 \
  --sigma 0.001 \
  --weight_decay 0.0 \
  --network convolutional_network_relu_with_hooks \
  --n_samples 1000000

