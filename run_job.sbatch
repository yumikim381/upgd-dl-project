#!/bin/bash

#SBATCH --job-name=run_stats_job
#SBATCH --account=dl_jobs
#SBATCH --partition=standard # Partition/queue name
#SBATCH --output=nvcc.out # Standard output file
#SBATCH --error=run_stats_job.err # Standard error file
#SBATCH --time=10:00:00 # Maximum execution time (HH:MM:SS)



# Activate virtual environment (if applicable)
conda init
conda activate upgdnew
pip install -r requirements.txt 
pip install HesScale/.
pip install .

# Execute the command
python3 core/run/run_stats.py \
  --task label_permuted_cifar10_stats \
  --learner kernel_pruning_upgd \
  --seed 19 \
  --lr 0.01 \
  --beta_utility 0.999 \
  --sigma 0.001 \
  --weight_decay 0.0 \
  --network convolutional_network_relu_with_hooks \
  --n_samples 1000000

