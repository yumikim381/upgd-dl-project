# Enhancing utility-based Perturbed Gradient Descent with Adaptive Noise Injection

## Deep Learning Project

This is the repository for the ETH Deep Learning project, based on the original paper "Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning" from [here](https://openreview.net/forum?id=sKPzAXoylB). We propose an improvement to the Utility-based Perturbed Gradient Descent (UPGD) method by injecting adaptive noise.

Details of our approach and results can be found in our Report (TODO: link final report)

Here we describe how to reproduce the results.

## Installation

### 1. You need to have an environemnt with python 3.7

``` sh
git clone  --recursive git@github.com:yumikim381/upgd-dl-project.git
python3.7 -m venv .upgd
source .upgd/bin/activate
```

### 2. Install Dependencies

```sh
python -m pip install --upgrade pip
pip install -r requirements.txt 
pip install HesScale/.
pip install .
```

## Run our Results

### Run Baseline

We have used Algorithm 1 of the original UPGD as our baseline. This code can be run as follows:

```sh
python3 core/run/run_stats.py \
  --task label_permuted_cifar10_stats \
  --learner baseline \
  --seed 19 \
  --lr 0.01 \
  --beta_utility 0.999 \
  --sigma 0.001 \
  --weight_decay 0.0 \
  --network convolutional_network_relu_with_hooks \
  --n_samples 1000000
```

### Run Visualizations

Use the notebook `visualize_kernels.ipynb` to run through the visualizations. All have individual cells, and should run with the same set of requirements as the rest of the code, with the potential exception of ipython and jupyterlab.

### Run best Method with Adaptive Noise Injection

```sh
python3 core/run/run_stats.py \
  --task label_permuted_cifar10_stats \
  --learner ratio_norm \
  --seed 19 \
  --lr 0.01 \
  --beta_utility 0.999 \
  --sigma 0.001 \
  --weight_decay 0.0 \
  --network convolutional_network_relu_with_hooks \
  --n_samples 1000000
```

`ratio_norm` can be replaced with the following options to run other variations we have tried out:

1. Layer-wise Noise Scaling
   - `weight_norm` for Scaling by the Norm of Weights
   - `grad_norm` for Scaling by the Norm of Gradients
2. Kernel Utility
   - `entire_kernel` for Entire Kernel Evaluation
   - `KernelConvexCombi` Convex Combination between Neuron and Kernel Evaluation
   - `column_kernel` for Column-wise Kernel Evaluation

### Get results for all Methods we have implemented

To get the evaluations of average accuracy, average plasticity, Lipschitz constant, forgetting and loss of plasticity, run:

```sh
python3 get_results.py
```

This will also provide graphs to visualize the performances.
