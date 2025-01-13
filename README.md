# Enhancing utility-based Perturbed Gradient Descent with Adaptive Noise Injection

## Deep Learning Project

This is the repository for the ETH Deep Learning project, based on the original paper "Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning" from [here](https://openreview.net/forum?id=sKPzAXoylB). We propose Utility-based Stochastic Gradient Descent with Adaptive Noise injection as an improvement to the Utility-based Perturbed Gradient Descent (UPGD) method. Experiments show that our model achieves higher average accuracy (55.38\% vs. 55.29\%) and average plasticity (45.52\% vs. 41.86\%).

Details of our approach and results can be found in our [Report](./usgd_with_adaptive_noise.pdf)

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

Use the notebook `notebooks/visualize_kernels.ipynb` to run through the visualizations. All have individual cells, and should run with the same set of requirements as the rest of the code, with the potential exception of ipython and jupyterlab.

### Run best Method with Adaptive Noise Injection
To conduct the accuracy and catastrophic forgetting experiment using the CIFAR-10 dataset, run the following:

```sh
python3 core/run/run_stats.py \
  --task label_permuted_cifar10_stats \
  --learner usgd \
  --seed 19 \
  --lr 0.01 \
  --beta_utility 0.999 \
  --sigma 0.001 \
  --weight_decay 0.0 \
  --network convolutional_network_relu_with_hooks \
  --n_samples 1000000
```

To conduct the loss of plasticity experiment based on the MNIST dataset, run:

```sh
python3 core/run/run_stats.py \
  --task input_permuted_mnist_stats \
  --learner usgd \
  --seed 19 \
  --lr 0.01 \
  --beta_utility 0.999 \
  --sigma 0.001 \
  --weight_decay 0.0 \
  --network conv_mnist \
  --n_samples 1000000
```

`usgd` can be replaced with the following options to run other variations we have tried out:

1. Layer-wise Noise Scaling
   - `weight_norm` for scaling by the norm of weights
   - `grad_norm` for scaling by the norm of gradients
   - `ratio_norm`for scaling by the ratio of the gradient norm to the weight norm
2. Kernel Utility
   - `entire_kernel` for entire kernel evaluation
   - `column_kernel` for column-wise kernel evaluation

### Get results for all Methods we have implemented
Use the notebook `notebooks/get_results.ipynb` to get the evaluations of the 2 experiments. The metrics will be printed out in a table format and graphical visualizations of the performance are provided. 
