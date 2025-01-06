# Enhancing utility-based Perturbed Gradient Descent with Adaptive Noise Injection

## Deep Learning Project
This is the repository for the ETH Deep Learning project, based on the original paper "Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning" from [here](https://openreview.net/forum?id=sKPzAXoylB). We propose an improvement to the Utility-based Perturbed Gradient Descent (UPGD) method by injecting adaptive noise. 

Details of our approach and results can be found in our Report (TODO: link final report)

Here we describe how to reproduce the results. 

## Installation 
### 1. You need to have an environemnt with python 3.7:
``` sh
git clone  --recursive git@github.com:yumikim381/upgd-dl-project.git
python3.7 -m venv .upgd
source .upgd/bin/activate
```

### 2. Install Dependencies:
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
  --learner upgd_fo_global \
  --seed 19 \
  --lr 0.01 \
  --beta_utility 0.999 \
  --sigma 0.001 \
  --weight_decay 0.0 \
  --network convolutional_network_relu_with_hooks \
  --n_samples 1000000
```

### Run Visualizations
TODO (check with Tilman)

### Run best Method with Adaptive Noise Injection
```sh
python3 core/run/run_stats.py \
  --task label_permuted_cifar10_stats \
  --learner UPGDScaledAdativeNormNoiseDLearner \
  --seed 19 \
  --lr 0.01 \
  --beta_utility 0.999 \
  --sigma 0.001 \
  --weight_decay 0.0 \
  --network convolutional_network_relu_with_hooks \
  --n_samples 1000000
```

`ratio_norm` can be replaced with the following options to runn other variations we have tried out:
1. Layer-wise Noise Scaling
   - `weight_norm` for Scaling by the Norm of Weights
   - `grad_norm` for Scaling by the Norm of Gradients
2. Kernel Utility
   - `entire_kernel` for Entire Kernel Evaluation
   - `kernel_pruning_upgd` Convex Combination between Neuron and Kernel Evaluation
   - `column_kernel` for Column-wise Kernel Evaluation


#### Get results for all Methods we have implemented
TODO (code doesn't exist yet)


------
### Old stuff from original repo that we might need:

If you only want the implementation of the UPGD algorithm you can find it here:

```python
import torch

class UPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.001, beta_utility=0.999, sigma=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma)
        super(UPGD, self).__init__(params, defaults)
    def step(self):
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                bias_correction_utility = 1 - group["beta_utility"] ** state["step"]
                noise = torch.randn_like(p.grad) * group["sigma"]
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction_utility) / global_max_util)
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + noise) * (1-scaled_utility),
                    alpha=-2.0*group["lr"],
                )
```


#### Label-permuted CIFAR-10/EMNIST/miniImageNet (Figure 6):
You first need to define the grid search of each method then you generate then python cmds using:
```sh
python experiments/label_permuted_cifar10.py
```
This would generate a list of python cmds you need to run them. After they are done, the results would be saved in `logs/` in a JSON format. To plot, use the following after choosing what to plot:
```sh
python core/plot/plotter.py
```

#### Input/Label-permuted Tasks Diagnostic Statistics (Figure 5):
You first need to choose the method and the hyperparameter setting you want to run the statistics on from:
```sh
python experiments/statistics_output_permuted_cifar10.py
```
This would generate a list of python cmds you need to run them. After they are done, the results would be saved in `logs/` in a JSON format.


#### Policy collapse experiment (Figure 8):
You need to choose the environment id and the seed number. In the paper, we averaged over 30 different seeds.
```sh
python core/run/rl/ppo_continuous_action_adam.py --seed 0 --env_id HalfCheetah-v4
python core/run/rl/ppo_continuous_action_upgd.py --seed 0 --env_id HalfCheetah-v4
```
