import torch
from torch.nn import functional as F

from core.network.fcn_relu import ConvolutionalNetworkReLUWithHooks

class FirstOrderGlobalKernelPruningUPGD(torch.optim.Optimizer):
    def __init__(self, params, model: ConvolutionalNetworkReLUWithHooks, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        self.model = model
        print(model.activations_out)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names)
        super(FirstOrderGlobalKernelPruningUPGD, self).__init__(params, defaults)

    def step(self):
        """
        Purpose: Tracks a running average of the utility (avg_utility) for each parameter:
        Utility is defined as -p.grad.data * p.data (gradient scaled by parameter value).
        The running average is computed using exponential smoothing with beta_utility.
        The maximum utility across all parameters is stored in global_max_util.
        """
        # maximum utility across all parameters is stored in global_max_util
        global_max_util = torch.tensor(-torch.inf)

        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                # Maintain the running average of the utility: state["avg_utility"].
                state = self.state[p]
                # When the optimizer encounters a parameter p for the first time, it initializes the state for that paramete
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                #  Maintains and updates a running average of a utility metric for each parameter.
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max
                    
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                # Add noise 
                noise = torch.randn_like(p.grad) * group["sigma"]
                # Scales the smoothed utility by the global max utility using a sigmoid function.
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)
                if len(scaled_utility.shape) == 4: # We are in convolutional layer:
                    name = name.split('.')[0]
                    if not name in self.model.activations_out.keys():
                        raise KeyError(f"Could not find layer {name} in activations ")
                    # print(self.model.activations_out)
                    activations: torch.Tensor = self.model.activations_out[name]

                    if not name in self.model.gradients.keys():
                        raise KeyError(f"Could not find layer {name} in activations ")
                    gradient = self.model.gradients[name]

                    result = activations * gradient
                    result = torch.mean(result, dim=(0,2,3)).view(-1, 1,1,1).expand(-1, scaled_utility.shape[1], scaled_utility.shape[2], scaled_utility.shape[3])
                    kernel_utility = torch.sigmoid(result)
                    # Inflate to same shape as scaled_utility

                    LOCAL_FACTOR = 0.75

                    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                        LOCAL_FACTOR * (p.grad.data + noise) * (1-scaled_utility) + (1-LOCAL_FACTOR) * (p.grad.data + noise) * (1-kernel_utility),
                        alpha=-2.0*group["lr"],
                    )
                
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                        (p.grad.data + noise) * (1-scaled_utility),
                        alpha=-2.0*group["lr"],
                    )