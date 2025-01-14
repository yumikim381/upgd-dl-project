import torch
from torch.nn import functional as F

class UPGDScaledWeightNormNoise(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names)
        super(UPGDScaledWeightNormNoise, self).__init__(params, defaults)

    def step(self):
        """
        Purpose: scales noise by weight norm 
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
                """
                NOTE: Scales noise by the weight norm of the layer (Layer-wise Scaled PGD).
                """
                scaling_factor = torch.norm(p.data) + 1e-8
                noise = torch.randn_like(p.grad) * group["sigma"]* scaling_factor
                # Scales the smoothed utility by the global max utility using a sigmoid function.
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)

                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + noise)
                    * (1 - scaled_utility),
                    alpha=-group["lr"],
                )


class UPGDScaledGradNormNoise(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names)
        super(UPGDScaledGradNormNoise, self).__init__(params, defaults)

    def step(self):
        """
        Purpose: scales noise by grade norm 
        """
        # maximum utility across all parameters is stored in global_max_util
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                # For each parameter, you're using state to:
                #Keep track of the step count: state["step"].
                #Maintain the running average of the utility: state["avg_utility"].
                state = self.state[p]
                #When the optimizer encounters a parameter p for the first time, it initializes the state for that paramete
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
                """
                NOTE: Scales noise by the gradient norm of the layer (Gradient-Norm-Based PGD).
                """
                scaling_factor = torch.norm(p.grad.data) + 1e-8
                noise = torch.randn_like(p.grad) * group["sigma"]* scaling_factor
                # Scales the smoothed utility by the global max utility using a sigmoid function.
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)

                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + noise)
                    * (1 - scaled_utility),
                    alpha=-group["lr"],
                )

class UPGDScaledAdativeNormNoise(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names)
        super(UPGDScaledAdativeNormNoise, self).__init__(params, defaults)

    def step(self):
        """
        Purpose: scales noise by ratio between grade norm and weight norm 
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
                """
                NOTE: Combines both weight and gradient norms (Adaptive Layer-wise PGD)
                """
                weight_norm = torch.norm(p.data) + 1e-8
                grad_norm = torch.norm(p.grad.data) + 1e-8
                scaling_factor = grad_norm / weight_norm
                noise = torch.randn_like(p.grad) * group["sigma"]* scaling_factor
                # Scales the smoothed utility by the global max utility using a sigmoid function.
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)

                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + noise)
                    * (1 - scaled_utility),
                    alpha=-group["lr"],
                )