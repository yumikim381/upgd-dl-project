import torch
from torch.nn import functional as F

# UPGD: Utilited-based Perturbed Gradient Descent: variation 2 (utility controls gradient)
# Yumi used this, will be our main function
class FirstOrderGlobalKernelUPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names)
        super(FirstOrderGlobalKernelUPGD, self).__init__(params, defaults)

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
                #print("name",name)
                #print("p",p)
                if 'gate' in name:
                    continue
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                # Add noise 
                """
                TODO: Adds Gaussian noise (torch.randn_like(p.grad) * sigma) to the gradient.
                """
                noise = torch.randn_like(p.grad) * group["sigma"]
                # Scales the smoothed utility by the global max utility using a sigmoid function.
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)
                """
                TODO: Change utility according to kernel utility 
                """
                
                if len(scaled_utility.shape) == 4: # We are in convolutional layer:
                    
                    avg = scaled_utility.mean(dim=[2, 3])  # avg shape: [out_channels, in_channels]

                    # Step 2: Inflate back to original shape
                    # First, add back the spatial dims
                    avg_expanded = avg.unsqueeze(-1).unsqueeze(-1)  # shape: [out_channels, in_channels, 1, 1]

                    # Now expand along the spatial dimensions
                    averagekernel_utility = avg_expanded.expand(-1, -1, scaled_utility.size(2), scaled_utility.size(3)) 
                    
                
                    try:
                        #print(type(group["lr"])) 
                        #print(group["lr"])
                        #print(type(group["weight_decay"]))
                        var1 = group["lr"] * group["weight_decay"]

                        
                    except Exception as e:
                        print(e)
                        raise e
                     # inflated shape: [out_channels, in_channels, kernel_height, kerne
                    alphavar = -2.0*group["lr"]
                    p.data.mul_(1 - var1).add_(
                            (p.grad.data + noise) * (1-averagekernel_utility),
                            alpha=alphavar,
                        )
                    
                else:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                        (p.grad.data + noise) * (1-scaled_utility),
                        alpha=-2.0*group["lr"],
                    )
                

                # p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                #     (p.grad.data + noise)
                #     * (1 - scaled_utility),
                #     alpha=-group["lr"],
                # )

# keep for now
class FirstOrderLocalKernelUPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names)
        super(FirstOrderLocalKernelUPGD, self).__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                noise = torch.randn_like(p.grad) * group["sigma"]
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                scaled_utility = torch.sigmoid_(
                    F.normalize((avg_utility / bias_correction), dim=-1)
                )
                # p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                #     (p.grad.data + noise) * (1 - scaled_utility), alpha=-group["lr"]
                # )
                if len(scaled_utility.shape) == 4: # We are in convolutional layer:
                    avg = scaled_utility.mean(dim=[2, 3])  # avg shape: [out_channels, in_channels]

                    # Step 2: Inflate back to original shape
                    # First, add back the spatial dims
                    avg_expanded = avg.unsqueeze(-1).unsqueeze(-1)  # shape: [out_channels, in_channels, 1, 1]

                    # Now expand along the spatial dimensions
                    averagekernel_utility = avg_expanded.expand(-1, -1, scaled_utility.size(2), scaled_utility.size(3))  
                    # inflated shape: [out_channels, in_channels, kernel_height, kerne
                    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                        (p.grad.data + noise) * (1-averagekernel_utility),
                        alpha=-2.0*group["lr"],
                    )
                else:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                        (p.grad.data + noise) * (1-scaled_utility),
                        alpha=-2.0*group["lr"],
                    )