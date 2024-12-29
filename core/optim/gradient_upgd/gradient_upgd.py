# import torch

# class UPGD_SGD(torch.optim.Optimizer):
#     def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
#         names, params = zip(*params)
#         defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names)
#         super(UPGD_SGD, self).__init__(params, defaults)

#     def step(self):
#         """
#         Purpose: Using UPGDScaledAdativeNormNoise as a basis, but combines SGD with utility gating and noise injection.
#         """
#         global_max_util = torch.tensor(-torch.inf)

#         for group in self.param_groups:
#             for name, p in zip(group["names"], group["params"]):
#                 if 'gate' in name or p.grad is None:
#                     continue

#                 state = self.state[p]

#                 # Initialize state if not present
#                 if len(state) == 0:
#                     state["step"] = 0
#                     state["avg_utility"] = torch.zeros_like(p.data)

#                 state["step"] += 1

#                 # Update running average of utility
#                 avg_utility = state["avg_utility"]
#                 avg_utility.mul_(group["beta_utility"]).add_(
#                     -p.grad.data * p.data, alpha=1 - group["beta_utility"]
#                 )

#                 # Update global max utility
#                 current_util_max = avg_utility.max()
#                 if current_util_max > global_max_util:
#                     global_max_util = current_util_max

#         for group in self.param_groups:
#             for name, p in zip(group["names"], group["params"]):
#                 if 'gate' in name or p.grad is None:
#                     continue

#                 state = self.state[p]
#                 bias_correction = 1 - group["beta_utility"] ** state["step"]

#                 # Compute norms
#                 weight_norm = torch.norm(p.data) + 1e-8
#                 grad_norm = torch.norm(p.grad.data) + 1e-8
#                 scaling_factor = grad_norm / weight_norm

#                 # Generate noise
#                 noise = torch.randn_like(p.grad) * group["sigma"] * scaling_factor

#                 # Scale utility using sigmoid and global max utility
#                 scaled_utility = torch.sigmoid(
#                     (state["avg_utility"] / bias_correction) / global_max_util
#                 )

#                 # Perform the SGD update with utility gating and noise
#                 p.data.mul_(1 - group["lr"] * group["weight_decay"])  # Apply weight decay
#                 p.data.add_(
#                     -(p.grad.data * (1 - scaled_utility)) + noise, alpha=group["lr"]
#                 )
import torch 

class UPGD_SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names)
        super(UPGD_SGD, self).__init__(params, defaults)

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
                # When the optimizer encounters a parameter p for the first time, it initializes the state for that parameter.
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                # Maintains and updates a running average of a utility metric for each parameter.
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

                if len(scaled_utility.shape) == 4:  # We are in a convolutional layer:
                    # Compute column-wise average of the utility (average over kernel height only).
                    avg = scaled_utility.mean(dim=2)  # avg shape: [out_channels, in_channels, kernel_width]

                    # Step 2: Expand back to match the original shape
                    avg_expanded = avg.unsqueeze(-2)  # Add kernel height dimension back (shape: [out_channels, in_channels, 1, kernel_width])
                    column_kernel_utility = avg_expanded.expand(-1, -1, scaled_utility.size(2), -1)

                    try:
                        var1 = group["lr"] * group["weight_decay"]
                    except Exception as e:
                        print(e)
                        raise e

                    alphavar = -2.0 * group["lr"]
                    p.data.mul_(1 - var1).add_(
                        (p.grad.data + noise) * (1 - column_kernel_utility),
                        alpha=alphavar,
                    )

                else:  # For non-convolutional layers
                    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                        (p.grad.data + noise) * (1 - scaled_utility),
                        alpha=-2.0 * group["lr"],
                    )
