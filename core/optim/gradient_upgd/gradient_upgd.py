import torch

class UPGD_SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names)
        super(UPGD_SGD, self).__init__(params, defaults)

    def step(self):
        """
        Purpose: Using UPGDScaledAdativeNormNoise as a basis, but combines SGD with utility gating and noise injection.
        """
        global_max_util = torch.tensor(-torch.inf)

        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name or p.grad is None:
                    continue

                state = self.state[p]

                # Initialize state if not present
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)

                state["step"] += 1

                # Update running average of utility
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )

                # Update global max utility
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max

        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name or p.grad is None:
                    continue

                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]

                # Compute norms
                weight_norm = torch.norm(p.data) + 1e-8
                grad_norm = torch.norm(p.grad.data) + 1e-8
                scaling_factor = grad_norm / weight_norm

                # Generate noise
                noise = torch.randn_like(p.grad) * group["sigma"] * scaling_factor

                # Scale utility using sigmoid and global max utility
                scaled_utility = torch.sigmoid(
                    (state["avg_utility"] / bias_correction) / global_max_util
                )

                # Perform the SGD update with utility gating and noise
                p.data.mul_(1 - group["lr"] * group["weight_decay"])  # Apply weight decay
                p.data.add_(
                    -(p.grad.data * (1 - scaled_utility)) + noise, alpha=group["lr"]
                )
