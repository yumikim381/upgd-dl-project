import torch 
class UPGD_ColumnKernel(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma)
        super(UPGD_ColumnKernel, self).__init__(params, defaults)
        

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
                if len(scaled_utility.shape) == 4:  # Convolutional layer
                    # Compute average per column of the kernel
                    avg = scaled_utility.mean(dim=2)  # Average across kernel height; shape: [out_channels, in_channels, kernel_width]

                    # Expand back to match the kernel shape
                    avg_expanded = avg.unsqueeze(-2)  # Add kernel height dimension back
                    averagekernel_utility = avg_expanded.expand(-1, -1, scaled_utility.size(2), scaled_utility.size(3))

                    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                        (p.grad.data + noise) * (1 - averagekernel_utility),
                        alpha=-2.0 * group["lr"],
                    )
                else:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                        (p.grad.data + noise) * (1 - scaled_utility),
                        alpha=-2.0 * group["lr"],
                    )
