import sys, os
from torch.nn import functional as F
sys.path.insert(1, os.getcwd())
from HesScale.hesscale import HesScale
import torch

class SOKernelThresholding(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, method_field=type(self).method.savefield, names=names)
        super(SOKernelThresholding, self).__init__(params, defaults)

    def step(self):
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                avg_utility = state["avg_utility"]
                try:
                   
                    hess_param = getattr(p, group["method_field"])
                except Exception as e:
                    print(e)
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(
                    utility, alpha=1 - group["beta_utility"]
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
                noise = torch.randn_like(p.grad) * group["sigma"]
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)
                if len(scaled_utility.shape) == 4: # We are in convolutional layer:
                    
                    #HYPER PARAMETEER 
                    LOW_THRESHOLD =0.1
                    FRACTION_LOW_THRESHOLD =0.8
                    EXPOENTIONAL_AVERAGE_FACTOR = 0.5
                    try:
                        low_mask = (scaled_utility < LOW_THRESHOLD).float()
                        # average along the spatial dimensions [2, 3]
                        fraction_low = low_mask.mean(dim=(2, 3))
                        is_x_percent_low_2d = (fraction_low >= FRACTION_LOW_THRESHOLD)
                        int_mask_2d = (~is_x_percent_low_2d).int()
                        #4D mask (one value per kernel, repeated across the spatial dims), do
                        is_x_percent_low_4d = int_mask_2d.unsqueeze(-1).unsqueeze(-1).expand(
                            -1, -1, scaled_utility.size(2), scaled_utility.size(3)
                        )

                        indices_out, indices_in = torch.where(is_x_percent_low_2d)
                        for oc, ic in zip(indices_out.tolist(), indices_in.tolist()):
                            print(f"Kernel at out_channel={oc}, in_channel={ic} has >= 90% low utility values.")
                        
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
                        utility_new = is_x_percent_low_4d.float()* EXPOENTIONAL_AVERAGE_FACTOR + scaled_utility*(1-EXPOENTIONAL_AVERAGE_FACTOR)
                        p.data.mul_(1 - var1).add_(
                                (p.grad.data + noise) * (1-utility_new),
                                alpha=alphavar,
                            )
                    except Exception as e:
                        print(e)
                        
                else:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                        (p.grad.data + noise) * (1-scaled_utility),
                        alpha=-2.0*group["lr"],
                    )
                # p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                #     (p.grad.data + noise) * (1 - scaled_utility), alpha=-group["lr"]
                # )
                