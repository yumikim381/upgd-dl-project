import torch
from torch.nn import functional as F

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
                # Maintain the running average of the utility: state["avg_utility"].
                state = self.state[p]
                # When the optimizer encounters a parameter p for the first time, it initializes the state for that paramete
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                #  Maintains and updates a running average of a utility metric for each parameter.
                avg_utility = state["avg_utility"]
                # if (len(avg_utility.shape) == 4):
                    # print("In convolutional layer")
                    # print(p.grad.data.shape)
                    # raise Exception("Pause")
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
                    
                    avg = scaled_utility.mean(dim=[2, 3])  # avg shape: [out_channels, in_channels]

                    # Step 2: Inflate back to original shape
                    # First, add back the spatial dims
                    avg_expanded = avg.unsqueeze(-1).unsqueeze(-1)  # shape: [out_channels, in_channels, 1, 1]

                    # Now expand along the spatial dimensions
                    averagekernel_utility = avg_expanded.expand(-1, -1, scaled_utility.size(2), scaled_utility.size(3)) 
                        
                    try:
                        var1 = group["lr"] * group["weight_decay"]                        
                    except Exception as e:
                        print(e)
                        raise e
                    #TODO Delete the following
                
                    import os
                    from datetime import datetime, timedelta

                    path = "/work/scratch/cpinkl/kernelLogs"
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Function to check the last modified time of the latest file in the directory
                    # Check if the directory exists and is not empty
                    if os.path.exists(path):
                        # Get all files in the directory with their modification times
                        files = [os.path.join(path, f) for f in os.listdir(path)]
                        if len(files) > 0:
                            latest_file = max(files, key=os.path.getmtime)  # Get the most recently modified file
                            last_modified_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
                        else: 
                            last_modified_time = datetime.now()
                        print(last_modified_time)
                        # Check if the last file was written more than the time limit ago
                        if datetime.now() - last_modified_time > timedelta(minutes=1) or len(files) == 0:
                            filename = f"utility_{timestamp}.pt"
                            file_path = os.path.join(path, filename)
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                            torch.save({"avg_utility": averagekernel_utility, "param_utility": scaled_utility}, file_path)
                            print(f"Saved new file: {file_path}")
                   
                    # delete til here

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