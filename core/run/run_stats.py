import sys
import os
from dotenv import load_dotenv

from core.learner.learner import Learner

load_dotenv()

# Add the project root directory to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch, sys
from core.utils import tasks, networks, learners, criterions
from core.logger import Logger
from backpack import backpack, extend
sys.path.insert(1, os.getcwd())
from HesScale.hesscale import HesScale
import signal
import traceback
import time
from functools import partial
# import the library
from tqdm import tqdm


def signal_handler(msg, signal, frame):
    print('Exit signal: ', signal)
    cmd, learner = msg
    with open(f'timeout_{learner}.txt', 'a') as f:
        f.write(f"{cmd} \n")
    exit(0)

USER = "yumkim"

class RunStats:
    name = 'run_stats'
    def __init__(self, n_samples=10000, task=None, learner=None, save_path="logs", seed=0, network=None, **kwargs):
        self.n_samples = int(n_samples)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task = tasks[task]()
        self.task_name = task
        self.learner: Learner = learners[learner](networks[network], kwargs)
        print(f"self.learner is{self.learner}")
        self.logger = Logger(save_path)
        self.seed = int(seed)
        print("i am alive")
    def save_model(self, save_path=None):
        """Save the trained model weights."""
        if save_path is None:
            save_path = "/work/scratch"+USER+"/column_kernel_avg.pth"
        save_data = {
            "model_state_dict": self.learner.network.state_dict(),
            #"optimizer_state_dict": self.learner.optimizer(self.learner.parameters).state_dict(),
            #"optimizer_state_dict": self.learner.optimizer.state_dict(),
            "task_name": self.task_name,
            "learner_name": self.learner.name,
            "seed": self.seed,
            "n_samples": self.n_samples
        }
        torch.save(save_data, save_path)
        print(f"Model and optimizer states saved to {save_path}")

    def start(self):
        print("starting")
        losses_per_task = []
        plasticity_per_task = []
        n_dead_units_per_task = []
        weight_rank_per_task = []
        weight_l2_per_task = []
        weight_l1_per_task = []
        grad_l2_per_task = []
        grad_l1_per_task = []
        grad_l0_per_task = []

        if self.task.criterion == 'cross_entropy':
            accuracy_per_task = []
        self.learner.set_task(self.task)
        if self.learner.extend:    
            extension = HesScale()
            #extension.set_module_extension(GateLayer, GateLayerGrad())
        criterion = extend(criterions[self.task.criterion]()) if self.learner.extend else criterions[self.task.criterion]()
        if self.learner.storeActivations:
            print("activations")
            print(self.learner.network.activations_out)

            optimizer = self.learner.optimizer(
                self.learner.parameters, self.learner.network, **self.learner.optim_kwargs
            )
        else:
            optimizer = self.learner.optimizer(
                self.learner.parameters, **self.learner.optim_kwargs
            )
            
        losses_per_step = []
        plasticity_per_step = []
        n_dead_units_per_step = []
        weight_rank_per_step = []
        weight_l2_per_step = []
        weight_l1_per_step = []
        grad_l2_per_step = []
        grad_l1_per_step = []
        grad_l0_per_step = []

        if self.task.criterion == 'cross_entropy':
            accuracy_per_step = []
        with tqdm(total=self.n_samples, desc="Training Progress", unit="step") as pbar:
            for i in range(self.n_samples):
                input, target = next(self.task)
                input, target = input.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.learner.predict(input)
                loss = criterion(output, target)
                if self.learner.extend:
                    with backpack(extension):
                        loss.backward()
                else:
                    loss.backward()
                optimizer.step()

                losses_per_step.append(loss.item())
                if self.task.criterion == 'cross_entropy':
                    accuracy_per_step.append((output.argmax(dim=1) == target).float().mean().item())

                # compute some statistics after each task change
                with torch.no_grad():
                    output_new = self.learner.predict(input)
                    loss_after = criterion(output_new, target)
                    loss_before = torch.clamp(loss, min=1e-8)
                    plasticity_per_step.append(torch.clamp((1-loss_after/loss_before), min=0.0, max=1.0).item())
                n_dead_units = 0
                #get number of dead units 
                for _, value in self.learner.network.activations.items():
                    n_dead_units += value
                n_dead_units_per_step.append(n_dead_units / self.learner.network.n_units)

                sample_weight_rank = 0.0
                sample_max_rank = 0.0
                sample_weight_l2 = 0.0
                sample_grad_l2 = 0.0
                sample_weight_l1 = 0.0
                sample_grad_l1 = 0.0
                sample_grad_l0 = 0.0
                sample_n_weights = 0.0

                for name, param in self.learner.network.named_parameters():
                    if 'weight' in name:
                        if 'conv' in name:
                            sample_weight_rank += torch.torch.linalg.matrix_rank(param.data).float().mean()
                            sample_max_rank += torch.min(torch.tensor(param.data.shape)[-2:])
                        else:
                            sample_weight_rank += torch.linalg.matrix_rank(param.data)
                            sample_max_rank += torch.min(torch.tensor(param.data.shape))
                        sample_weight_l2 += torch.norm(param.data, p=2) ** 2
                        sample_weight_l1 += torch.norm(param.data, p=1)

                        sample_grad_l2 += torch.norm(param.grad.data, p=2) ** 2
                        sample_grad_l1 += torch.norm(param.grad.data, p=1)

                        sample_grad_l0 += torch.norm(param.grad.data, p=0)
                        sample_n_weights += torch.numel(param.data)

                weight_l2_per_step.append(sample_weight_l2.sqrt().item())
                weight_l1_per_step.append(sample_weight_l1.item())
                grad_l2_per_step.append(sample_grad_l2.sqrt().item())
                grad_l1_per_step.append(sample_grad_l1.item())
                grad_l0_per_step.append(sample_grad_l0.item()/sample_n_weights)
                weight_rank_per_step.append(sample_weight_rank.item() / sample_max_rank.item())


                if i % self.task.change_freq == 0:
                    losses_per_task.append(sum(losses_per_step) / len(losses_per_step))
                    if self.task.criterion == 'cross_entropy':
                        accuracy_per_task.append(sum(accuracy_per_step) / len(accuracy_per_step))
                    plasticity_per_task.append(sum(plasticity_per_step) / len(plasticity_per_step))
                    n_dead_units_per_task.append(sum(n_dead_units_per_step) / len(n_dead_units_per_step))
                    weight_rank_per_task.append(sum(weight_rank_per_step) / len(weight_rank_per_step))
                    weight_l2_per_task.append(sum(weight_l2_per_step) / len(weight_l2_per_step))
                    weight_l1_per_task.append(sum(weight_l1_per_step) / len(weight_l1_per_step))
                    grad_l2_per_task.append(sum(grad_l2_per_step) / len(grad_l2_per_step))
                    grad_l1_per_task.append(sum(grad_l1_per_step) / len(grad_l1_per_step))
                    grad_l0_per_task.append(sum(grad_l0_per_step) / len(grad_l0_per_step))

                    losses_per_step = []
                    if self.task.criterion == 'cross_entropy':
                        accuracy_per_step = []
                    plasticity_per_step = []
                    n_dead_units_per_step = []
                    weight_rank_per_step = []
                    weight_l2_per_step = []
                    weight_l1_per_step = []
                    grad_l2_per_step = []
                    grad_l1_per_step = []
                    grad_l0_per_step = []
                if i == 10:
                    self.logger.log(losses=losses_per_task,
                                            accuracies=accuracy_per_task,
                                            plasticity_per_task=plasticity_per_task,
                                            task=self.task_name, 
                                            learner=self.learner.name,
                                            network=self.learner.network.name,
                                            optimizer_hps=self.learner.optim_kwargs,
                                            n_samples=self.n_samples,
                                            seed=self.seed,
                                            n_dead_units_per_task=n_dead_units_per_task,
                                            weight_rank_per_task=weight_rank_per_task,
                                            weight_l2_per_task=weight_l2_per_task,
                                            weight_l1_per_task=weight_l1_per_task,
                                            grad_l2_per_task=grad_l2_per_task,
                                            grad_l0_per_task=grad_l0_per_task,
                                            grad_l1_per_task=grad_l1_per_task,
                            )
                if i % 100000 == 0 and i != 0:
                    try:
                        path = f"/work/scratch/{USER}/model_{self.learner.name}_{self.task_name}_{i}.pth"
                        self.save_model(path)
                        print(f"Saving the results into {path}")
                        if i == 100000 or i == 200000 or i == 400000 or i == 600000 or i == 800000:
                            self.logger.log(losses=losses_per_task,
                                            accuracies=accuracy_per_task,
                                            plasticity_per_task=plasticity_per_task,
                                            task=self.task_name, 
                                            learner=self.learner.name,
                                            network=self.learner.network.name,
                                            optimizer_hps=self.learner.optim_kwargs,
                                            n_samples=self.n_samples,
                                            seed=self.seed,
                                            n_dead_units_per_task=n_dead_units_per_task,
                                            weight_rank_per_task=weight_rank_per_task,
                                            weight_l2_per_task=weight_l2_per_task,
                                            weight_l1_per_task=weight_l1_per_task,
                                            grad_l2_per_task=grad_l2_per_task,
                                            grad_l0_per_task=grad_l0_per_task,
                                            grad_l1_per_task=grad_l1_per_task,
                            )
                            
                    except Exception as e:
                        print(f"Failed to save the model: {e}")
                    
                pbar.update(1)

            self.save_model(f"/work/scratch/{USER}/model_{self.learner.name}_{self.task_name}_final.pth")
            if self.task.criterion == 'cross_entropy':
                self.logger.log(losses=losses_per_task,
                                accuracies=accuracy_per_task,
                                plasticity_per_task=plasticity_per_task,
                                task=self.task_name, 
                                learner=self.learner.name,
                                network=self.learner.network.name,
                                optimizer_hps=self.learner.optim_kwargs,
                                n_samples=self.n_samples,
                                seed=self.seed,
                                n_dead_units_per_task=n_dead_units_per_task,
                                weight_rank_per_task=weight_rank_per_task,
                                weight_l2_per_task=weight_l2_per_task,
                                weight_l1_per_task=weight_l1_per_task,
                                grad_l2_per_task=grad_l2_per_task,
                                grad_l0_per_task=grad_l0_per_task,
                                grad_l1_per_task=grad_l1_per_task,
                )
            else:
                self.logger.log(losses=losses_per_task,
                                plasticity=plasticity_per_task,
                                task=self.task_name,
                                learner=self.learner.name,
                                network=self.learner.network.name,
                                optimizer_hps=self.learner.optim_kwargs,
                                n_samples=self.n_samples,
                                seed=self.seed,
                                n_dead_units_per_task=n_dead_units_per_task,
                                weight_rank_per_task=weight_rank_per_task,
                                weight_l2_per_task=weight_l2_per_task,
                                weight_l1_per_task=weight_l1_per_task,
                                grad_l2_per_task=grad_l2_per_task,
                                grad_l0_per_task=grad_l0_per_task,
                                grad_l1_per_task=grad_l1_per_task,
                )


if __name__ == "__main__":
    # start the run using the command line arguments
    ll = sys.argv[1:]
    args = {k[2:]:v for k,v in zip(ll[::2], ll[1::2])}
    run = RunStats(**args)
    cmd = f"python3 {' '.join(sys.argv)}"
    signal.signal(signal.SIGUSR1, partial(signal_handler, (cmd, args['learner'])))
    current_time = time.time()
    try:
        run.start()
        with open(f"finished_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} time_elapsed: {time.time()-current_time} \n")
    except Exception as e:
        with open(f"failed_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} \n")
        with open(f"failed_{args['learner']}_msgs.txt", "a") as f:
            f.write(f"{cmd} \n")
            f.write(f"{traceback.format_exc()} \n\n")