from core.learner.learner import Learner
from core.optim.kernel_avg import UPGD_Kernel

class UPGD_KernelLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = UPGD_Kernel
        name = "upgd_kernel"
        super().__init__(name, network, optimizer, optim_kwargs)