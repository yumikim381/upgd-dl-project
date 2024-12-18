from core.learner.learner import Learner
from core.optim.kernel_avg_optim import UPGD_Kernel

class UPGD_KernelLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        try:
            print("inside kernel learner")
            optimizer = UPGD_Kernel
            name = "upgd_kernel"
            super().__init__(name, network, optimizer, optim_kwargs)
        except Exception as e:
            print(e)
