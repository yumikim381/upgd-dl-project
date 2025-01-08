from core.learner.learner import Learner
from core.optim.kernel_upgd.first_order import FirstOrderGlobalKernelUPGD

class UPGD_KernelLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        try:
            optimizer = FirstOrderGlobalKernelUPGD
            name = "upgd_kernel"
            super().__init__(name, network, optimizer, optim_kwargs)
        except Exception as e:
            print(e)
