from core.learner.learner import Learner
from core.optim.upgd_kernelPruning.fist_order import FirstOrderGlobalKernelPruningUPGD

class KernelPruning_UPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        try:
            optimizer = FirstOrderGlobalKernelPruningUPGD
            name = "kernel_pruning_upgd"
            super().__init__(name, network, optimizer, optim_kwargs, storeActivations=True)
        except Exception as e:
            print(e)