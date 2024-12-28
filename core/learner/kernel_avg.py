from core.learner.learner import Learner
from core.optim.kernel_upgd.first_order import FirstOrderGlobalKernelUPGD
from core.optim.kernel_upgd.second_order import SecondOrderGlobalKernelUPGD

class UPGD_KernelLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        try:
            print("inside kernel learner")
            optimizer = FirstOrderGlobalKernelUPGD
            name = "upgd_kernel"
            super().__init__(name, network, optimizer, optim_kwargs)
        except Exception as e:
            print(e)


class UPGD_SecondOrderKernelLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        try:
            print("inside kernel learner")
            optimizer = SecondOrderGlobalKernelUPGD
            name = "upgd_kernel"
            super().__init__(name, network, optimizer,optim_kwargs, extend=True)
        except Exception as e:
            print(e)
