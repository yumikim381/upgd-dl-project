from core.learner.learner import Learner
from core.optim.column_kernel_upgd.column_kernel_avg_optim import FirstOrderGlobalColumnKernelUPGD

class UPGD_ColumnKernelLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        try:
            print("inside kernel learner")
            optimizer = FirstOrderGlobalColumnKernelUPGD
            name = "upgd_kernel"
            super().__init__(name, network, optimizer, optim_kwargs)
        except Exception as e:
            print(e)
