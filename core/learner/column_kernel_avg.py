from core.learner.learner import Learner
#from core.optim.kernel_upgd.first_order import FirstOrderGlobalKernelUPGD #wrong?
from core.optim.column_kernel_avg_optim import UPGD_ColumnKernel #correct?

class UPGD_ColumnKernelLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        try:
            print("inside kernel learner")
            optimizer = UPGD_ColumnKernel
            name = "upgd_kernel"
            super().__init__(name, network, optimizer, optim_kwargs)
        except Exception as e:
            print(e)
