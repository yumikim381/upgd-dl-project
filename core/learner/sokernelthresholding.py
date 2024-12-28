from core.learner.learner import Learner
from core.optim.kernel_upgd_new_approach.second_order import SOKernelThresholding


class UPGDThresholdingLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        try:
            optimizer = SOKernelThresholding
            name = "upgd_thresholder"
            print("upgd_thresholder")
            super().__init__(name, network, optimizer, optim_kwargs,extend=True)
        except Exception as e:
            print(e)
            