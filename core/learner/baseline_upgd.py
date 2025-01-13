from core.learner.learner import Learner
from core.optim.baseline_upgd.first_order import FirstOrderGlobalUPGD

class FirstOrderGlobalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderGlobalUPGD
        name = "upgd_fo_global"
        super().__init__(name, network, optimizer, optim_kwargs)
