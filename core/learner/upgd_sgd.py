from core.learner.learner import Learner
from core.optim.gradient_upgd.gradient_upgd import UPGD_SGD, UPGD_DynamicclippedGradient

class UPGD_SGD_Learner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = UPGD_SGD
        name = "UPGD_SGD"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGD_DynamicclippedGradient_Learner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = UPGD_DynamicclippedGradient
        name = "UPGD_DynamicclippedGradient"
        super().__init__(name, network, optimizer, optim_kwargs)