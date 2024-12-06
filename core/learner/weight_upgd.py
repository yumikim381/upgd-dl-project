from core.learner.learner import Learner
from core.optim.weight_upgd.first_order import FirstOrderLocalUPGD, FirstOrderGlobalUPGD
from core.optim.weight_upgd.second_order import SecondOrderLocalUPGD, SecondOrderGlobalUPGD

class FirstOrderLocalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderLocalUPGD
        name = "upgd_fo_local"
        super().__init__(name, network, optimizer, optim_kwargs)

class SecondOrderLocalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderLocalUPGD
        name = "upgd_so_local"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FirstOrderGlobalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderGlobalUPGD
        name = "upgd_fo_global"
        super().__init__(name, network, optimizer, optim_kwargs)

class SecondOrderGlobalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderGlobalUPGD
        name = "upgd_so_global"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)