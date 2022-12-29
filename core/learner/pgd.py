from core.learner.learner import Learner
from core.optim.perturbation.pgd import ExtendedPGD

class PGDLearner(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = ExtendedPGD
        name = "pgd"
        super().__init__(name, network, optimizer, optim_kwargs)