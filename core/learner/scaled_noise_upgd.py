from core.learner.learner import Learner
from core.optim.weight_upgd_scaled_noise.scaled_noise import UPGDScaledWeightNormNoise,UPGDScaledGradNormNoise,UPGDScaledAdativeNormNoise


class UPGDScaledWeightNormNoiseLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = UPGDScaledWeightNormNoise
        name = "UPGDScaledWeightNormNoiseLearner"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDScaledGradNormNoiseLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = UPGDScaledGradNormNoise
        name = "UPGDScaledGradNormNoiseLearner"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDScaledAdativeNormNoiseDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = UPGDScaledAdativeNormNoise
        name = "UPGDScaledAdativeNormNoiseDLearner"
        super().__init__(name, network, optimizer, optim_kwargs)
