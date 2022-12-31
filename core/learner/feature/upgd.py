from core.learner.learner import Learner
from core.optim.feature.search.first_order import FirstOrderSearchAntiCorrNormalized

class FeatureUPGDv2LearnerFONormalized(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderSearchAntiCorrNormalized
        name = "feature_upgdv2_fo_anticorr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)
