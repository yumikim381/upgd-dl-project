from core.learner.learner import Learner
from core.optim.kernel_convex.first_order import KernelConvexCombi

class KernelConvexCombiLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        try:
            print("inside kernel learner")
            optimizer = KernelConvexCombi
            name = "KernelConvexCombi"
            super().__init__(name, network, optimizer, optim_kwargs, storeActivations=True)
        except Exception as e:
            print(e)