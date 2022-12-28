from core.grid_search import GridSearch
from core.learner.upgd import UPGDv2NormalizedLearner
from core.learner.sgd import SGDLearner
from core.task.summer_with_signals_change import SummerWithSignalsChange
from core.network.fcn_tanh import FullyConnectedTanh
from core.runner import Runner
from core.run import Run
from core.utils import create_script_generator, create_script_runner


task = SummerWithSignalsChange()

grids = [
    GridSearch(
        seed=[i for i in range(0, 2)],
        # lr=[10 ** -i for i in range(0, 3)],
        lr=[0.01],
        beta_utility=[0.0],
        temp=[1.0],
        sigma=[1.0],
        network=[FullyConnectedTanh()],
        n_samples=[500000],
    ),
    GridSearch(seed=[i for i in range(0, 2)],
            #    lr=[10 ** -i for i in range(0, 3)],
               lr=[0.01],
               network=[FullyConnectedTanh()],
               n_samples=[500000],
    ),
]

learners = [
    UPGDv2NormalizedLearner(FullyConnectedTanh(), dict()),
    SGDLearner(FullyConnectedTanh(), dict()),
]

for learner, grid in zip(learners, grids):
    runner = Runner(Run, learner, grid, task, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{task.name}")
    create_script_runner(f"generated_cmds/{task.name}")