from zellij.strategies.tools.cooling import MulExponential
from torch import nn
from dragon.search_algorithm.variation_operators import DAGTwoPoint
#from utils.ssea import SteadyStateEA
from utils.callbacks import Checkpoint, EarlyStopping, EndOnNan, TrainingDuration
from utils.metamodels import LoadForecastingNetwork
from utils.optimizers import cyclical_lr
from utils.searchspace import load_search_space
from utils.metrics import MAPE

steady_state_config = {
    #"MetaHeuristic": SteadyStateEA,
    "n_iterations": 4000,
    "population_size": 400,
    "selection_size": 20,
    "crossover": DAGTwoPoint(),
    "Neighborhood": "Full",
    "models": False
}

rs_config = {
    #"MetaHeuristic": SteadyStateEA,
    "n_iterations": 4000,
    "population_size": 4000,
    "selection_size": 20,
    "crossover": DAGTwoPoint(),
    "Neighborhood": "Full",
    "models": False
}


automl_config = {
    # Main Config
    "SaveDir": "dataset/save/",
    "SearchSpace": load_search_space,
    "Model": LoadForecastingNetwork,
    "Device": "cpu",
    # Data
    "Borders": {
        "Border1s": ["2015-03-01 00:00:00", "2015-03-01 00:00:00", "2019-03-01 00:00:00", "2015-03-01 00:00:00"],
        "Border2s": ["2019-02-28 23:30:00", "2019-02-28 23:30:00", "2020-02-28 23:30:00", "2019-02-28 23:30:00"]
    },
    "ValData": False,
    "Freq": 48,
    #Features Selection
    "Features": [f'f_{i}' for i in range(31)],
    "GradientFeatures": True,
    "SimpleAfterFeatures": True,
    "MaxFeaturesEpochs": 500,
    "MinFeaturesEpochs": 20,
    "FeaturesOptimizer": {
        "Optimizer": "AdamW",
        "LearningRate": 0.01
    },
    "L1Eps": 10e-10,
    "FeaturesCallbacks": [
        Checkpoint(save_dir="/checkpoints"),
        EarlyStopping(patience=200),
        EndOnNan(),
        TrainingDuration(15),
    ],
    #Objective function and trainer
    "Criterium": MAPE,
    "Ep": 200,
    "BatchSize": 32,
    "WeightsCallbacks": [
        Checkpoint(save_dir="/snapshots", save_top_k=3, max_checkpoints=10, max_epoch=200),
        EndOnNan(),
        TrainingDuration(15),
    ],
    
    "WeightsOptimizer": {
        "Optimizer": "SGD",
        "Scheduler": lambda it: cyclical_lr(it, 10, 200, 0.01),
        "LearningRate": 1,
    },
    "Loss": nn.MSELoss(),
    # Prediction
    "OnlinePrediction": False,
}


exp_configs = {
    "automl_config": automl_config
}

mh_configs = {
    "SSEA": steady_state_config,
    "RS": rs_config
}