import os
import torch
import numpy as np
import pprint
from dragon.utils.tools import logger
import tempfile
from dataset.dataloader import LoadDataLoader
from utils.config import exp_configs, mh_configs
from utils.trainer import Trainer
import torch.nn as nn
from dragon.search_space.bricks.basics import Identity, MLP
from dragon.search_space.dags import AdjMatrix, Node
from dragon.search_space.bricks.convolutions import Conv1d
from dragon.search_space.bricks.pooling import AVGPooling1D

def generate_old_models(config):
    name_features = ['f_4', 'f_6', 'f_7', 'f_10', 'f_11', 'f_13', 'f_15', 'f_20', 'f_23', 'f_24', 'f_25', 'f_28', 'f_29']
    features = [5 if f in name_features  else -5 for f in config['Features']]
    m1 = np.array([[0,1],[0,0]])
    n1 = [Node(combiner="add", name=Identity, hp={}, activation=nn.Identity()),
        Node(combiner="add", name=Identity, hp={}, activation=nn.Identity())]

    m2 = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0]])
    n2 = [Node(combiner="add", name=Identity, hp={}, activation=nn.Identity()), 
        Node(combiner="add", name=MLP, hp={"out_channels":50}, activation=nn.ReLU()),
        Node(combiner="add", name=Conv1d, hp={"kernel_size": 48, "out_channels": 10, "padding": "same"}, activation=nn.ReLU()),
        Node(combiner="add", name=AVGPooling1D, hp={"pool_size": 5}, activation=nn.Identity()),
        Node(combiner="add", name=Conv1d, hp={"kernel_size": 12, "out_channels": 12, "padding": "same"}, activation=nn.ReLU()),
        Node(combiner="add", name=AVGPooling1D, hp={"pool_size": 5}, activation=nn.Identity()),
        Node(combiner="concat", name=MLP, hp={"out_channels":150}, activation=nn.ReLU()),
        Node(combiner="add", name=MLP, hp={"out_channels":50}, activation=nn.ReLU())

    ]

    neural_net = [features, AdjMatrix(n1, m1), AdjMatrix(n2,m2),
              Node(combiner="add", name=MLP, hp={"out_channels": 1}, activation=nn.Identity()), 10]

    return [neural_net]


def fit_energy_dragon(config):
    exp_config = exp_configs[config['config']]
    config.update(exp_config)  
    mh_config = mh_configs[config['mh']]
    config.update(mh_config)

    config['SaveDir'] += config["save_dir"]

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    p = comm.Get_size()
    status = MPI.Status()
    n_gpu = torch.cuda.device_count()
    num_gpu = "cuda:" + str((rank - 1) % n_gpu)
    if n_gpu > 0:
        config["Device"] = num_gpu

    if rank == 0: #type: ignore
        save_dir= f"{config['SaveDir']}/{tempfile.NamedTemporaryFile().name}"
        for r in range(1, p):#type: ignore
            comm.send(dest=r, tag=0, obj=save_dir)#type: ignore
        logger.info(f'SaveDir = {save_dir}')
    else:
        save_dir = comm.recv(source=0, tag=0, status=status) #type: ignore
    config['SaveDir'] = save_dir
    if rank ==0:
        config_str = pprint.pformat(config)
        logger.info(f'Final config:\n{config_str}')

    logger.info(f'This is processus {rank} on {MPI.Get_processor_name()} using {num_gpu}')
    if not os.path.exists(config["SaveDir"]):
        os.makedirs(config["SaveDir"], exist_ok=True)

    data_loader = LoadDataLoader(config)
    trainer = Trainer(data_loader)
    sp = config['SearchSpace'](config)
    trainer.labels = [e.label for e in sp]
    config['Seed'] = 0
    if mh_config['models']:
        mh_config['models'] = generate_old_models(config)
    search_algorithm = mh_config['MetaHeuristic'](sp, config=config, evaluation=trainer.train_and_test, **mh_config)
    search_algorithm.run()