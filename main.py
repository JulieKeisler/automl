import argparse
import pprint

from dragon.utils.tools import set_seed, logger

from main_autopytorch import fit_autopytorch
from main_energy_dragon import fit_energy_dragon
from utils.config import mh_configs, exp_configs


if __name__ == "__main__":

    ############ GENERAL SETUP ############

    # Args parser for experiment
    parser = argparse.ArgumentParser(description='DNN optimization for load forecasting')
    parser.add_argument('--filename', type=str, required=True, help='Name of the file containing the data.')
    parser.add_argument('--target', type=str, required=True, help='Name of the target in the data file.')
    parser.add_argument('--config', type=str, required=True, choices=list(exp_configs.keys()), help=f"Configuration for your experimentation. If yours is not one of {list(exp_configs.keys())}, check the documentation.")
    parser.add_argument('--mh', type=str, required=False, default="GA", choices=list(mh_configs.keys())+['autopytorch'], help=f"MetaHeuristic to use if we perform an optimization. Should be one of: {list(mh_configs.keys())}")
    parser.add_argument('--save_dir', type=str, required=True, default=None, help="Name of saving directory.")

    parser.add_argument('--seed', type=int, required=False, default=0, help='General seed for the experiment reproducibility')
    # MPI
    parser.add_argument('--MPI', action="store_true")
    parser.add_argument('--no-MPI', dest='MPI', action="store_false")
    parser.set_defaults(MPI=True)

    args = parser.parse_args()
    # Set seed for reproducibility
    set_seed(args.seed)

    #######################################
    args = vars(args)

    if args['mh'] == 'autopytorch':
        fit_autopytorch(args)
    else:
        fit_energy_dragon(args)



