# AutoDL for Load Forecasting

This is the code for the paper: *Automated Deep Learning for Load Forecasting*.


The data used is stored ins ```dataset/data.csv```. The target is called ```conso_rte```, the other variables are anonymized.

The main file is ```main.py```. You can run it with several arguments: 
* ```filename```: the data to use for your experimentations (here ```dataset/data.csv```)
* ```target```: the target name (here ```conso_rte```)
* ```mh```: the algorithm to use, autopytorch, random search or evolutionary algorithm
* ```save_dir```: where to save your experiments results
* ```seed```: general seed for your experiments

You can run the code with AutoPytorch without ```mpi4py``` but you must have it for EnergyDragon. A small example can be found ```small_test.ipynb```. The code to recreate the paper figures can be found ```plots_paper.ipynb```.

The ```dragon``` folder contains the code from the ```DRAGON``` package that we are using for EnergyDragon.

The ```dataset``` folder contains the data used for our experiments as well as the saved outputs from our baseline.

The ```utils``` folder contains our code. In the ```utils/config.py``` you can find our configurations for our use case: training procedure, features, etc. To incorporate the CNN/MLP model within the SSEA algorithm, please set ```'models' = True``` in the ```steady_state_config``` variable in the ```config.py```.

The experiments have be ran using a bash environment. Two bash files for AutoPytorch ```script_autopytorch.sh``` and for the Dragon optimization ```script_dragon.sh``` can be found in the repository. Yo will need an MPI environment to use the second one.


## Installation

The code works with Python version 3.9.18.

* Create a conda (or mamba) environment: ```conda create -n new_env python=3.9.18```
* Activate the environment: ```conda activate new_env```
* Install the zellij package using pip (be carefull to use the pip from your conda environment): ```pip install git+https://github.com/ThomasFirmin/zellij.git@dag```
* Install matplotlib: ```conda install matplotlib```
* Install AutoPytorch with pip: ```pip install autoPyTorch```
* Install Ipykernel (for the notebooks): ```conda install -c anaconda ipykernel```
* Add your environment as a jupyter kernel: ```python -m ipykernel install --user --name=new_env```
(If not already installed by also the previous installation: ```conda install graphviz```)

## Norway use case

The plots for the Norway use case can be recreated using the ```plots_norway.ipynb``` notebook. The necessary data file should be included in the ```dataset/save_norway``` folder.


