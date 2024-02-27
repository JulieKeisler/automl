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
