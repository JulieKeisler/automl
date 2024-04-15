from copy import deepcopy
import os
import pickle
import random
import shutil
from dragon.utils.tools import set_seed, logger
import numpy as np
import random
import torch
from mpi4py import MPI


class SteadyStateEA:
    def __init__(self, search_space, n_iterations: int, population_size: int, selection_size: int, evaluation, crossover, config, models, **args):
        self.search_space=search_space
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.evaluation = evaluation
        self.selection_size = selection_size
        self.crossover = crossover
        self.config = config
        self.comm = MPI.COMM_WORLD
        self.status = MPI.Status()
        self.p_name = MPI.Get_processor_name()
        self.rank = self.comm.Get_rank()
        self.p = self.comm.Get_size()
        if models != False:
            self.models = models
        else:
            self.models = []

    def create_first_population(self):
        population = self.search_space.random(size=self.population_size-len(self.models))
        if len(self.models)>0:
            population += self.models
        logger.info(f'The whole population has been created (size = {len(population)}), models = {len(self.models)}')
        return population

    def evaluate_first_population(self, population):
        storage = {}
        nb_send = 0
        min_loss = np.inf
        # Dynamically send and evaluate the first population
        logger.info(f'We start by evaluating the whole population (size={len(population)})')
        while (nb_send< len(population)) and (nb_send < self.p-1):
            x = population[nb_send]
            logger.info(f'Sending individual {nb_send} to processus {nb_send+1} < {self.p}')
            self.comm.send(dest=nb_send+1, tag=0, obj=("evaluate", x, nb_send))
            nb_send +=1
        nb_receive = 0
        while nb_send < len(population):
            loss, x_path, idx = self.comm.recv(source=MPI.ANY_SOURCE, tag=0, status=self.status)
            source = self.status.Get_source()
            try:
                shutil.copy(x_path+ "/x.pkl", f"{self.config['SaveDir']}/x_{idx}.pkl")
                storage[idx] = {"Individual": f"{self.config['SaveDir']}/x_{idx}.pkl", "Loss": loss}
                if loss < min_loss:
                    logger.info(f"Best found ! {loss} < {min_loss}")
                    min_loss = loss
                    if len(os.listdir(x_path))>1:
                        try:
                            shutil.copytree(x_path, self.config["SaveDir"]+"/best_model/")
                        except FileExistsError:
                            shutil.rmtree(self.config["SaveDir"]+"/best_model/")
                            shutil.copytree(x_path, self.config["SaveDir"]+"/best_model/")
            except torch.cuda.OutOfMemoryError:
                logger.error(f'Failed to load x, idx= {idx}.')
            shutil.rmtree(x_path)
            x = population[nb_send]
            logger.info(f'Sending individual {nb_send} to processus {source}')
            self.comm.send(dest=source, tag=0, obj=("evaluate", x, nb_send))
            nb_send+=1
            nb_receive +=1
        while nb_receive < len(population):
            loss, x_path, idx = self.comm.recv(
                source=MPI.ANY_SOURCE, tag=0, status=self.status
            )
            source = self.status.Get_source()
            try:
                shutil.copy(x_path + "/x.pkl", f"{self.config['SaveDir']}/x_{idx}.pkl")
                storage[idx] = {"Individual": f"{self.config['SaveDir']}/x_{idx}.pkl", "Loss": loss}
                if loss < min_loss:
                    logger.info(f"Best found ! {loss} < {min_loss}")
                    min_loss = loss
                    if len(os.listdir(x_path))>1:
                        try:
                            shutil.copytree(x_path, self.config["SaveDir"]+"/best_model/")
                        except FileExistsError:
                            shutil.rmtree(self.config["SaveDir"]+"/best_model/")
                            shutil.copytree(x_path, self.config["SaveDir"]+"/best_model/")
            except torch.cuda.OutOfMemoryError:
                logger.error(f'Failed to load x')
            shutil.rmtree(x_path)
            nb_receive+=1
        logger.info(f"All models have been at least evaluated once, t = {len(population)} < {self.n_iterations}.")
        return storage, min_loss

    def selection(self, storage, K):
        logger.info(f"Storage: {len(storage)}")
        selection = [random.choice(list(storage.keys())) for i in range(min(self.selection_size, len(storage)))]
        best_1 = selection[np.argmin([storage[i]['Loss'] for i in selection])]
        parent1 = storage.pop(best_1)
        selection = [random.choice(list(storage.keys())) for i in range(min(self.selection_size, len(storage)))]
        best_2 = selection[np.argmin([storage[i]['Loss'] for i in selection])]
        parent2 = storage[best_2]
        storage[best_1] = parent1
        with open(parent1['Individual'], 'rb') as f:
            x1 = pickle.load(f)
        with open(parent2['Individual'], 'rb') as f:
            x2 = pickle.load(f)
        offspring_1, offspring_2 = deepcopy(x1), deepcopy(x2)
        self.crossover(offspring_1, offspring_2)
        offspring_1 = self.search_space.neighbor(deepcopy(offspring_1))
        offspring_2 = self.search_space.neighbor(deepcopy(offspring_2))
        logger.info(f"Evolving {best_1} and {best_2} to {K+1} and {K+2}")
        return offspring_1, offspring_2   

    def run(self):
        rank = self.comm.Get_rank()
        set_seed(self.config['Seed'])
        if rank == 0:
            logger.info(f"Master here ! start steady state EA algorithm.")

            ### Create first population
            population = self.create_first_population()

            ### Evaluate first population
            storage, min_loss  = self.evaluate_first_population(population) # type: ignore

            ### Start evolution
            t = len(population)
            K = len(population)

            # Store individuals waiting for a free processus
            to_evaluate = {}
            nb_send = 0

            # Send first offspring to all processus
            while nb_send < self.p-1:
                if len(to_evaluate) == 0:
                    selected = False
                    while not selected:
                        try:
                            offspring_1, offspring_2 = self.selection(storage, K)
                            selected = True
                        except torch.cuda.OutOfMemoryError:
                            pass
                    with open(f"{self.config['SaveDir']}/x_{K+1}.pkl", 'wb') as f:
                        pickle.dump(offspring_1, f)
                    with open(f"{self.config['SaveDir']}/x_{K+2}.pkl", 'wb') as f:
                        pickle.dump(offspring_2, f)
                    del offspring_1
                    del offspring_2
                    to_evaluate[K+1] = f"{self.config['SaveDir']}/x_{K+1}.pkl"
                    to_evaluate[K+2] = f"{self.config['SaveDir']}/x_{K+2}.pkl"
                    K+=2
                idx = list(to_evaluate.keys())[0]
                logger.info(f'Master, sending individual to processus {idx}')
                self.comm.send(dest=nb_send+1, tag=0, obj=("evaluate", to_evaluate[idx], idx))
                del to_evaluate[idx]
                nb_send+=1
            
            # dynamically receive and send evaluations
            while t < (self.n_iterations-self.p-1):
                loss, x_path, idx = self.comm.recv(source=MPI.ANY_SOURCE, tag=0, status=self.status)
                t+=1
                source = self.status.Get_source()
                try:
                    shutil.copy(x_path + "/x.pkl", f"{self.config['SaveDir']}/x_{idx}.pkl")
                    idx_max_loss = list(storage.keys())[np.argmax([storage[i]['Loss'] for i in storage.keys()])]

                    if loss < storage[idx_max_loss]['Loss']:
                        storage.pop(idx_max_loss)
                        logger.info(f'Replacing {idx_max_loss} by {idx}')
                        storage[idx] = {"Individual": f"{self.config['SaveDir']}/x_{idx}.pkl", "Loss": loss}
                        os.remove(f"{self.config['SaveDir']}/x_{idx_max_loss}.pkl")

                    if loss < min_loss:
                        logger.info(f"Best found! {loss} < {min_loss}")
                        min_loss = loss
                        if len(os.listdir(x_path))>1:
                            try:
                                shutil.copytree(x_path, self.config["SaveDir"]+"/best_model/")
                            except FileExistsError:
                                shutil.rmtree(self.config["SaveDir"]+"/best_model/")
                                shutil.copytree(x_path, self.config["SaveDir"]+"/best_model/")
                except torch.cuda.OutOfMemoryError:
                    logger.error(f'Failed to load x.')
                shutil.rmtree(x_path)

                if len(to_evaluate) == 0:
                    selected = False
                    while not selected:
                        try:
                            offspring_1, offspring_2 = self.selection(storage, K)
                            selected = True
                        except torch.cuda.OutOfMemoryError:
                            pass
                    with open(f"{self.config['SaveDir']}/x_{K+1}.pkl", 'wb') as f:
                        pickle.dump(offspring_1, f)
                    with open(f"{self.config['SaveDir']}/x_{K+2}.pkl", 'wb') as f:
                        pickle.dump(offspring_2, f)
                    del offspring_1
                    del offspring_2
                    to_evaluate[K+1] = f"{self.config['SaveDir']}/x_{K+1}.pkl"
                    to_evaluate[K+2] = f"{self.config['SaveDir']}/x_{K+2}.pkl"
                    K+=2
                idx = list(to_evaluate.keys())[0]
                logger.info(f'Master, sending individual to processus {idx}.')
                self.comm.send(dest=source, tag=0, obj=("evaluate", to_evaluate[idx], idx))
                del to_evaluate[idx]
                nb_send+=1
                
            nb_receive = 0
            # Receive last evaluation
            while (nb_receive < self.p-1):
                loss, x_path, idx = self.comm.recv(source=MPI.ANY_SOURCE, tag=0, status=self.status)
                nb_receive +=1
                source = self.status.Get_source()
                try:
                    shutil.copy(x_path + "/x.pkl", f"{self.config['SaveDir']}/x_{idx}.pkl")
                    idx_max_loss = list(storage.keys())[np.argmax([storage[i]['Loss'] for i in storage.keys()])]

                    if loss < storage[idx_max_loss]['Loss']:
                        storage.pop(idx_max_loss)
                        shutil.rmtree(x_path)     
                        logger.info(f'Replacing {idx_max_loss} by {idx}')
                        storage[idx] = {"Individual": f"{self.config['SaveDir']}/x_{idx}.pkl", "Loss": loss}
                        os.remove(f"{self.config['SaveDir']}/x_{idx_max_loss}.pkl")

                    if loss < min_loss:
                        logger.info(f"Best found! {loss} < {min_loss}")
                        min_loss = loss
                        if len(os.listdir(x_path))>1:
                            try:
                                shutil.copytree(x_path, self.config["SaveDir"]+"/best_model/")
                            except FileExistsError:
                                shutil.rmtree(self.config["SaveDir"]+"/best_model/")
                                shutil.copytree(x_path, self.config["SaveDir"]+"/best_model/")
                except torch.cuda.OutOfMemoryError:
                    logger.error(f'Failed to load x.')
                shutil.rmtree(x_path)
                        
            logger.info(f"Steady-State GA ending: min loss = {min_loss}")
            for i in range(1, self.p):
                self.comm.send(dest=i, tag=0, obj=(None,None, None))
            return min_loss
        else:
            logger.info(f"Worker {rank} here.")
            stop = True
            while stop:
                action, x, idx = self.comm.recv(source=0, tag=0, status=self.status)
                if action != None:
                    if action == "evaluate":
                        if isinstance(x, str):
                            x_path = x
                            with open(x_path, 'rb') as f:
                                x = pickle.load(f)
                            os.remove(x_path)
                        loss, model = self.evaluation(x, idx=idx)
                        x_path = os.path.join(self.config['SaveDir'], f"{idx}")
                        os.makedirs(x_path, exist_ok=True)
                        if hasattr(model, "save"):
                            model.save(x_path)
                        if isinstance(loss, tuple):
                            current_loss, loss = loss
                        with open(x_path + "/x.pkl", 'wb') as f:
                            pickle.dump(x, f)
                        self.comm.send(dest=0, tag=0, obj=[loss, x_path, idx])
                else:
                    logger.info(f'Worker {rank} has been stopped')
                    stop = False