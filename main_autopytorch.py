import os
import shutil
import tempfile
from dragon.utils.tools import logger
from dataset.dataloader import LoadDataLoader
from autoPyTorch.api.tabular_regression import TabularRegressionTask
from utils.metrics import MAPE
import pandas as pd
from utils.config import exp_configs
import numpy as np

def fit_autopytorch(config):
    exp_config = exp_configs[config['config']]
    config.update(exp_config) 
    config['SaveDir'] += config["save_dir"]
    save_dir= f"{config['SaveDir']}/{tempfile.NamedTemporaryFile().name}"
    logger.info(f'SaveDir = {save_dir}')
    logger.info(f'Creating DataLoader for AutoPytorch')
    data_loader = LoadDataLoader(config)
    trainset = data_loader.get_dataset('train', scaler=False)
    testset = data_loader.get_dataset('test', scaler=False)
    X_train = np.swapaxes(trainset.data_2d, 1, 2).reshape(trainset.data_2d.shape[0]*config['Freq'], -1)
    y_train = trainset.y.reshape(-1,)
    X_test = np.swapaxes(testset.data_2d, 1, 2).reshape(testset.data_2d.shape[0]*config['Freq'], -1)
    y_test = testset.y.reshape(-1,)
    logger.info(f"DataLoader created.")
    if os.path.exists(config['SaveDir']):
        logger.warning(f"{config['SaveDir']} already exists. Deleting.")
        shutil.rmtree(config['SaveDir'])
    api = TabularRegressionTask(seed=config['seed'],
                                n_jobs=10,
                                temporary_directory=config['SaveDir'] + '/tmp/',
                                output_directory=config['SaveDir'] + '/output',
                                delete_tmp_folder_after_terminate=True)
    logger.info(f"Api created with seed = {config['seed']}, search starts...")
    api.search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        optimize_metric='root_mean_squared_error',
        budget_type="epochs",
        min_budget=10,
        max_budget=200,
        total_walltime_limit=3600,
        func_eval_time_limit_secs=60*15,
        enable_traditional_pipeline=False,
        memory_limit=None        
    )
    logger.info(f"API is done!!")
    y_pred = api.predict(X_test)
    mape = MAPE(y_test, y_pred)

    logger.info(f"Criterium on test dataset: {mape}")
    if config['Freq'] == 48:
        freq_n = "30min"
    elif config['Freq'] == 24:
        freq_n = "H"
    df = pd.DataFrame()
    df['Pred'] = y_pred.reshape(-1,)
    df['Actual'] = y_test.reshape(-1,)
    test_dataset = data_loader.get_dataset("test")
    if isinstance(freq_n, str):
        dates = pd.date_range(test_dataset.border1s[test_dataset.set_type], test_dataset.border2s[test_dataset.set_type], freq = freq_n)
        df['Date'] = dates
        df.set_index("Date", inplace=True)
    path = config['SaveDir']
    if not os.path.exists(path):
            os.makedirs(path)
    logger.info(f'Saving prediction to {os.path.join(path, f"best_model_outputs.csv")}')
    df.to_csv(os.path.join(path, f"best_model_outputs.csv"))