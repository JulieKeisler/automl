import os
import tempfile
import shutil
import numpy as np
import torch
import pandas as pd
from torch import nn
from utils.utils_training import load_model, predict
from dragon.utils.exceptions import EndOnNan
from dragon.utils.tools import logger, set_seed

from utils.optimizers import configure_optimizers


class Trainer:
    def __init__(self, dataloader):
        super().__init__()
        config = dataloader.config
        self.model = config['Model']
        self.config = config
        self.data_loader = dataloader
        self.labels = None

    def build_model(self, args):
        self.config['TSShape'] = self.data_loader.config['TSShape']
        model = self.model(args, self.labels, self.config)
        for n, p in model.named_parameters():
            if p.dim() > 1:
                try:
                    nn.init.xavier_uniform_(p)
                except ZeroDivisionError as e:
                    logger.error(f'{n} = {p}')
                    raise e
        model.to(self.config["Device"])
        return model

    def train_model(self, model, train_loader, val_loader, mode, save_dir):
        self.config['FeaturesSelection'] = mode is not None
        if mode == "optim":
            obj = "Features"
            num_epochs = self.config["Ew"]
        elif mode == "fixed" or mode is None:
            obj = "Weights"
            num_epochs = self.config["Ep"]
        train_loss_fn = self.config["Loss"].to(self.config['Device'])
        val_loss_fn = self.config["Loss"].to(self.config['Device'])
        optimizer, scheduler = configure_optimizers(model, self.config[obj + 'Optimizer'])
        callbacks = self.config[obj + 'Callbacks']
        for cb in callbacks:
            cb.reset()
            if hasattr(cb, "update_save_dir"):
                cb.update_save_dir(save_dir)
                os.makedirs(cb.save_dir)
        try:
            device = self.config['Device']
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0
                for batch_idx, (input, target) in enumerate(train_loader):
                    if isinstance(input, list):
                        for i in range(len(input)):
                            input[i] = input[i].float().to(device)
                    else:
                        input = input.float().to(device)
                    target = target.to(device).float()
                    pred = model(input, mode=mode)
                    loss = train_loss_fn(pred, target)
                    if mode == "optim":
                        loss += self.config['L1Eps'] * sum(
                            nn.Sigmoid()(p).abs().sum() for p in model.features_parameters())
                    train_loss += train_loss_fn(pred, target)
                    loss.backward(retain_graph=False)
                    optimizer.step()
                    optimizer.zero_grad()
                train_loss /= len(train_loader)

                for cb in callbacks:
                    try:
                        cb.on_train_epoch_end(train_loss=train_loss, model=model, epoch=epoch)
                    except AttributeError as e:
                        logger.error(f'{cb}')
                        raise e
                    if hasattr(cb, "stop_training"):
                        if cb.stop_training:
                            raise InterruptedError
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_idx, (input, target) in enumerate(val_loader):
                        if isinstance(input, list):
                            for i in range(len(input)):
                                input[i] = input[i].float().to(device)
                        else:
                            input = input.float().to(device)
                        target = target.to(device).float()
                        pred = model(input, mode=mode)
                        val_loss += val_loss_fn(pred, target)
                val_loss = val_loss / len(val_loader)
                for cb in callbacks:
                    cb.on_validation_epoch_end(val_loss=val_loss, model=model, epoch=epoch, obj=obj)
                    if hasattr(cb, "stop_training"):
                        if cb.stop_training:
                            raise InterruptedError
                if scheduler is not None:
                    scheduler.step()
        except KeyboardInterrupt:
            for cb in callbacks:
                if hasattr(cb, "load_best"):
                    cb.load_best(model)
            pass
        except InterruptedError:
            for cb in callbacks:
                if hasattr(cb, "load_best"):
                    cb.load_best(model)
            pass
        torch.cuda.empty_cache()

        if ("OnlinePrediction" in self.config) and self.config["OnlinePrediction"] and (obj == "Weights"):
            val_dataset = val_loader.dataset
            loader = [val_dataset.__getitem__(i) for i in range(val_dataset.__len__()-self.config["PredictionHistory"], val_dataset.__len__())]
            loader = [(torch.tensor(x).unsqueeze(0).float().to(device), torch.tensor(y).unsqueeze(0).float().to(device)) for (x,y) in loader]
            self.config["OnlinePrediction"] = {"TrainLoss": train_loss_fn, "Optimizer": optimizer,
                                               "NumStreamingEpochs": self.config["NumStreamingEpochs"],
                                               "Loader": loader,
                                               "PredictionHistory": self.config["PredictionHistory"],
                                               "PredictionInterval": self.config["PredictionInterval"]}
        elif obj == "Weights":
            self.config["OnlinePrediction"] = None
        for cb in callbacks:
            if hasattr(cb, "on_training_end"):
                cb.on_training_end()
        return model

    def prediction(self, model, seed, save_dir, mode=None, filters=None, online=None):
        device = self.config["Device"]
        if filters is None:
            test_loader = self.data_loader.get_loader("test", seed=None)
        else:
            test_loader = self.data_loader.get_loader("test", seed=None, filters=filters)
        dir_path = save_dir + "/snapshots/"
        if os.path.exists(dir_path):
            preds = None
            models = [load_model(model, os.path.join(dir_path, f_name), device) for f_name in os.listdir(dir_path)]
            for i, m in enumerate(models): 
                pred_k, trues = predict(m, device, test_loader, mode, online)
                if preds is None:
                    preds = pred_k
                else:
                    preds += pred_k
            if preds is None:
                preds, trues = predict(model, device, test_loader, mode)
            else:
                preds /=(i+1)

        else:
            preds, trues = predict(model, device, test_loader, mode)
        test_dataset = self.data_loader.get_dataset("test")
        preds = test_dataset.scaler.target_inverse_transform(preds)
        trues = test_dataset.scaler.target_inverse_transform(trues)
        crit = self.config["Criterium"](trues, preds)
        if self.config['Freq'] == 48:
            freq_n = "30min"
        elif self.config['Freq'] == 24:
            freq_n = "H"
        df = pd.DataFrame()
        df['Pred'] = preds.reshape(-1,)
        df['Actual'] = trues.reshape(-1,)
        if isinstance(freq_n, str):
            dates = pd.date_range(test_dataset.border1s[test_dataset.set_type], test_dataset.border2s[test_dataset.set_type], freq = freq_n)
            df['Date'] = dates
            df.set_index("Date", inplace=True)
        
        model.set_prediction_to_save("test", df)
        model.set_loss(crit)
        model.set_save_dir(save_dir)
        logger.info(f"With seed: {seed} ===> criterium on test dataset: {crit}")
        return crit

    def train_and_test(self, args, idx=None):
        init_args = args.copy()
        if isinstance(args, list):
            args = dict(zip(self.labels, args))
        if ("GradientFeatures" in self.config) & self.config['GradientFeatures']:
            self.config["Ew"] = self.config["MaxFeaturesEpochs"] 
        if 'Seed' in args:
            seed = args['Seed']
        else:
            seed = 0
        for key in args.keys():
            if key in self.data_loader.config:
                self.data_loader.config[key] = args[key]
        args = list({key: args[key] for key in self.labels}.values())

        if idx is None:
            save_dir = self.config["SaveDir"] + tempfile.NamedTemporaryFile().name
        else:
            save_dir = self.config['SaveDir'] + f"/{idx}/"
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            logger.warning(f"Save dir: {save_dir} already exists, deleting {save_dir}")
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)

        set_seed(seed)

        train_loader = self.data_loader.get_loader(flag='train', seed=seed)
        val_loader = self.data_loader.get_loader(flag='val', seed=seed)
        model = self.build_model(args)
        try:
            if self.config["GradientFeatures"]:
                # e < Ew: train theta and w jointly
                model = self.train_model(model, train_loader, val_loader, mode="optim", save_dir=save_dir)
                # e > Ew: train only theta
                if ("SimpleAfterFeatures" in self.config) and self.config["SimpleAfterFeatures"]:
                    model.set_mode("fixed")
                    filters = model.sample_idx_TS.cpu().numpy()
                    if np.sum(filters) > 0:
                        train_loader = self.data_loader.get_loader(flag='train', seed=seed, filters=filters)
                        val_loader = self.data_loader.get_loader(flag='val', seed=seed, filters=filters)
                        self.config['GradientFeatures'] = False

                        model_simple = self.build_model(args)
                        self.config['GradientFeatures'] = True
                        model = self.train_model(model_simple, train_loader, val_loader, mode=None, save_dir=save_dir)
                        mode=None
                    else:
                        logger.error("No features have been selected")
                        raise EndOnNan(self.config["Ew"])
                else:
                    filters=None
                    self.config["SimpleAfterFeatures"] = False
                    model = self.train_model(model, train_loader, val_loader, mode="fixed", save_dir=save_dir)
                    mode="fixed"
            else:
                model = self.train_model(model, train_loader, val_loader, mode=None, save_dir=save_dir)
                filters=None
                mode=None
            
            loss = self.prediction(model=model, seed=seed, save_dir=save_dir, mode=mode, filters=filters, online=self.config["OnlinePrediction"])
        except EndOnNan:
            loss = np.inf
            args = init_args
            if idx is not None:
                logger.error(f'Idx = {idx}, with seed: {seed} ===> criteriums = {loss}')
            else:
                logger.error(f'With seed: {seed} ===> criteriums = {loss}')
        except RuntimeError as e:
            logger.error(f"Failed with {e}", exc_info=True)
            loss = np.inf
            args = init_args
        if np.isnan(loss):
            loss = np.inf
            args = init_args
            if idx is not None:
                logger.error(f'Idx = {idx}, with seed: {seed} ===> criteriums = {loss}')
            else:
                logger.error(f'With seed: {seed} ===> criteriums = {loss}')
        model.args = args
        shutil.rmtree(save_dir)
        return loss, model