import math
import pandas as pd
import torch
import numpy as np
from dragon.utils.tools import logger

def count_epochs(max_epochs, min_epochs, max_generations, generation):
    beta = math.log(max_epochs)
    alpha = - (beta - math.log(min_epochs)) / max_generations
    return int(math.exp(alpha * generation + beta))


def predict(model, device, loader, mode=None, online=None, mixture=None):
    preds = []
    trues = []
    
    if isinstance(online, dict):
        train_loss_fn = online['TrainLoss']
        optimizer = online['Optimizer']
        new_loader = online['Loader']
        interval = 0
        for batch_idx, (input, target) in enumerate(loader):
            # Do prediction
            target = target.to(device).float()
            pred = do_prediction(input, model, mode, device)
            preds.append(pred.detach().cpu().numpy())
            trues.append(target.detach().cpu().numpy())
            # Update model weights with last streaming data
            new_loader.append((input, target))
            interval+=1
            if (interval % online["PredictionInterval"]) == 0:
                new_loader = new_loader[-online["PredictionHistory"]:]
                for e in range(online['NumStreamingEpochs']):
                    for i, t in new_loader:
                        pred = do_prediction(i, model, mode, device)
                        loss = train_loss_fn(pred, t)
                        loss.backward(retain_graph=False)
                        optimizer.step()
                        optimizer.zero_grad()
    else:
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader):
                # Do prediction
                target = target.to(device).float()
                pred = do_prediction(input, model, mode, device)
                preds.append(pred.detach().cpu().numpy())
                trues.append(target.detach().cpu().numpy())
    preds = np.array(preds).reshape(-1, 1)
    trues = np.array(trues).reshape(-1, 1)
    return preds, trues

def load_model(model, f_ckpt, device):
    state_dict = torch.load(f_ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def mix_predictions(loader, models, device, mixture, mode=None, online=None):
    mix = mixture['Mix']
    preds = []
    trues = []
    if isinstance(online, dict):
        raise NotImplementedError
    else:
        with torch.no_grad():
            preds_df = pd.DataFrame()
            new_experts = [[] for k in range(len(models))]
            for batch_idx, (input, target) in enumerate(loader):
                # Do prediction
                target = target.to(device).float()
                for i, k in enumerate(models):
                    pred_k = do_prediction(input, k, mode, device)
                    new_experts[i].append(pred_k.detach().cpu().numpy())
                    logger.info(f"New experts {i}: {len(new_experts[i])}")
                trues.append(target.detach().cpu().numpy())
                # Update mixture with last streaming data
                if (batch_idx % mixture["PredictionInterval"]) == 0:
                    for i in range(len(models)):
                        preds_df[i] = new_experts[i]
                    preds.append(mix.predict(new_experts=preds_df))
                    logger.info(f'experts: {preds_df.shape}, y = {np.squeeze(trues[-mixture["PredictionInterval"]:]).shape}')
                    mix.update(new_experts=preds_df, new_y=np.squeeze(trues[-mixture["PredictionInterval"]:]))
                    preds_df = pd.DataFrame()
                    new_experts = [[] for k in range(len(models))]
    preds = np.array(preds).reshape(-1, 1)
    trues = np.array(trues).reshape(-1, 1)
    return preds, trues



def do_prediction(input, model, mode, device):
    if isinstance(input, list):
        for i in range(len(input)):
            input[i] = input[i].float().to(device)
    else:
        input = input.float().to(device)
    pred = model(input, mode=mode)
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    return pred
