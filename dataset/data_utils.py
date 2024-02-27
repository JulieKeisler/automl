import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler, DataLoader
from sklearn.feature_selection import mutual_info_regression
from dragon.utils.tools import logger

def build_features(data, features, freq):
    data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y %H:%M')
    missing_features = [f for f in features if f not in data.columns]
    totDays = int(data.shape[0] / freq)
    data = data.iloc[-freq * totDays:].reset_index(drop=True)
    return data


def temp_smoothing(temp, theta):
    for t in range(1, temp.shape[0] - 1):
        temp[t] = (1 - theta) * temp[t - 1] + theta * temp[t]
    return temp


def filter_method(data, borders):
    if "date" in data.columns:
        data.index = pd.to_datetime(data['date'], format='%d/%m/%Y %H:%M')
        data.drop('date', axis=1, inplace=True)
    train_data = data.loc[borders["Border1s"][0]:borders["Border2s"][0]]
    # Step 1: identify input features having high correlation with the target variable
    importances = train_data.drop("Target", axis=1).apply(lambda x: x.corr(train_data['Target']))
    indices = np.argsort(importances)
    high_correlated_features = []
    names = train_data.drop("Target", axis=1).columns
    for i in range(0, len(indices)):
        if np.abs(importances[i]) > 0.1:
            high_correlated_features.append(names[i])
    X = train_data[high_correlated_features]
    # Step 2: identify input features that have a low correlation with other independent variables
    not_correlated_features = [X.columns[0]]
    for new in X.columns[1:]:
        correlated = False
        for old in not_correlated_features:
            if new != old:
                corr = np.abs(X[new].corr(X[old]))
                if corr > 0.85:
                    correlated = True
                    new_index = list(names).index(new)
                    old_index = list(names).index(old)
                    if np.abs(importances[new_index]) > np.abs(importances[old_index]):
                        not_correlated_features.remove(old)
                        correlated = False
                    break
        if not correlated:
            not_correlated_features.append(new)
    # Step 3: Find the information gain or mutual information of the independant variable with respect to a target
    # variable
    X = X[not_correlated_features]
    X.dropna(inplace=True)
    y = train_data.dropna()['Target']
    mi = mutual_info_regression(X, y)
    mi = pd.Series(mi)
    time_final_features = list(mi.sort_values(ascending=False).index)
    final_features = data.columns[time_final_features].tolist()
    filtered_features = list(filter(lambda f: f != "Target", final_features))
    cols = ['date', 'Target'] + filtered_features
    data.reset_index(drop=False, inplace=True)
    return data[cols]

class MinMaxScaler:
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def fit(self, data):
        self.min = data.min(0) * 0.9
        self.max = data.max(0) * 1.1

    def transform(self, data):
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        return (data - min) / (max - min)

    def inverse_transform(self, data):
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        return data * (max - min) + min

    def target_inverse_transform(self, data):
        min = self.min
        max = self.max
        if (len(data.shape) == 1) or (data.shape[1] == 1):
            if not type(self.min) == int:
                min = self.min[-1]
                max = self.max[-1]
        data = data.astype(np.float64)
        data = data * (max - min) + min
        return data

def split_train_val(train_data, parameters, seed, size=0.2):
    split = int(np.floor(size * train_data.__len__()))
    np.random.seed(seed)
    indices = list(range(train_data.__len__()))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(train_data, batch_size=parameters['BatchSize'], sampler=train_sampler)
    validation_loader = DataLoader(train_data, batch_size=parameters['BatchSize'], sampler=valid_sampler)
    return train_loader, validation_loader


