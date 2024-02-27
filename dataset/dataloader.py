import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from dragon.utils.tools import logger

from dataset.data_utils import build_features, filter_method, split_train_val, MinMaxScaler


class LoadDataset(Dataset):
    def __init__(self, df_raw, flag, borders, freq, features, scaler=True):
        try:
            assert flag in ['train', 'val', 'test', 'train+val']
        except AssertionError:
            logger.info("Flag should be in ['train', 'val', 'test', 'train+val'], got {} instead".format(flag))
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2, 'train+val': 3}
        self.set_type = type_map[flag]
        self.features = features
        self.border1s = borders['Border1s']
        self.border2s = borders['Border2s']
        self.freq = freq
        self.scale = scaler
        self.__read_data__(df_raw)

    def __read_data__(self, data):
        df = data.copy()
        if self.scale:
            self.scaler = MinMaxScaler()
        try:
            df.set_index('date', inplace=True)
        except KeyError:
            df.set_index('Date', inplace=True)
        border1 = self.border1s[self.set_type]
        border2 = self.border2s[self.set_type]
        cols = self.features + ['Target']

        train_data = df.loc[self.border1s[0]:self.border2s[0], cols]
        if self.scale:
            self.scaler.fit(train_data.values)
            scaled_data = self.scaler.transform(df.loc[:, cols].values)
        else:
            scaled_data = df.loc[:,cols].values
        df = pd.DataFrame(scaled_data, index=df.index, columns=cols)
        df = df.loc[border1: border2, cols]

        df_2d = df[self.features]


        data_2d = np.stack([df_2d[col].values.reshape(-1, self.freq) for col in df_2d.columns], axis=1)
        data_2d = np.swapaxes(data_2d, 1, 2)
        self.data_2d = np.expand_dims(data_2d, axis=-1)
        self.y = df['Target'].values.reshape(-1, self.freq)
       
    def __getitem__(self, index):
        return self.data_2d[index], self.y[index]
    
    def __len__(self):
        return self.data_2d.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class LoadDataLoader:
    def __init__(self, config):
        self.config = config
        self.features = self.config['Features']
        self.borders = self.config['Borders']
        self.data = self.load_data()
        self.val_data = self.config['ValData']
        self.dataset = self.get_dataset('train')
        self.config['TSShape'] = self.dataset.data_2d.shape[1:]

    def load_data(self):
        # Load csv file
        data = pd.read_csv(self.config['filename'])
        data['Target'] = data[self.config['target']]
        data = build_features(data, self.features, self.config['Freq'])
        data = data[['date', 'Target']+self.features]

        if ('FilterFeatures' in self.config) and (self.config['FilterFeatures']):
            data = filter_method(data, self.borders)
            features = list(filter(lambda f: (f != "Target") or (f !="date"), data.columns[2:].tolist()))
            self.features = features
            self.config['Features'] = self.features
        return data

    def get_loader(self, flag, seed=None, filters=None):
        TS_filter = filters
        if flag == 'train':
            train_data = self.get_dataset("train")
            if TS_filter is not None:
                data_2d = train_data.data_2d[:, :, TS_filter.astype(bool)]
                train_data.data_2d = data_2d
            if self.val_data:
                train_loader = DataLoader(train_data, batch_size=int(self.config['BatchSize']), shuffle=True, num_workers=1,
                                          drop_last=True)
                
            else:
                train_loader, _ = split_train_val(train_data, self.config, seed, size=0.1)
            self.config['TSShape'] = train_data.data_2d.shape[1:]
            return train_loader
        elif flag == "val":
            if self.val_data:
                val_data = self.get_dataset("val")
            else:
                val_data = self.get_dataset("train")
            if TS_filter is not None:
                data_2d = val_data.data_2d[:, :, TS_filter.astype(bool)]
                val_data.data_2d = data_2d
            if self.val_data:
                val_loader = DataLoader(val_data, batch_size=int(self.config['BatchSize']), shuffle=True, num_workers=1,
                                        drop_last=True)
            else:
                _, val_loader = split_train_val(val_data, self.config, seed, size=0.1)
            self.config['TSShape'] = val_data.data_2d.shape[1:]
            return val_loader
        elif flag == "test":
            test_data = self.get_dataset("test")
            if TS_filter is not None:
                data_2d = test_data.data_2d[:, :, TS_filter.astype(bool)]
                test_data.data_2d = data_2d
            self.config['TSShape'] = test_data.data_2d.shape[1:]
            test_loader = DataLoader(test_data, num_workers=1, shuffle=False)
            return test_loader

        elif flag == "train+val":
            data = self.get_dataset("train+val")
            if TS_filter is not None:
                data_2d = data.data_2d[:, :, TS_filter.astype(bool)]
                data.data_2d = data_2d
            self.config['TSShape'] = data.data_2d.shape[1:]
            loader = DataLoader(data, num_workers=1, shuffle=False)
            return loader

    def get_dataset(self, flag, scaler=True):
        self.dataset = LoadDataset(self.data, flag, self.borders, freq=self.config['Freq'], features=self.features, scaler=scaler)
        return self.dataset
