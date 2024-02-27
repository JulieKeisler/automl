import os
import shutil
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from dragon.utils.tools import logger

class Callback:
    def __init__(self):
        pass

    def on_train_epoch_end(self, train_loss, **kwargs) -> None:
        """Called when the train epoch ends."""
        pass

    def on_validation_epoch_end(self, val_loss, **kwargs) -> None:
        """Called when the validation epoch ends."""
        pass

    def reset(self):
        pass


class TrainingHistory(Callback):
    def __init__(self, save_dir=None):
        super().__init__()
        self.loss_history = []
        self.val_loss_history = []
        self.save_dir = save_dir

    def on_train_epoch_end(self, train_loss, **kwargs) -> None:
        self.loss_history.append(train_loss.tolist())

    def on_validation_epoch_end(self, val_loss, **kwargs) -> None:
        self.val_loss_history.append(val_loss.tolist())

    def reset(self):
        self.loss_history = []
        self.val_loss_history = []
    
    def on_training_end(self):
        if self.save_dir is not None:
            x = np.arange(len(self.loss_history))
            plt.plot(x, self.loss_history, label="Training loss")
            plt.plot(x, self.val_loss_history, label="Validation loss")
            plt.legend()
            plt.savefig(self.save_dir+"/training_plot.pdf")


class Checkpoint(Callback):
    def __init__(self, save_dir, save_top_k=1, max_checkpoints=1, max_epoch=1, load_best=True, permanent_dir=None):
        super().__init__()
        self.original_save_dir = save_dir
        self.save_top_k = save_top_k
        self.saved = {}
        self.load_best_bool = load_best
        self.max_epoch = max_epoch
        if max_checkpoints is None:
            max_checkpoints = self.max_epoch
        self.every_n_epochs = self.max_epoch // max_checkpoints
        self.permanent_dir = permanent_dir

    def set_permanent_dir(self, path):
        try:
            os.makedirs(path)
        except FileExistsError:
            logger.warning(f"Path: {path} already exists.")
            pass
        self.permanent_dir = path

    def update_save_dir(self, save_dir):
        self.save_dir = save_dir + self.original_save_dir

    def on_validation_epoch_end(self, val_loss, epoch, model, **kwargs) -> None:
        if epoch % self.every_n_epochs == 0:
            path = self.save_dir + '/' + f'checkpoint{epoch}_{val_loss}.pth'
            if len(self.saved.keys()) < self.save_top_k:
                torch.save(model.state_dict(), path)
                self.saved[val_loss] = path
            else:
                max_loss = max(self.saved.keys())
                if val_loss < max_loss:
                    try:
                        os.remove(self.saved[max_loss])
                    except FileNotFoundError:
                        pass
                    self.saved.pop(max_loss)
                    torch.save(model.state_dict(), path)
                    self.saved[val_loss] = path

        if self.load_best_bool:
            if epoch == self.max_epoch:
                self.load_best(model)

    def load_best(self, model, **kwargs):
        if len(self.saved.keys())>0:
            best_loss = min(self.saved.keys())
            ckpt_path = self.saved[best_loss]
            model.load_state_dict(torch.load(ckpt_path))
    def reset(self):
        self.saved = {}
    def on_training_end(self):
        if self.permanent_dir is not None:
            allfiles = os.listdir(self.save_dir)
            # iterate on all files to move them to destination folder
            for f in allfiles:
                src_path = os.path.join(self.save_dir, f)
                dst_path = os.path.join(self.permanent_dir, f)
                shutil.move(src_path, dst_path)
            logger.info(f"Permanent dir: {self.permanent_dir}")



class EarlyStopping(Callback):
    def __init__(self, patience=0):
        super().__init__()
        self.patience = patience
        self.patience_count = 0
        self.min_loss = np.inf
        self.stop_training = False

    def on_validation_epoch_end(self, val_loss, **kwargs) -> None:
        if val_loss >= self.min_loss:
            self.patience_count += 1
        else:
            self.min_loss = val_loss
        if self.patience_count > self.patience:
            self.stop_training = True

    def reset(self):
        self.patience_count = 0
        self.min_loss = np.inf
        self.stop_training = False


class EndOnNan(Callback):
    def __init__(self):
        super().__init__()
        self.stop_training = False

    def on_train_epoch_end(self, train_loss, **kwargs) -> None:
        if torch.isnan(train_loss).item():
            self.stop_training = True

    def on_validation_epoch_end(self, val_loss, **kwargs) -> None:
        if torch.isnan(val_loss).item():
            self.stop_training = True
    def reset(self):
        self.stop_training = False


class Verbose(Callback):
    def __init__(self, logger):
        super().__init__()
        self.current_train_loss = None
        self.current_val_loss = None
        self.current_epoch = None
        self.obj = None
        self.logger = logger

    def on_train_epoch_end(self, train_loss, epoch, **kwargs) -> None:
        self.current_train_loss = train_loss
        self.current_epoch = epoch

    def on_validation_epoch_end(self, val_loss, epoch, obj, **kwargs) -> None:
        self.obj = obj
        self.current_val_loss = val_loss
        self.current_epoch = epoch
        self.log()

    def load_best(self, model, **kwargs) -> None:
        self.log()

    def log(self):
        self.logger.info(
            f'Epoch {self.current_epoch} ==> Opt {self.obj} train loss: {self.current_train_loss}, '
            f'val loss: {self.current_val_loss}')

    def reset(self):
        self.current_train_loss = None
        self.current_val_loss = None
        self.current_epoch = None
        self.obj = None

class TrainingDuration(Callback):
    def __init__(self, duration):
        super().__init__()
        self.duration = duration * 60 # convert minutes in secondes
        self.start_time = time.time()
        self.stop_training = False

    def on_train_epoch_end(self, train_loss, **kwargs) -> None:
        duration = time.time() - self.start_time
        if duration > self.duration:
            self.stop_training = True

    def on_validation_epoch_end(self, val_loss, **kwargs) -> None:
        duration = time.time() - self.start_time
        if duration > self.duration:
            self.stop_training = True

    def reset(self):
        self.start_time = time.time()
        self.stop_training = False
    
