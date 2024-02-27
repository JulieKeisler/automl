import numpy as np
import torch
from dragon.search_space.cells import WeightsAdjCell
from dragon.utils.tools import logger
from torch import nn
import os


class LoadForecastingNetwork(nn.Module):
    def __init__(self, args, labels, config):
        super().__init__()
        self.args = args
        self.labels = labels
        self.crit = np.nan
        self.input_shape = config['TSShape']

        if config['GradientFeatures']:
            self.mode = None
            self.TS_alphas = nn.Parameter(torch.Tensor(args[0]), requires_grad=True)
            self.sample_idx_TS = nn.Sigmoid()(self.TS_alphas)
        if "Features" in self.labels:    
            i_start = 1
        else:
            i_start = 0
        # First DAG    
        if hasattr(args[i_start].operations[0], "operation"):
            args[i_start].operations[0].modification(input_shapes=[self.input_shape])
        else:
            args[i_start].operations[0].set_operation(input_shapes=[self.input_shape])
        for j in range(1, len(args[i_start].operations)):
            input_shapes = [args[i_start].operations[i].output_shape for i in range(j) if args[i_start].matrix[i, j] == 1]
            if hasattr(args[i_start].operations[j], "operation"):
                args[i_start].operations[j].modification(input_shapes=input_shapes)
            else:
                args[i_start].operations[j].set_operation(input_shapes=input_shapes)
        self.cell_2d = WeightsAdjCell(args[i_start])

        # Save shape after flatten
        self.flat_shape = (self.cell_2d.output_shape[0],np.prod(self.cell_2d.output_shape[1:]))

        # Second DAG
        if hasattr(args[i_start+1].operations[0], "operation"):
            args[i_start+1].operations[0].modification(input_shapes=[self.flat_shape])
        else:
            args[i_start+1].operations[0].set_operation(input_shapes=[self.flat_shape])
        for j in range(1, len(args[i_start+1].operations)):
            input_shapes = [args[i_start+1].operations[i].output_shape for i in range(j) if args[i_start+1].matrix[i, j] == 1]
            if hasattr(args[i_start+1].operations[j], "operation"):
                args[i_start+1].operations[j].modification(input_shapes=input_shapes)
            else:
                args[i_start+1].operations[j].set_operation(input_shapes=input_shapes)
        self.cell_1d = WeightsAdjCell(args[i_start+1])

        # Output layer
        self.output = args[i_start+2]
        if hasattr(self.output, "operation"):
            self.output.modification(input_shapes=[self.cell_1d.output_shape])
        else:
            self.output.set_operation(input_shapes=[self.cell_1d.output_shape])

    def forward(self, X, mode=None):
        if mode is not None:
            self.set_mode(mode)
            self.sample_idx_TS = nn.Sigmoid()(self.TS_alphas)
            X = (X.squeeze(-1) * self.sample_idx_TS).unsqueeze(-1)
        out_2d = self.cell_2d(X)
        flat = nn.Flatten(start_dim=2)(out_2d)
        out_1d = self.cell_1d(flat)
        out = self.output(out_1d)
        return out.squeeze(-1)

    def set_mode(self, mode):
        self.mode = mode

    def weight_parameters(self):
        for p in self.cell_2d.parameters():
            yield p
        for p in self.cell_1d.parameters():
            yield p
        for p in self.output.parameters():
            yield p

    def features_parameters(self):
        yield self.TS_alphas

    def set_prediction_to_save(self, name, df):
        if hasattr(self, "prediction"):
            self.prediction[name] = df
        else:
            self.prediction = {name: df}
    def set_save_dir(self, path):
        self.save_dir = path

    def set_loss(self, crit):
        self.crit = crit

    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = os.path.join(path, "best_model.pth")
        torch.save(self.state_dict(), full_path)
        if hasattr(self, "prediction"):
            for k in self.prediction.keys():
                self.prediction[k].to_csv(os.path.join(path, f"best_model_{k}_outputs.csv"))
        full_path = os.path.join(path, "best_model_archi.csv")
        if hasattr(self, "save_dir"):
            snapshots = self.save_dir + "/snapshots/"
            if os.path.exists(snapshots):
                os.system(f"cp -rf {snapshots} {path}") 
                os.system(f"rm -rf {snapshots}")
        with open(full_path, "w") as f:
            f.write(";".join(str(e) for e in self.labels+['loss\n']))
            f.write(";".join(str(e) for e in self.args+[str(self.crit)+'\n']))
        logger.info(f"Best model have been saved to {path}")