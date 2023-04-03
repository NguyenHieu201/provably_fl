import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from .fedbase import BasicClient, BasicServer
from .fedprob_utils.smooth import Smooth
from .fedprob_utils.accuracy import ApproximateAccuracy


class Server(BasicServer):
    # TODO: change hard fix options
    def __init__(self, option, model: nn.Module, clients: list[BasicClient], test_data=None):
        super().__init__(option, model, clients, test_data)
        self.num_classes = 23
        self.noise = 0.5
        self.N0 = 100
        self.N = 1000
        self.alpha = 0.05

class Client(BasicClient):
    # TODO: change hard fix options
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super().__init__(option, name, train_data, valid_data)
        self.sigma = 0.5
        self.N0 = 100
        self.N = 1000
        self.alpha = 0.05
        self.num_classes = 23

    def train(self, model: nn.Module):
        """
        Training process for smoothed classifier
        Client training with noisy data
        """
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)

        # Traing phase for base classifer
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                inputs, outputs = batch_data
                inputs = inputs + torch.rand_like(inputs, device=self.calculator.device) * self.noise
                
                noisy_batch = [inputs, outputs]

                loss = self.calculator.get_loss(model, noisy_batch)
                loss.backward()
                optimizer.step()

    def certify(self, model: nn.Module, data_loader: DataLoader) -> pd.DataFrame:
        """
        Return predict, radius
        """
        certify_model = Smooth(model, self.num_classes, self.sigma)
        certify_results = []

        for batch_id, batch_data in enumerate(data_loader):
            inputs, outputs = batch_data
            batch_size = inputs.shape[0]
            
            for i in range(batch_size):
                input, output = inputs[i], outputs[i]
                pred, radius = certify_model.certify(input)
                correct = pred == output.data.max()
                certify_result = {
                    "radius": radius,
                    "correct": correct
                }
                certify_results.append(certify_result)
        return pd.DataFrame(certify_results)
    
    def accuracy_at_radii(self, model: nn.Module, data_loader: DataLoader, radii: np.ndarray) -> np.ndarray:
        certify_results = self.certify(model, data_loader)
        accuracy_calculator = ApproximateAccuracy(certify_results)
        return accuracy_calculator.at_radii(radii)

    def certify_train_radii(self, model: nn.Module, radii: np.ndarray):
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        return self.accuracy_at_radii(model, data_loader, radii)

    def certify_test_radius(self, model: nn.Module, radii: np.ndarray):
        data_loader = self.calculator.get_data_loader(self.valid_data, batch_size=self.batch_size)
        return self.accuracy_at_radii(model, data_loader, radii)