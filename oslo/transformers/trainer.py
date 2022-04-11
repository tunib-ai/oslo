import torch
from copy import deepcopy
import numpy as np

class Trainer():
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit            # loss function
    
    # compute train_loss.
    def _train(self, x, y, config):
        self.model.train()    # Turn train mode on.
        
        # In every epoch, shuffle before begin.
        indices = torch.randperm(x.size(0), device = x.device)
        x = torch.index_select(x, dim = 0, index = indices).split(config.batch_size, dim = 0)
        y = torch.index_select(y, dim = 0, index = indices).split(config.batch_size, dim = 0)
        
        total_loss = 0
        
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())
            
            # Initialize the gradients of the model.
            self.optimizer.zero_grad()
            loss_i.backward()
            self.optimizer.step()
            
            if config.verbose >= 2:
                print(f"Train Iteration({i + 1}/{len(x)}): loss = {float(loss_i):.4e}")
            
            # Detach loss to prevent memory leak.
            total_loss += loss_i.item()
        
        return total_loss / len(x)             # Divide by number of batch.
    
    # compute validation_loss.
    def _validate(self, x, y, config):
        self.model.eval()    # Turn evaluation mode on.
        
        with torch.no_grad():
            # Shuffle before begin.
            indices = torch.randperm(x.size(0), device = x.device)
            x = torch.index_select(x, dim = 0, index = indices).split(config.batch_size, dim = 0)
            y = torch.index_select(y, dim = 0, index = indices).split(config.batch_size, dim = 0)
            
            total_loss = 0
            
            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())
                 
                if config.verbose >= 2:
                    print(f"Valid Iteration({i + 1}/{len(x)}): loss = {float(loss_i):.4e}")
                
                # Detach loss to prevent memory leak.
                total_loss += loss_i.item()
            
            return total_loss / len(x)             # divide by number of batch.
        
    def run(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None
        epochs = config.n_epochs
        
        for epoch in range(epochs):
            train_loss = self._train(train_data[0], train_data[1], config)     # train_data, train_label
            valid_loss = self._validate(valid_data[0], valid_data[1], config)  # valid_data, valid_label
            
            # Using deep copy to take a snapshot of current best weights.
            if valid_loss <= lowest_loss:
                # print("epoch of lowest_loss : ", epoch)
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())
            
            if epoch % 10 == 0:
                print(f"Epoch({epoch + 1}/{epochs}): train_loss = {train_loss:.4e} valid_loss = {valid_loss:.4e} lowest_loss = {lowest_loss:.4e}")
        
        # Restore to best model.
        self.model.load_state_dict(best_model)