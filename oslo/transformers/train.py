import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from trainer import Trainer

try:
    import transformers
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")

def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_fn', required = True)
    p.add_argument('--gpu_id', type = int, default = 0 if torch.cuda.is_available() else -1)
    
    p.add_argument('--train_ratio', type = float, default = .8)
    
    p.add_argument('--batch_size', type = int, default = 64)
    p.add_argument('--n_epochs', type = int, default = 20)
    p.add_argument('--verbose', type = int, default = 2)
    
    config = p.parse_args()
    
    return config

def main(config):
    # device = torch.device('cpu') if config.gpu_id < 0 else torch.device(f'cuda:{config.gpu_id}')
    # trainer = Trainer(model, optimizer, crit)
    # trainer.run()
    pass

if __name__ == "__main__":
    config = define_argparser()
    main(config)