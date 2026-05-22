__author__ = "Pinkas Matěj"
__maintainer__ = "Pinkas Matěj"
__email__ = "pinkas.matej@gmail.com"
__created__ = "22/05/2026"
__date__ = "22/05/2026"
__status__ = "Prototype"
__version__ = "0.1.0"
__credits__ = []

"""
Project: BBAP20
Filename: trainer.py
Directory: src/shared/
"""

import os.path
import torch
import torch.optim as optim
from torch.utils.data import Subset
from pathlib import Path

from src.shared.utils import project_directory
from src.shared.wrappers import monitor

class PytorchTrainer:
    def __init__(self, model_setup, config, dataset_loader=None, dataset_loader_train=None, dataset_loader_validation=None):
        self.dataset_loader = dataset_loader
        self.dataset_loader_train = dataset_loader_train
        self.dataset_loader_validation = dataset_loader_validation
        self.model_setup = model_setup
        self.config = config

        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.train_loader = None
        self.test_loader = None

        self.start_epoch = 0
        self.lowest_test_loss = float('inf')

        self.__setup()

    def __setup(self):
        self.__load_config()

        self.__device_init()

        #self.criterion = InfoNCELossTrain(initial_temperature=0.07).to(self.device)

        self.__model_init()
        self.__optimizer_init()
        self.__scheduler_init()

        self.__dataset_init()

    @monitor
    def __load_config(self):
        cfg = self.config

        self.learning_rate = float(cfg['train_parameters']['learning_rate'])
        self.epochs = int(cfg['train_parameters']['epochs'])
        self.batch_size = int(cfg['train_parameters']['batch_size'])
        self.test_ratio = float(cfg['train_parameters']['test_ratio'])
        self.weight_decay = float(cfg['train_parameters']['weight_decay'])
        self.lr_drop_factor = float(cfg['train_parameters']['lr_drop_factor'])
        self.lr_drop_patience = int(cfg['train_parameters']['lr_drop_patience'])
        self.lr_drop_min = float(cfg['train_parameters']['lr_drop_min'])

        directory = project_directory(Path("data"))

        self.model_path = directory / cfg['directory']['model_path']
        self.checkpoint_path = directory / cfg['directory']['checkpoint_dir']
        self.dataset_dir = directory / cfg['directory']['dataset_dir']
        self.tensorboard_dir = directory / cfg['directory']['tensorboard_dir']

        self.sigma = float(cfg['dataset_parameters']['sigma'])

        self.embedding_dim = float(cfg['model_parameters']['embedding_dim'])
        self.load_model = bool(cfg['model_parameters']['load_model'])

    @monitor
    def __device_init(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @monitor
    def __model_init(self):
        self.model = self.model_setup.to(self.device)

    @monitor
    def __optimizer_init(self):
        if hasattr(self.criterion, "parameters"):
            all_parameters = list(self.model.parameters()) + list(self.criterion.parameters())
        else:
            all_parameters = list(self.model.parameters())

        self.optimizer = optim.AdamW(
            all_parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    @monitor
    def __scheduler_init(self):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.lr_drop_factor, patience=self.lr_drop_patience,
                                                               min_lr=self.lr_drop_min)

    @monitor
    def __dataset_init(self):
        if self.dataset_loader is not None:
            train_size = int((1-self.test_ratio) * len(self.dataset_loader))
            test_size = len(self.dataset_loader) - train_size

            # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
            train_dataset = Subset(self.dataset_loader, range(train_size))
            test_dataset = Subset(self.dataset_loader, range(train_size, len(self.dataset_loader)))

            train_dataset.dataset.augment_data = True
            test_dataset.dataset.augment_data = False

            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        else:
            self.train_loader = torch.utils.data.DataLoader(self.dataset_loader_train, batch_size=self.batch_size, shuffle=True,
                                                            num_workers=8)
            self.test_loader = torch.utils.data.DataLoader(self.dataset_loader_validation, batch_size=self.batch_size, shuffle=True,
                                                           num_workers=8)


    def save_checkpoint(self, epoch):
        checkpoint = {'epoch':epoch,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'scheduler_state_dict': self.scheduler.state_dict(),
                      'lowest_test_loss':self.lowest_test_loss}

        checkpoints = [
            f for f in os.listdir(self.checkpoint_path)
            if f.startswith("checkpoint_") and f.endswith(".pth")
        ]

        if len(checkpoints) == 0:
            max_number = -1
        else:
            max_number = max(int(f.split("_")[1].split(".")[0]) for f in checkpoints)

        filename = os.path.join(self.checkpoint_path, f"checkpoint_{max_number+1}.pth")

        torch.save(checkpoint, filename)

    def load_checkpoint(self):
        checkpoints = [
            f for f in os.listdir(self.checkpoint_path)
            if f.startswith("checkpoint_") and f.endswith(".pth")
        ]
        max_number = max(int(f.split("_")[1].split(".")[0]) for f in checkpoints)
        filename = os.path.join(self.checkpoint_path, f"checkpoint_{max_number}.pth")

        checkpoint = torch.load(filename, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.lowest_test_loss = checkpoint['lowest_test_loss']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            print("Optimizer state moved to GPU.")

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            if checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print("Checkpoint loaded.")

if __name__ == '__main__':
    pass
