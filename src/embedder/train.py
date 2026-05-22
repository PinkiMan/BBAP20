__author__ = "Pinkas Matěj"
__maintainer__ = "Pinkas Matěj"
__email__ = "pinkas.matej@gmail.com"
__created__ = "22/05/2026"
__date__ = "22/05/2026"
__status__ = "Prototype"
__version__ = "0.1.0"
__copyright__ = ""
__license__ = ""
__credits__ = []

"""
Project: BBAP20
Filename: train.py
Directory: src/embedder/
"""

import wandb
import yaml
import torch
import pathlib
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from src.shared.trainer import PytorchTrainer
from src.embedder.model import CrossModalNetwork, CrossModalNetwork2, CrossModalNetwork3
from src.loader.loader import Loader
from src.shared.wrappers import monitor
from src.shared.utils import project_directory


class Trainer(PytorchTrainer):
    def __init__(self):

        cfg = self.__setup_config()
        model = self.__setup_model(cfg)
        dataset_loader_train, dataset_loader_validation = self.__load_dataset(cfg)

        # self.criterion = InfoNCELoss(temperature=0.07)
        self.criterion = InfoNCELossTrain(initial_temperature=cfg['train_parameters']['initial_temperature'], temp_min=cfg['train_parameters']['temp_min'], temp_max=cfg['train_parameters']['temp_max'])

        super().__init__(model_setup=model, config=cfg, dataset_loader_train=dataset_loader_train, dataset_loader_validation=dataset_loader_validation)

        #self.load_checkpoint()
        """for param_group in self.optimizer.param_groups:
            param_group['lr'] = 1.25e-6"""

        self.criterion.to(self.device)

        self.epoch = None

        self.writer_train = None
        self.writer_test = None

        self.scaler = torch.amp.GradScaler('cuda')  # lowers 32 bit to 16 bit float

    @staticmethod
    @monitor
    def __setup_config():
        directory = project_directory().parent
        with open(directory/pathlib.Path("configs/embedder.yaml"), 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)

        return cfg

    @staticmethod
    @monitor
    def __setup_model(cfg):
        embedding_dim = cfg['model_parameters']['embedding_dim']

        model = CrossModalNetwork3(embedding_dim=embedding_dim)

        return model

    @staticmethod
    @monitor
    def __load_dataset(cfg):
        directory = project_directory(Path("data/"))
        dataset_dir_train = directory / cfg['directory']['dataset_dir_train']
        dataset_dir_validation = directory / cfg['directory']['dataset_dir_validation']
        sigma = cfg['dataset_parameters']['sigma']

        dataset_loader_train = Loader(dataset_dir_train, sigma=sigma, augment_data=True, max_data_size=cfg['dataset_parameters']['dataset_size'],
                                satellite_map_size=(256, 256), height_map_size=(128, 128),
                                      satellite_map_augment_max_angle=cfg['augments']['satellite_map_augment_max_angle'],
                                      min_brightness_ratio=cfg['augments']['min_brightness_ratio'],
                                      max_brightness_ratio=cfg['augments']['max_brightness_ratio'],
                                      min_contrast_ratio=cfg['augments']['min_contrast_ratio'],
                                      max_contrast_ratio=cfg['augments']['max_contrast_ratio'],
                                      min_saturation_ratio=cfg['augments']['min_saturation_ratio'],
                                      max_saturation_ratio=cfg['augments']['max_saturation_ratio'],
                                      noise_ratio=cfg['augments']['noise_ratio'],
                                      heightmap_augment_max_angle=cfg['augments']['heightmap_augment_max_angle'],
                                      radial_drop_prob=cfg['augments']['radial_drop_prob'],
                                      radial_black_pixels_prob=cfg['augments']['radial_black_pixels_prob'],
                                      satellite_map_crop=cfg['augments']['satellite_map_crop'],)

        """dataset_loader_validation = Loader(dataset_dir_validation, sigma=sigma, augment_data=True, max_data_size=cfg['dataset_parameters']['dataset_size'],
                                satellite_map_size=(256, 256), height_map_size=(128, 128),
                                      satellite_map_augment_max_angle=cfg['augments']['satellite_map_augment_max_angle'],
                                      min_brightness_ratio=cfg['augments']['min_brightness_ratio'],
                                      max_brightness_ratio=cfg['augments']['max_brightness_ratio'],
                                      min_contrast_ratio=cfg['augments']['min_contrast_ratio'],
                                      max_contrast_ratio=cfg['augments']['max_contrast_ratio'],
                                      min_saturation_ratio=cfg['augments']['min_saturation_ratio'],
                                      max_saturation_ratio=cfg['augments']['max_saturation_ratio'],
                                      noise_ratio=cfg['augments']['noise_ratio'],
                                      heightmap_augment_max_angle=cfg['augments']['heightmap_augment_max_angle'],
                                      radial_drop_prob=cfg['augments']['radial_drop_prob'],
                                      radial_black_pixels_prob=cfg['augments']['radial_black_pixels_prob'],)"""
        dataset_loader_validation = Loader(dataset_dir_validation, sigma=sigma, augment_data=True,
                                           max_data_size=cfg['dataset_parameters']['dataset_size'],
                                           satellite_map_size=(256, 256), height_map_size=(128, 128),)

        return dataset_loader_train, dataset_loader_validation

    def train_model_eval(self, height_map_image, satellite_map_image):
        self.optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            sat_emb, lidar_emb = self.model(height_map_image, satellite_map_image)
            loss = self.criterion(sat_emb, lidar_emb)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss

    def train_model_eval_old(self, height_map_image, satellite_map_image):
        self.optimizer.zero_grad()

        sat_emb, lidar_emb = self.model(height_map_image, satellite_map_image)

        loss = self.criterion(sat_emb, lidar_emb)
        loss.backward()

        self.optimizer.step()

        return loss

    def train(self):
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch_data in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            lidar_img, sat_img, _, _ = batch_data

            # move to device
            sat_img = sat_img.to(self.device, dtype=torch.float32)
            lidar_img = lidar_img.to(self.device, dtype=torch.float32)

            loss = self.train_model_eval(lidar_img, sat_img)

            total_loss += loss.item()

        train_loss = total_loss / len(self.train_loader)
        self.writer_train.add_scalar("Loss/compare", train_loss, self.epoch+1)

        wandb.log({
            "epoch": self.epoch+1,
            "train_loss": train_loss,
        }, commit=False)

        print(f"Epoch: {self.epoch}. Train Loss: {train_loss:.4f}")


    def test(self):
        self.model.eval()
        total_loss = 0.0

        all_sat_emb = []
        all_lidar_emb = []

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.test_loader):
                lidar_img, sat_img, _, _ = batch_data

                sat_img = sat_img.to(self.device, dtype=torch.float32)
                lidar_img = lidar_img.to(self.device, dtype=torch.float32)

                sat_emb, lidar_emb = self.model(lidar_img, sat_img)

                loss = self.criterion(sat_emb, lidar_emb)
                total_loss += loss.item()

                all_sat_emb.append(sat_emb.cpu())
                all_lidar_emb.append(lidar_emb.cpu())

        test_loss = total_loss / len(self.test_loader)

        all_sat_emb = torch.cat(all_sat_emb, dim=0)
        all_lidar_emb = torch.cat(all_lidar_emb, dim=0)

        dist_matrix = torch.cdist(all_lidar_emb, all_sat_emb)  # (N_queries, N_gallery)

        gt_indices = torch.arange(len(all_lidar_emb))

        _, top_indices = torch.sort(dist_matrix, dim=1)

        top_1_hits = (top_indices[:, :1] == gt_indices.view(-1, 1)).any(dim=1)
        top_1 = top_1_hits.float().mean().item()

        top_5_hits = (top_indices[:, :5] == gt_indices.view(-1, 1)).any(dim=1)
        top_5 = top_5_hits.float().mean().item()

        top_10_hits = (top_indices[:, :10] == gt_indices.view(-1, 1)).any(dim=1)
        top_10 = top_10_hits.float().mean().item()

        self.writer_test.add_scalar("Loss/compare", test_loss, self.epoch + 1)
        self.writer_test.add_scalar("Accuracy/Top-1", top_1*100, self.epoch + 1)
        self.writer_test.add_scalar("Accuracy/Top-5", top_5*100, self.epoch + 1)
        self.writer_test.add_scalar("Accuracy/Top-10", top_10 * 100, self.epoch + 1)

        ranks = (top_indices == gt_indices.view(-1, 1)).nonzero(as_tuple=True)[1] + 1
        ranks_f = ranks.float()

        mrr = (1.0 / ranks.float()).mean().item()
        best_rank = ranks.min().item()
        worst_rank = ranks.max().item()
        mean_rank = ranks_f.mean().item()
        median_rank = ranks.median().item()

        self.writer_test.add_scalar("Accuracy/MRR", mrr, self.epoch + 1)

        self.writer_test.add_scalar("Rank/Best", best_rank, self.epoch + 1)
        self.writer_test.add_scalar("Rank/Worst", worst_rank, self.epoch + 1)
        self.writer_test.add_scalar("Rank/Mean", mean_rank, self.epoch + 1)
        self.writer_test.add_scalar("Rank/Median", median_rank, self.epoch + 1)

        wandb.log({
            "epoch": self.epoch+1,
            "test_loss": test_loss,
            "Top-1": top_1*100,
            "Top-5": top_5*100,
            "Top-10": top_10 * 100,
            "mrr": mrr,
            "best_rank": best_rank,
            "worst_rank": worst_rank,
            "mean_rank": mean_rank,
            "median_rank": median_rank,
        }, commit=False)

        print(f"Epoch: {self.epoch}. Test Loss: {test_loss:.4f} | Top-1: {top_1*100:.4f} | Top-5: {top_5*100:.4f} | Top-10: {top_10*100:.4f}")


        return test_loss

    def wandb_init(self):
        wandb.init(
            project="Embedder",
            name=f"Run_{self.config['directory']['model_path'].split('/')[-1]}",
            config={
                "architecture": "resnet18-CrossModalNetwork3",
                "initial_learning_rate": self.config['train_parameters']['learning_rate'],
                "batch_size": self.config['train_parameters']['batch_size'],
                "epochs": self.config['train_parameters']['epochs'],
                "embedding_dim": self.config['model_parameters']['embedding_dim'],
                "weight_decay": self.config['train_parameters']['weight_decay'],
                "lr_drop_factor": self.config['train_parameters']['lr_drop_factor'],
                "lr_drop_patience": self.config['train_parameters']['lr_drop_patience'],
                "lr_drop_min": self.config['train_parameters']['lr_drop_min'],
                "init_temperature": self.config['train_parameters']['initial_temperature'],
                "temp_min": self.config['train_parameters']['temp_min'],
                "temp_max": self.config['train_parameters']['temp_max'],
                "sigma": self.config['dataset_parameters']['sigma'],
                "dataset_size": len(self.dataset_loader_train),
                "satellite_map_augment_max_angle": self.config['augments']['satellite_map_augment_max_angle'],
                "min_brightness_ratio": self.config['augments']['min_brightness_ratio'],
                "max_brightness_ratio": self.config['augments']['max_brightness_ratio'],
                "min_contrast_ratio": self.config['augments']['min_contrast_ratio'],
                "max_contrast_ratio": self.config['augments']['max_contrast_ratio'],
                "min_saturation_ratio": self.config['augments']['min_saturation_ratio'],
                "max_saturation_ratio": self.config['augments']['max_saturation_ratio'],
                "noise_ratio": self.config['augments']['noise_ratio'],
                "heightmap_augment_max_angle": self.config['augments']['heightmap_augment_max_angle'],
                "radial_black_pixels_prob": self.config['augments']['radial_black_pixels_prob'],
                "radial_drop_prob": self.config['augments']['radial_drop_prob'],
                "satellite_map_crop": self.config['augments']['satellite_map_crop'],
            }
        )

    def main(self):
        self.writer_train = SummaryWriter(log_dir=f"{self.tensorboard_dir}/train")
        self.writer_test = SummaryWriter(log_dir=f"{self.tensorboard_dir}/test")

        self.wandb_init()

        patience_counter = 0

        for self.epoch in range(self.start_epoch, self.epochs):
            self.train()
            test_loss = self.test()
            self.scheduler.step(test_loss)

            lr = self.optimizer.param_groups[0]["lr"]
            self.writer_train.add_scalar("LearningRate", lr, (self.epoch + 1))

            print(f"Actual temperature: {self.criterion.temperature.item():.4f}")
            wandb.log({
                "epoch": self.epoch + 1,
                "temperature": self.criterion.temperature.item(),
                "learning_rate": lr
            }, commit=True)

            if self.lowest_test_loss > test_loss:
                torch.save(self.model.state_dict(), self.model_path)
                print(f"Model saved to: {self.model_path}")
                self.lowest_test_loss = test_loss

                patience_counter = 0
            else:
                patience_counter += 1

            #self.save_checkpoint(epoch=self.epoch)
            print(f"patience_counter: {patience_counter}")

            if patience_counter >= 10:
                wandb.finish()
                break



class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, sat_emb, lidar_emb):
        logits = torch.matmul(sat_emb, lidar_emb.T) / self.temperature

        labels = torch.arange(sat_emb.size(0), device=sat_emb.device)

        loss_sat = torch.nn.functional.cross_entropy(logits, labels)        # loss of satellite finds own lidar
        loss_lidar = torch.nn.functional.cross_entropy(logits.T, labels)    # loss of lidar finds own satellite

        return (loss_sat + loss_lidar) / 2.0    # average of losses

def custom_collate(batch):
    lidar_images = torch.stack([item[0] for item in batch]) # to 4D array
    sat_images = torch.stack([item[1] for item in batch])   # to 4D array

    ignored_vars = [item[2] for item in batch]
    labels = [item[3] for item in batch]

    return lidar_images, sat_images, ignored_vars, labels

class InfoNCELossTrain(torch.nn.Module):
    def __init__(self, initial_temperature=0.07, temp_min=0.04, temp_max=0.5):
        super(InfoNCELossTrain, self).__init__()
        self.temperature = torch.nn.Parameter(torch.tensor([initial_temperature]))
        self.temp_min = temp_min
        self.temp_max = temp_max

    def forward(self, sat_emb, lidar_emb):
        temp = torch.clamp(self.temperature, min=self.temp_min, max=self.temp_max)

        logits = torch.matmul(sat_emb, lidar_emb.T) / temp

        labels = torch.arange(sat_emb.size(0), device=sat_emb.device)

        loss_sat = torch.nn.functional.cross_entropy(logits, labels)
        loss_lidar = torch.nn.functional.cross_entropy(logits.T, labels)

        return (loss_sat + loss_lidar) / 2.0


if __name__ == '__main__':
    pass
