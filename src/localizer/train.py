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
Directory: src/localizer/
"""

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import yaml
import wandb

from src.loader.loader import Loader
from src.localizer.model import GlobalLocalizationNet
from src.localizer.ranker import heatmap_kl_loss, calculate_distance, calculate_topk_accuracy, calculate_recall_at_radius
from src.shared.trainer import PytorchTrainer
from src.shared.utils import Colors, project_directory


def write_epoch_data(writer, epoch, loss, epoch_duration, distance, probability, top_1, top_5, top_10, radius_3, radius_5, radius_10):
    epoch = epoch + 1

    writer.add_scalar("Loss/compare", loss, epoch)

    writer.add_scalar("Time/epoch", epoch_duration, epoch)

    writer.add_scalar("Dist/compare", distance, epoch)

    writer.add_scalar("Prob/max/compare", probability.max().item(), epoch)

    writer.add_scalar("Top/1", top_1 * 100, epoch)
    writer.add_scalar("Top/5", top_5 * 100, epoch)
    writer.add_scalar("Top/10", top_10 * 100, epoch)

    writer.add_scalar("Rad/3", radius_3 * 100, epoch)
    writer.add_scalar("Rad/5", radius_5 * 100, epoch)
    writer.add_scalar("Rad/10", radius_10 * 100, epoch)


class Trainer(PytorchTrainer):
    def __init__(self):
        directory = project_directory()
        with open(directory / 'configs/localizer.yaml', 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
        sigma = cfg['train_parameters']['sigma']
        dataset_dir = directory / cfg['directory']['pair_dir']
        embedding_dim = cfg['model_parameters']['embedding_dim']

        dataset_dir_train = directory / cfg['directory']['dataset_dir_train']

        dataset_loader = Loader(dataset_dir, sigma=sigma, augment_data=True, max_data_size=-1)
        dataset_loader_train = Loader(dataset_dir_train, sigma=sigma, augment_data=True, heatmap_size=(65,65),
                                      max_data_size=cfg['dataset_parameters']['dataset_size'],
                                      satellite_map_size=(256, 256), height_map_size=(128, 128),
                                      satellite_map_augment_max_angle=cfg['augments'][
                                          'satellite_map_augment_max_angle'],
                                      min_brightness_ratio=cfg['augments']['min_brightness_ratio'],
                                      max_brightness_ratio=cfg['augments']['max_brightness_ratio'],
                                      min_contrast_ratio=cfg['augments']['min_contrast_ratio'],
                                      max_contrast_ratio=cfg['augments']['max_contrast_ratio'],
                                      min_saturation_ratio=cfg['augments']['min_saturation_ratio'],
                                      max_saturation_ratio=cfg['augments']['max_saturation_ratio'],
                                      noise_ratio=cfg['augments']['noise_ratio'],
                                      heightmap_augment_max_angle=cfg['augments']['heightmap_augment_max_angle'],
                                      radial_drop_prob=cfg['augments']['radial_drop_prob'],
                                      radial_black_pixels_prob=cfg['augments']['radial_black_pixels_prob'], )

        dataset_dir_validation = directory / cfg['directory']['dataset_dir_validation']

        dataset_loader_validation = Loader(dataset_dir_validation, sigma=sigma, augment_data=True, heatmap_size=(65,65),
                                           max_data_size=cfg['dataset_parameters']['dataset_size'],
                                           satellite_map_size=(256, 256), height_map_size=(128, 128), )

        model = GlobalLocalizationNet(embed_dim=embedding_dim)
        self.criterion = heatmap_kl_loss

        super().__init__(config=cfg, model_setup=model, dataset_loader_train=dataset_loader_train, dataset_loader_validation=dataset_loader_validation)

        #self.criterion = None
        #self.epoch = None

        self.writer_train = None
        self.writer_test = None

    def move_to_device(self, height_map_image, satellite_map_image, heatmap_reference):
        height_map_image = height_map_image.to(self.device)
        satellite_map_image = satellite_map_image.to(self.device)
        heatmap_reference = heatmap_reference.to(self.device)

        return height_map_image, satellite_map_image, heatmap_reference

    def train_model_eval(self, height_map_image, satellite_map_image, heatmap_reference):
        self.optimizer.zero_grad()

        heatmap_predicted = self.model(height_map_image, satellite_map_image)

        loss = self.criterion(heatmap_predicted, heatmap_reference)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return heatmap_predicted, loss

    def train(self) -> None:
        start_time = time.time()

        self.model.train()

        #new_sigma = max(2.0, 10.0 * (pow(10, -1/50) ** self.epoch))
        #self.train_loader.dataset.dataset.set_sigma(new_sigma)
        #self.writer_train.add_scalar("Sigma", new_sigma, (self.epoch+1))

        total_loss = 0
        total_distance = 0
        total_count = 0

        top1_acc = 0.0
        top5_acc = 0.0
        top10_acc = 0.0

        rad_3 = 0.0
        rad_5 = 0.0
        rad_10 = 0.0

        for batch_idx, batch_data in enumerate(tqdm(self.train_loader, leave=False)):

            height_map_image, satellite_map_image, heatmap_reference, position_reference = batch_data

            # Move to device
            height_map_image, satellite_map_image, heatmap_reference = self.move_to_device(height_map_image, satellite_map_image, heatmap_reference)

            # Run model
            heatmap_predicted, loss = self.train_model_eval(height_map_image, satellite_map_image, heatmap_reference)

            heatmap_reference = heatmap_reference + 1e-8
            heatmap_reference = heatmap_reference / heatmap_reference.sum(dim=[1, 2], keepdim=True)

            distance = calculate_distance(heatmap_predicted, position_reference[0], position_reference[1])
            total_distance += distance
            total_count += heatmap_predicted.size(0)

            self.writer_train.add_scalar("Loss/train_batch", loss.item(), batch_idx + self.epoch * len(self.train_loader))

            total_loss += loss.item()

            gt_cx = position_reference[0].to(self.device)
            valid_mask = gt_cx >= 0
            num_valid_in_batch = valid_mask.sum().item()

            top1_acc += calculate_topk_accuracy(heatmap_predicted, heatmap_reference, k=1, gt_x=gt_cx)
            top5_acc += calculate_topk_accuracy(heatmap_predicted, heatmap_reference, k=5, gt_x=gt_cx)
            top10_acc += calculate_topk_accuracy(heatmap_predicted, heatmap_reference, k=10, gt_x=gt_cx)

            rad_3 += calculate_recall_at_radius(heatmap_predicted, heatmap_reference, radius_px=3, gt_x=gt_cx) * num_valid_in_batch
            rad_5 += calculate_recall_at_radius(heatmap_predicted, heatmap_reference, radius_px=5, gt_x=gt_cx) * num_valid_in_batch
            rad_10 += calculate_recall_at_radius(heatmap_predicted, heatmap_reference, radius_px=10, gt_x=gt_cx) * num_valid_in_batch

        with torch.no_grad():
            prob = F.softmax(heatmap_predicted.view(heatmap_predicted.size(0), -1), dim=1)

        #train_distance = total_distance / len(self.train_loader)
        train_distance = total_distance / total_count
        train_loss = total_loss / len(self.train_loader)

        avg_top1 = top1_acc / len(self.train_loader)
        avg_top5 = top5_acc / len(self.train_loader)
        avg_top10 = top10_acc / len(self.train_loader)

        avg_rad_3 = rad_3 / len(self.train_loader)
        avg_rad_5 = rad_5 / len(self.train_loader)
        avg_rad_10 = rad_10 / len(self.train_loader)

        # calculate elapse time of train epoch
        end_time = time.time()
        time_elapsed = end_time - start_time

        wandb.log({
            "epoch": self.epoch + 1,
            "Train/Loss": train_loss,
            "Train/Distance": train_distance,
            "Train/Top1": avg_top1 * 100,
            "Train/Top5": avg_top5 * 100,
            "Train/Top10": avg_top10 * 100,
            "Train/Rad3": avg_rad_3 * 100,
            "Train/Rad5": avg_rad_5 * 100,
            "Train/Rad10": avg_rad_10 * 100,
            "Learning Rate": self.optimizer.param_groups[0]['lr'],
            "Epoch": self.epoch
        }, commit=False)

        # write data to TensorBoard
        write_epoch_data(writer=self.writer_train, epoch=self.epoch, loss=train_loss, epoch_duration=time_elapsed, distance=train_distance, probability=prob, top_1=avg_top1, top_5=avg_top5, top_10=avg_top10, radius_3=avg_rad_3, radius_5=avg_rad_5, radius_10=avg_rad_10)
        print(f"Train epoch:{self.epoch} [{Colors.Fg.green}✓{Colors.reset}]")

    def wandb_init(self):
        wandb.init(
            project="Localizer",
            name=f"Run_{self.config['directory']['model_path'].split('/')[-1]}",
            config={
                "architecture": "resnet18-GlobalLocalizationNet",
                "initial_learning_rate": self.config['train_parameters']['learning_rate'],
                "batch_size": self.config['train_parameters']['batch_size'],
                "epochs": self.config['train_parameters']['epochs'],
                "embedding_dim": self.config['model_parameters']['embedding_dim'],
                "weight_decay": self.config['train_parameters']['weight_decay'],
                "lr_drop_factor": self.config['train_parameters']['lr_drop_factor'],
                "lr_drop_patience": self.config['train_parameters']['lr_drop_patience'],
                "lr_drop_min": self.config['train_parameters']['lr_drop_min'],
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
            }
        )


    def test(self):
        start_time = time.time()

        self.model.eval()

        test_loss = 0.0
        total_distance = 0

        top1_acc = 0.0
        top5_acc = 0.0
        top10_acc = 0.0

        rad_3 = 0.0
        rad_5 = 0.0
        rad_10 = 0.0

        with torch.no_grad():
            for batch_data in self.test_loader:
                height_map_image, satellite_map_image, heatmap_reference, position_reference = batch_data

                height_map_image, satellite_map_image, heatmap_reference = self.move_to_device(height_map_image,
                                                                                               satellite_map_image,
                                                                                               heatmap_reference)

                pred_heatmap = self. model(height_map_image, satellite_map_image)

                heatmap_reference = heatmap_reference / heatmap_reference.sum(dim=[1, 2], keepdim=True)

                distance = calculate_distance(pred_heatmap, position_reference[0], position_reference[1])
                total_distance += distance

                loss = self.criterion(pred_heatmap, heatmap_reference)
                test_loss += loss.item()

                gt_cx = position_reference[0].to(self.device)

                top1_acc += calculate_topk_accuracy(pred_heatmap, heatmap_reference, k=1, gt_x=gt_cx)
                top5_acc += calculate_topk_accuracy(pred_heatmap, heatmap_reference, k=5, gt_x=gt_cx)
                top10_acc += calculate_topk_accuracy(pred_heatmap, heatmap_reference, k=10, gt_x=gt_cx)

                rad_3 += calculate_recall_at_radius(pred_heatmap, heatmap_reference, radius_px=3, gt_x=gt_cx)
                rad_5 += calculate_recall_at_radius(pred_heatmap, heatmap_reference, radius_px=5, gt_x=gt_cx)
                rad_10 += calculate_recall_at_radius(pred_heatmap, heatmap_reference, radius_px=10, gt_x=gt_cx)

        with torch.no_grad():
            prob = F.softmax(pred_heatmap.view(pred_heatmap.size(0), -1), dim=1)

        test_distance = total_distance / len(self.test_loader)
        test_loss = test_loss / len(self.test_loader)

        avg_top1 = top1_acc / len(self.train_loader)
        avg_top5 = top5_acc / len(self.train_loader)
        avg_top10 = top10_acc / len(self.train_loader)

        avg_rad_3 = rad_3 / len(self.train_loader)
        avg_rad_5 = rad_5 / len(self.train_loader)
        avg_rad_10 = rad_10 / len(self.train_loader)

        # calculate elapse time of test epoch
        end_time = time.time()
        time_elapsed = end_time - start_time

        wandb.log({
            "Test/Loss": test_loss,
            "Test/Distance": test_distance,
            "Test/Top1": avg_top1 * 100,
            "Test/Top5": avg_top5 * 100,
            "Test/Top10": avg_top10 * 100,
            "Test/Rad3": avg_rad_3 * 100,
            "Test/Rad5": avg_rad_5 * 100,
            "Test/Rad10": avg_rad_10 * 100,
            "epoch": self.epoch+1,
        }, commit=False)

        # write data to TensorBoard
        write_epoch_data(writer=self.writer_test, epoch=self.epoch, loss=test_loss, epoch_duration=time_elapsed,
                         distance=test_distance, probability=prob, top_1=avg_top1, top_5=avg_top5, top_10=avg_top10,
                         radius_3=avg_rad_3, radius_5=avg_rad_5, radius_10=avg_rad_10)

        return test_loss

    def main(self):
        if self.load_model:
            #model.load_state_dict(torch.load(config.MODEL_PATH, map_location=self.device))
            self.load_checkpoint()

        #self.model.to(self.device)

        self.writer_train = SummaryWriter(log_dir=f"{self.tensorboard_dir}/train")
        self.writer_test = SummaryWriter(log_dir=f"{self.tensorboard_dir}/test")

        self.wandb_init()

        no_change_best_loss = 0
        for self.epoch in range(self.start_epoch, self.epochs):
            self.train()

            test_loss = self.test()

            self.scheduler.step(test_loss)
            lr = self.optimizer.param_groups[0]["lr"]

            self.writer_train.add_scalar("LearningRate", lr, (self.epoch+1))

            wandb.log({
                "epoch": self.epoch + 1,
                "learning_rate": lr
            }, commit=True)

            if self.lowest_test_loss > test_loss:
                torch.save(self.model.state_dict(), self.model_path)
                print(f"Model saved to: {self.model_path}")
                self.lowest_test_loss = test_loss

                no_change_best_loss = 0
            else:
                no_change_best_loss += 1

            self.save_checkpoint(epoch=self.epoch)

            if no_change_best_loss >= 10:
                wandb.finish()
                break


    def run_on_img(self):
        self.model.eval()

if __name__ == '__main__':
    pass
