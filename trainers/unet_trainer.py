import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from models.segmentors.unet import UNet
from models.segmentors.rdau_net import RDAU_NET

from common.hyperparameters import HyperparameterReader
from common.dataset_handler import load_dataset
from common.image_transformations import load_img_transforms
from common.data_augmentation import load_data_augmentation_pipes
from common.utils import torch_dice_loss
from common.segmentation_metrics import get_evaluation_metrics


class UnetTrainer:

    def __init__(self, hyperparams_file):

        hyperparameter_loader = HyperparameterReader(hyperparams_file)
        self.parameter_dict = hyperparameter_loader.load_param_dict()

        self.parameter_dict["device"] = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        transforms_dict = load_img_transforms()
        augmentation_dict = load_data_augmentation_pipes(data_aug=self.parameter_dict["data_augmentation"])

        self.parameter_dict["transforms"] = transforms_dict
        self.parameter_dict["augmentation_pipelines"] = augmentation_dict

        self.dataset = load_dataset(self.parameter_dict)

        if self.parameter_dict["net"] == "UNet":
            self.model = UNet(3, 1).to(self.parameter_dict["device"])
            self.model.init_weights()
        elif self.parameter_dict["net"] == "RDAUNet":
            self.model = RDAU_NET().to(self.parameter_dict["device"])
            self.model.init_weights()
        else:
            #TODO
            pass

        if self.parameter_dict["optimizer"] == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.parameter_dict["learning_rate"],
                                        betas=(0.9, 0.999))
        else:
            #TODO
            pass

        self.experiment_folder = self.check_experiments_folder()

        if self.parameter_dict["tensorboard"]:
            self.tb_runs_folder = self.check_runs_folder(self.experiment_folder.split('/')[-1])
            self.writer = SummaryWriter(f"{self.tb_runs_folder}")
        else:
            self.writer = None

    @staticmethod
    def compute_loss(prediction, target, bce_weight=0.25):

        bce = F.binary_cross_entropy_with_logits(prediction, target)

        prediction = torch.sigmoid(prediction)

        dice = torch_dice_loss(prediction, target)

        loss = bce * bce_weight + dice * (1 - bce_weight)

        return loss


    def train_step(self):

        self.model.train()
        avg_loss = 0
        
        for i, batched_sample in enumerate(self.dataset.trainset_loader):

            self.optimizer.zero_grad()

            images, masks = batched_sample["image"].to(self.parameter_dict["device"]),\
                            batched_sample["mask"].to(self.parameter_dict["device"])

            output = self.model(images)

            loss = self.compute_loss(output, masks)

            avg_loss += loss.item()

            loss.backward()
            self.optimizer.step()

        return avg_loss / len(self.dataset.trainset_loader)

    def val_step(self):

        self.model.eval()

        with torch.no_grad():
            avg_loss = 0
            
            for i, batched_sample in enumerate(self.dataset.valset_loader):

                images, masks = batched_sample["image"].to(self.parameter_dict["device"]),\
                                batched_sample["mask"].to(self.parameter_dict["device"])

                output = self.model(images)

                loss = self.compute_loss(output, masks)

                avg_loss += loss.item()

        return avg_loss / len(self.dataset.valset_loader)

    def train(self):

        for epoch in range(1, self.parameter_dict["n_epochs"]+1):

            train_loss = self.train_step()
            val_loss = self.val_step()

            print(f"Epoch: {epoch} -- Train loss: {train_loss} - Val loss: {val_loss}")

            if self.writer is not None:
                self.writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)

            metrics = get_evaluation_metrics(self.writer, epoch, self.dataset.valset_loader, self.model,
                                    self.parameter_dict["device"], writer=self.writer,
                                    SAVE_SEGS=True, N_EPOCHS_SAVE=5, folder=f"{self.experiment_folder}/segmentations")


    @staticmethod
    def check_experiments_folder():
        
        if not os.path.isdir("experiments"):
            os.mkdir("experiments")
            os.mkdir("experiments/exp1")
            return "experiments/exp1"
        else:
            numbers = [int(x.replace("exp", "")) for x in os.listdir("experiments")]
            if len(numbers) > 0:
                n_folder = max(numbers)+1
            else:
                n_folder = 1

            os.mkdir(f"experiments/exp{n_folder}")
            return f"experiments/exp{n_folder}"


    @staticmethod
    def check_runs_folder(exp_folder):
        
        if not os.path.isdir("runs"):
            os.mkdir("runs")

        os.mkdir(f"runs/{exp_folder}")
        return f"runs/{exp_folder}/{exp_folder}"




if __name__ == '__main__':

    unet_trainer = UnetTrainer("hyperparameters.yaml")