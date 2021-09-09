import torch
import torch.nn as nn
import torch.optim as optim

from models.segmentors.unet import UNet
from common.hyperparameters import HyperparameterReader
from common.dataset_handler import load_dataset
from common.image_transformations import load_img_transforms
from common.data_augmentation import load_data_augmentation_pipes


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
        else:
            #TODO
            pass

        if self.parameter_dict["optimizer"] == "Adam":
            self.optimizerG = optim.Adam(self.model.parameters(), lr=self.parameter_dict["learning_rate"],
                                        betas=(0.9, 0.999))
        else:
            #TODO
            pass


    def train_step(self):
        
        for i, batched_sample in enumerate(self.dataset.trainset_loader):

            images, masks = batched_sample["image"].to(self.DEVICE), batched_sample["mask"].to(self.DEVICE)



if __name__ == '__main__':

    unet_trainer = UnetTrainer("hyperparameters.yaml")