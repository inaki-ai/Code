import time
import os
import random
import imgaug
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from models.segmentors.unet import UNet, UNet2
from models.segmentors.rdau_net import RDAU_NET
from models.segmentors.unetpp import UNetpp

from common.hyperparameters import HyperparameterReader
from common.dataset_handler import load_dataset
from common.image_transformations import load_img_transforms
from common.data_augmentation import load_data_augmentation_pipes
from common.utils import torch_dice_loss, check_experiments_folder, check_runs_folder
from common.segmentation_metrics import get_evaluation_metrics
from common.progress_logger import ProgressBar


class UnetTrainer:

    def __init__(self, hyperparams_file):

        self.experiment_folder = check_experiments_folder()

        self.LOG("Launching UnetTrainer...")
        self.LOG(f"Reading hyperparameters from {hyperparams_file}")

        hyperparameter_loader = HyperparameterReader(hyperparams_file)
        self.parameter_dict = hyperparameter_loader.load_param_dict()

        self.LOG("Hyperparameters succesfully read:")
        for key, val in self.parameter_dict.items():
            self.LOG(f"\t{key}: {val}")

        self.set_random_seed(self.parameter_dict["random_seed"])

        self.parameter_dict["device"] = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.LOG("Found device: {}".format(self.parameter_dict["device"]))

        transforms_dict = load_img_transforms()
        self.LOG("Transforms dict load succesfully")

        augmentation_dict = load_data_augmentation_pipes(data_aug=self.parameter_dict["data_augmentation"])
        self.LOG("Data augmentation dict load succesfully")

        self.parameter_dict["transforms"] = transforms_dict
        self.parameter_dict["augmentation_pipelines"] = augmentation_dict

        self.dataset = load_dataset(self.parameter_dict)
        self.LOG("Dataset load succesfully")

        if self.parameter_dict["net"] == "UNet":
            self.model = UNet(3, 1, bilinear=self.parameter_dict["bilinear"]).to(self.parameter_dict["device"])
            self.model.init_weights()
        elif self.parameter_dict["net"] == "RDAUNet":
            self.model = RDAU_NET().to(self.parameter_dict["device"])
            self.model.init_weights()
        elif self.parameter_dict["net"] == "UNet2":
            self.model = UNet2(3, 1).to(self.parameter_dict["device"])
            self.model.init_weights()
        elif self.parameter_dict["net"] == "UNetpp":
            self.model = UNetpp(1).to(self.parameter_dict["device"])
            self.model.init_weights()
        else:
            #TODO
            pass

        self.LOG("Model {} load succesfully".format(self.parameter_dict["net"]))

        if self.parameter_dict["optimizer"] == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.parameter_dict["learning_rate"],
                                        betas=(0.9, 0.999))
        else:
            #TODO
            pass

        if self.parameter_dict["pretrained_weights"]:
            self.load_weights()
            self.LOG("Pretained weights load succesfully")  

        if self.parameter_dict["tensorboard"]:
            self.tb_runs_folder = check_runs_folder(self.experiment_folder.split('/')[-1])
            self.writer = SummaryWriter(f"{self.tb_runs_folder}")
        else:
            self.writer = None

        self.dsc_best =  -1


    @staticmethod
    def set_random_seed(random_seed):
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        imgaug.seed(random_seed)


    def load_weights(self):
        try:
            self.model.load_state_dict(torch.load(self.parameter_dict["pretrained_weights_path"]))

        except:
            path = self.parameter_dict["pretrained_weights_path"]
            raise Exception(f"[E] Pretrained weights do not exist at {path} or they are not compatible")


    def save_weights(self, best=False):

        path = os.path.join(self.experiment_folder, "weights")

        if not os.path.isdir(path):
            os.mkdir(path)

        if not best:
            torch.save(self.model.state_dict(), os.path.join(path, "last.pt"))  
        else:
            torch.save(self.model.state_dict(), os.path.join(path, "best.pt"))


    @staticmethod
    def compute_loss(prediction, target, bce_weight=0.5):

        bce = F.binary_cross_entropy_with_logits(prediction, target)

        prediction = torch.sigmoid(prediction)

        dice = torch_dice_loss(prediction, target)

        loss = bce * bce_weight + dice * (1 - bce_weight)

        return loss


    def train_step(self):

        self.model.train()
        avg_loss = 0

        print("Train step")
        bar = ProgressBar(len(self.dataset.trainset_loader))
        
        for i, batched_sample in enumerate(self.dataset.trainset_loader):

            self.optimizer.zero_grad()

            images, masks = batched_sample["image"].to(self.parameter_dict["device"]),\
                            batched_sample["mask"].to(self.parameter_dict["device"])

            output = self.model(images)

            loss = self.compute_loss(output, masks)

            avg_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            bar.step_bar()

        return avg_loss / len(self.dataset.trainset_loader)

    def val_step(self):

        self.model.eval()

        print("Validation step")
        bar = ProgressBar(len(self.dataset.valset_loader))

        with torch.no_grad():
            avg_loss = 0
            
            for i, batched_sample in enumerate(self.dataset.valset_loader):

                images, masks = batched_sample["image"].to(self.parameter_dict["device"]),\
                                batched_sample["mask"].to(self.parameter_dict["device"])

                output = self.model(images)

                loss = self.compute_loss(output, masks)

                avg_loss += loss.item()

                bar.step_bar()

        return avg_loss / len(self.dataset.valset_loader)

    def train(self):

        self.LOG("Starting training the model...")

        for epoch in range(1, self.parameter_dict["n_epochs"]+1):

            self.LOG(f"Starting epoch {epoch}")

            start = time.time()
            train_loss = self.train_step()
            val_loss = self.val_step()
            end = time.time()
            elapsed = end - start

            msg = f"Epoch {epoch} finished -- Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f} -- Elapsed time: {elapsed:.1f}s"
            print(msg + "\n")
            self.LOG(msg)

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)

            metrics = get_evaluation_metrics(self.writer, epoch, self.dataset.valset_loader, self.model,
                                    self.parameter_dict["device"], writer=self.writer,
                                    SAVE_SEGS=True, N_EPOCHS_SAVE=20, folder=f"{self.experiment_folder}/segmentations")

            self.save_weights()
            self.LOG(f"Last weights saved at epoch {epoch}")

            if metrics.dice > self.dsc_best:
                self.LOG(f"New best value of DSC reach: {metrics.dice:.4f} (last: {self.dsc_best:.4f})")
                self.dsc_best = metrics.dice
                self.save_weights(best=True)
                self.LOG(f"Best weights saved at epoch {epoch}")

            self.LOG_METRICS(metrics, epoch, train_loss, val_loss)

    def validate(self):

        self.load_weights()
        self.model.eval()

        print("[I] Evalutating the model...")

        metrics = get_evaluation_metrics(None, -1, self.dataset.valset_loader, self.model,
                                    self.parameter_dict["device"], None, COLOR=True,
                                    SAVE_SEGS=True, N_EPOCHS_SAVE=5, folder=f"{self.experiment_folder}/segmentations")

        print("\n----------------------------------------------------------------------------")
        print("EVALUTAION RESULTS:")
        print("\tCCR: {:.4f}".format(metrics.CCR))
        print("\tPrecision: {:.4f}".format(metrics.precision))
        print("\tRecall: {:.4f}".format(metrics.recall))
        print("\tSensibility: {:.4f}".format(metrics.sensibility))
        print("\tSpecifity: {:.4f}".format(metrics.specifity))
        print("\tF1 score: {:.4f}".format(metrics.f1_score))
        print("\tJaccard coef: {:.4f}".format(metrics.jaccard))
        print("\tDSC coef: {:.4f}".format(metrics.dice))
        print("\tROC-AUC: {:.4f}".format(metrics.roc_auc))
        print("\tPrecision-recall AUC: {:.4f}".format(metrics.precision_recall_auc))
        print("\tHausdorf error: {:.4f}".format(metrics.hausdorf_error))
        print("----------------------------------------------------------------------------")

    
    def LOG(self, msg):

        file = os.path.join(self.experiment_folder, "log.txt")

        if not os.path.isfile(file):
            with open(file, 'w') as f:
                pass

        with open(file, 'a') as f:

            timestamp = str(datetime.now()).split('.')[0]
            f.write(f"{timestamp}: {msg}\n")

    def LOG_METRICS(self, metrics, epoch, train_loss, val_loss):

        file = os.path.join(self.experiment_folder, "metrics.csv")

        if not os.path.isfile(file):
            with open(file, 'w') as f:
                f.write("epoch,train_loss,val_loss,ccr,precision,recall,sensibility,specifity,f1_score,"+
                       "jaccard_coef,dsc_coef,roc_auc,pr_auc,hausdorf_error\n")

        with open(file, 'a') as f:

            f.write(f"{epoch},{train_loss},{val_loss},{metrics.CCR},{metrics.precision},"
                    f"{metrics.recall},{metrics.sensibility},{metrics.specifity},{metrics.f1_score},"
                    f"{metrics.jaccard},{metrics.dice},{metrics.roc_auc},{metrics.precision_recall_auc},"
                    f"{metrics.hausdorf_error}\n")


if __name__ == '__main__':

    unet_trainer = UnetTrainer("hyperparameters.yaml")