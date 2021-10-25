import time
import os
import random
import imgaug
from datetime import datetime
import mlflow
from mlflow.tracking.fluent import set_experiment
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision import transforms

from models.segmentors.unet import UNet, UNet2
from models.segmentors.rdau_net import RDAU_NET
from models.segmentors.unetpp import UNetpp
from self_attention_cv.transunet import TransUnet

from common.hyperparameters import HyperparameterReader
from common.dataset_handler import load_dataset
from common.image_transformations import load_img_transforms, UnNormalize
from common.data_augmentation import load_data_augmentation_pipes
from common.utils import torch_dice_loss, check_experiments_folder, check_runs_folder
from common.segmentation_metrics import get_evaluation_metrics
from common.progress_logger import ProgressBar
from common.utils import generate_output_img

import mlflow

class UnetTrainer:

    def __init__(self, hyperparams_file):

        self.experiment_folder = check_experiments_folder()

        hyperparameter_loader = HyperparameterReader(hyperparams_file)
        self.parameter_dict = hyperparameter_loader.load_param_dict()

        mlflow.set_tag('Exp', self.experiment_folder)
        mlflow.log_artifacts(self.experiment_folder, artifact_path="log")

        self.LOG("Launching UnetTrainer...")
        self.LOG("Hyperparameters succesfully read from {hyperparams_file}:")
        for key, val in self.parameter_dict.items():
            self.LOG(f"\t{key}: {val}")
            mlflow.log_param(key, val)

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
        elif self.parameter_dict["net"] == "TransUNet":
            self.model = TransUnet(in_channels=3, img_dim=128, vit_blocks=self.parameter_dict["vit_blocks"],
                                    vit_dim_linear_mhsa_block=self.parameter_dict["vit_dim"],
                                    classes=1).to(self.parameter_dict["device"])
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

        self.un_normalizer = UnNormalize(mean=0.485, std=0.225)

        self.dsc_best =  -1


    @staticmethod
    def set_random_seed(random_seed):
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        imgaug.seed(random_seed)
        np.random.seed(random_seed)


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

            filenames = batched_sample["filename"]

            if self.parameter_dict['adversarial_training']:
                images.requires_grad = True

            output = self.model(images)

            loss = self.compute_loss(output, masks)

            avg_loss += loss.item()

            if self.parameter_dict['adversarial_training']:
                loss.backward(retain_graph=True)
                images_grad = images.grad.data
                perturbed_images = self.fgsm_attack(images, images_grad, epsilon=self.parameter_dict['epsilon'])
                output = self.model(perturbed_images)
                loss = self.compute_loss(output, masks)
            
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

            mlflow.log_metric("train_loss", train_loss)
            mlflow.log_metric("val_loss", val_loss)

            msg = f"Epoch {epoch} finished -- Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f} -- Elapsed time: {elapsed:.1f}s"
            print(msg + "\n")
            self.LOG(msg)

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)

            metrics = get_evaluation_metrics(self.writer, epoch, self.dataset.valset_loader, self.model,
                                    self.parameter_dict["device"], writer=self.writer,
                                    SAVE_SEGS=True, N_EPOCHS_SAVE=5, folder=f"{self.experiment_folder}/segmentations")

            mlflow.log_metric("CCR", metrics.CCR, step=epoch)
            mlflow.log_metric("Precision", metrics.precision, step=epoch)
            mlflow.log_metric("Recall - Sensitivity", metrics.recall, step=epoch)
            mlflow.log_metric("Specifity", metrics.specifity, step=epoch)
            mlflow.log_metric("F1 score", metrics.f1_score, step=epoch)
            mlflow.log_metric("Jaccard coef - IoU", metrics.jaccard, step=epoch)
            mlflow.log_metric("Dice score - DSC", metrics.dice, step=epoch)
            mlflow.log_metric("ROC AUC", metrics.roc_auc, step=epoch)
            mlflow.log_metric("Precision-recall AUC", metrics.precision_recall_auc, step=epoch)
            mlflow.log_metric("Hausdorf error", metrics.hausdorf_error, step=epoch)
            
            self.save_weights()
            self.LOG(f"Last weights saved at epoch {epoch}")

            if metrics.dice > self.dsc_best:

                mlflow.log_metric("CCR_best", metrics.CCR, step=epoch)
                mlflow.log_metric("Precision_best", metrics.precision, step=epoch)
                mlflow.log_metric("Recall - Sensitivity_best", metrics.recall, step=epoch)
                mlflow.log_metric("Specifity_best", metrics.specifity, step=epoch)
                mlflow.log_metric("F1 score_best", metrics.f1_score, step=epoch)
                mlflow.log_metric("Jaccard coef - IoU_best", metrics.jaccard, step=epoch)
                mlflow.log_metric("Dice score - DSC_best", metrics.dice, step=epoch)
                mlflow.log_metric("ROC AUC_best", metrics.roc_auc, step=epoch)
                mlflow.log_metric("Precision-recall AUC_best", metrics.precision_recall_auc, step=epoch)
                mlflow.log_metric("Hausdorf error_best", metrics.hausdorf_error, step=epoch)

                self.LOG(f"New best value of DSC reach: {metrics.dice:.4f} (last: {self.dsc_best:.4f})")
                self.dsc_best = metrics.dice
                self.save_weights(best=True)
                self.LOG(f"Best weights saved at epoch {epoch}")

            self.LOG_METRICS(metrics, epoch, train_loss, val_loss)


    def test(self):

        self.load_weights()
        self.model.eval()

        print("[I] Evalutating the model...")

        metrics = get_evaluation_metrics(None, -1, self.dataset.testset_loader, self.model,
                                    self.parameter_dict["device"], None, COLOR=True,
                                    SAVE_SEGS=True, N_EPOCHS_SAVE=5, folder=f"{self.experiment_folder}/segmentations")

        print("\n----------------------------------------------------------------------------")
        print("EVALUTAION RESULTS ON TEST SET:")
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
        print(f"Segmentations saved at {self.experiment_folder}/segmentations")

    
    @staticmethod
    def fgsm_attack(image, data_grad, epsilon=0.01):

        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, -2, 2)
        # Return the perturbed image
        return perturbed_image


    def generate_adversarial_examples(self):

        self.load_weights()
        self.model.eval()

        trans = transforms.ToPILImage()

        for i, batched_sample in enumerate(self.dataset.testset_loader):

            images, masks = batched_sample["image"].to(self.parameter_dict["device"]),\
                            batched_sample["mask"].to(self.parameter_dict["device"])

            filenames = batched_sample["filename"]

            images.requires_grad = True

            output = self.model(images)

            loss = self.compute_loss(output, masks)

            self.model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            perturbed_data = self.fgsm_attack(images, images.grad.data)
            output = self.model(perturbed_data)

            hard_sigmoid = nn.Hardsigmoid()
            output = hard_sigmoid(output)

            for j in range(perturbed_data.shape[0]):
                image, mask = perturbed_data[j].to("cpu"), masks[j].to("cpu")
                segmentation = output[j].to("cpu")
                mask_save = trans(mask)
                name = filenames[j].split('/')[-1]

                image_save = trans(image.mul_(0.225).add_(0.485))
                segmentation_save = trans(segmentation)

                opencv_image = np.array(image_save)
                opencv_image = opencv_image[:, :, ::-1].copy()
                opencv_gt = np.array(mask_save)
                opencv_segmentation = np.array(segmentation_save)

                save_image = generate_output_img(opencv_image, opencv_gt, opencv_segmentation)
                cv2.imwrite(os.path.join('/workspace/shared_files/pruebas', f"{name}"), save_image)


        """
            def train_step(self):

        self.model.train()
        avg_loss = 0

        print("Train step")
        bar = ProgressBar(len(self.dataset.trainset_loader))
        
        for i, batched_sample in enumerate(self.dataset.trainset_loader):

            self.optimizer.zero_grad()

            images, masks = batched_sample["image"].to(self.parameter_dict["device"]),\
                            batched_sample["mask"].to(self.parameter_dict["device"])

            filenames = batched_sample["filename"]

            if self.parameter_dict['adversarial_training']:
                images.requires_grad = True

            output = self.model(images)

            loss = self.compute_loss(output, masks)

            avg_loss += loss.item()

            if self.parameter_dict['adversarial_training']:
                loss.backward(retain_graph=True)
                images_grad = images.grad.data
                perturbed_images = self.fgsm_attack(images, images_grad)
                output = self.model(perturbed_images)
                loss = self.compute_loss(output, masks)

                for j in range(perturbed_images.shape[0]):
                    image, mask = perturbed_images[j].to("cpu"), masks[j].to("cpu")
                    segmentation = output[j].to("cpu")
                    name = filenames[j].split('/')[-1]
                    trans = trans = transforms.ToPILImage()
                    image_save = trans(image.mul_(0.225).add_(0.485))
                    mask_save = trans(mask)
                    
                    # guardar
            
            loss.backward()
            self.optimizer.step()

            bar.step_bar()

        return avg_loss / len(self.dataset.trainset_loader)
        """

    def LOG(self, msg):

        file = os.path.join(self.experiment_folder, "log.txt")

        mlflow.log_artifacts(self.experiment_folder, artifact_path="log")

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