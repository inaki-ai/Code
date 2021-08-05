import os
from torch.utils.tensorboard import SummaryWriter
from rdau_net import RDAU_NET
from discriminator import Discriminator
import torch
import torch.optim as optim
from utils import *
import time
from dataset_handler import DataSet
import numpy as np
from segmentation_metrics import get_evaluation_metrics

class WGAN:

    def __init__(self, root_dir, parameter_dict, dataset, writer=None):

        self.root_dir = root_dir

        self.N_EPOCHS = parameter_dict["n_epochs"]
        self.INITIAL_EPOCH = parameter_dict["initial_epoch"]
        self.GENERATOR_LEARNING_RATE = parameter_dict["generator_lr"]
        self.CRITIC_LEARNING_RATE = parameter_dict["critic_lr"]
        self.N_CRITIC = parameter_dict["n_critic"]

        self.DEVICE = parameter_dict["device"]

        self.PRETRAINED_WEIGHTS = parameter_dict["pretained_weights"]
        self.PRETRAINED_WEIGHTS_PATH = parameter_dict["pretained_weights_path"]

        self.OPTIMIZER = parameter_dict["optimizer"]

        self.LOG_FILE = parameter_dict["log_file"]

        self.dataset = dataset

        if writer is not None:
            if not os.path.isdir(os.path.join(root_dir, writer)):
                os.mkdir(os.path.join(root_dir, writer))
            self.writer = SummaryWriter(os.path.join(root_dir, writer))
        else:
            self.writer = None

        self.init_wgan()

        with open(os.path.join(self.root_dir, self.LOG_FILE), 'w') as file:
            file.write("epoch,generator_loss,critic_loss,ccr,precision,recall,sensibility,specifity,f1_score,"
                       "jaccard_coef,dsc_coef,roc_auc,pr_auc,hausdorf_error\n")

    def init_wgan(self):

        self.generator = RDAU_NET().to(self.DEVICE)
        self.generator.init_weights()

        self.critic = Discriminator().to(self.DEVICE)
        self.critic.init_weights()

        if self.PRETRAINED_WEIGHTS:
            self.generator.load_state_dict(torch.load(os.path.join(self.root_dir, self.PRETRAINED_WEIGHTS_PATH,
                                                                   "generator_weights")))
            self.critic.load_state_dict(torch.load(os.path.join(self.root_dir, self.PRETRAINED_WEIGHTS_PATH,
                                                                       "critic_weights")))

        if self.OPTIMIZER == "Adam":
            pass
        elif self.OPTIMIZER == "RMSprop":
            self.optimizerG = optim.RMSprop(self.generator.parameters(), lr=self.GENERATOR_LEARNING_RATE)
            self.optimizerC = optim.RMSprop(self.critic.parameters(), lr=self.CRITIC_LEARNING_RATE)


    def train(self):

        forward_passed_batches = 0
        for epoch in range(self.INITIAL_EPOCH, self.INITIAL_EPOCH+self.N_EPOCHS):

            t_init = time.time()
            print(f"Starting epoch {epoch}")

            G_losses = []
            C_losses = []

            for i, batched_sample in enumerate(self.dataset.trainset_loader):

                images, masks = batched_sample["image"].to(self.DEVICE), batched_sample["mask"].to(self.DEVICE)

                loss_C = self.critic_step(images, masks)
                C_losses.append(loss_C.item())
                if self.writer is not None:
                    self.writer.add_scalar("Train loss/critic", loss_C.item(),
                                           epoch * len(self.dataset.trainset_loader) + i)
                forward_passed_batches += 1

                if forward_passed_batches == self.N_CRITIC:
                    loss_G = self.generator_step(images)
                    G_losses.append(loss_G.item())
                    if self.writer is not None:
                        self.writer.add_scalar("Train loss/generator", loss_G.item(),
                                               epoch * len(self.dataset.trainset_loader) + i)
                    forward_passed_batches = 0



            t_end = time.time()

            if self.writer is not None:
                self.writer.add_scalar("Per epoch train loss/generator", np.mean(np.array(G_losses)), epoch)
                self.writer.add_scalar("Per epoch train loss/discriminator", np.mean(np.array(C_losses)), epoch)

            self.validation_step(epoch, np.mean(np.array(G_losses)), np.mean(np.array(C_losses)))

            print("Epoch {} finished in {:.1f} seconds".format(epoch, t_end-t_init))


    def generator_step(self, images):

        self.optimizerG.zero_grad()

        segmentations = self.generator(images)
        images_with_segmentations = merge_images_with_masks(images, segmentations).to(self.DEVICE)

        loss_G = -torch.mean(self.critic(images_with_segmentations))

        loss_G.backward()
        self.optimizerG.step()

        return loss_G

    def critic_step(self, images, masks):

        self.optimizerC.zero_grad()

        images_with_masks = merge_images_with_masks(images, masks).to(self.DEVICE)

        segmentations = self.generator(images).detach()
        #segmentations = torch.autograd.Variable((segmentations > 0.5).float(), requires_grad=True)
        images_with_segmentations = merge_images_with_masks(images, segmentations).to(self.DEVICE)

        loss_C = -torch.mean(self.critic(images_with_masks)) + torch.mean(self.critic(images_with_segmentations))
        _gradient_penalty = self.gradient_penalty(self.critic, images_with_masks, images_with_segmentations,
                                                  10*2, self.DEVICE)
        loss_C += _gradient_penalty

        loss_C.backward()

        self.optimizerC.step()

        return loss_C

    def validation_step(self, epoch, generator_loss, critic_loss):

        metrics = get_evaluation_metrics(epoch, self.dataset.trainset_loader, self.generator, self.DEVICE, SAVE_SEGS=True,
                                         writer=self.writer, COLOR=True, N_EPOCHS_SAVE=10,
                                         folder=os.path.join(self.root_dir, "Samples"))

        with open(os.path.join(self.root_dir, self.LOG_FILE), 'a') as file:
            file.write(f"{epoch},{generator_loss},{critic_loss},{metrics.CCR},{metrics.precision},"
                       f"{metrics.recall},{metrics.sensibility},{metrics.specifity},{metrics.f1_score},"
                       f"{metrics.jaccard},{metrics.dice},{metrics.roc_auc},{metrics.precision_recall_auc},"
                       f"{metrics.hausdorf_error}\n")

    @staticmethod
    def gradient_penalty(critic, real_segmentations, generated_segmentations, penalty, device):

        n_elements = real_segmentations.nelement()
        batch_size = real_segmentations.size()[0]
        colors = real_segmentations.size()[1]
        image_width = real_segmentations.size()[2]
        image_height = real_segmentations.size()[3]
        alpha = torch.rand(batch_size, 1).expand(batch_size, int(n_elements / batch_size)).contiguous()
        alpha = alpha.view(batch_size, colors, image_width, image_height).to(device)

        fake_data = generated_segmentations.view(batch_size, colors, image_width, image_height)
        interpolates = alpha * generated_segmentations.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.to(device)
        interpolates.requires_grad_(True)
        critic_interpolates = critic(interpolates)

        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(critic_interpolates.size()).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty

        return gradient_penalty

if __name__ == "__main__":

    DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    param_dict = {
        "n_epochs": 450,
        "initial_epoch": 1,
        "generator_lr": 5e-4,
        "critic_lr": 5e-4,
        "n_critic": 5,
        "device": DEVICE,
        "pretained_weights": False,
        "pretained_weights_path": "/fff",
        "optimizer": "RMSprop",
        "log_file": "execution_log.txt",
        "random_seed": 42
    }

    root_dir = "/workspace/shared_files/TFM/Execution"

    train_data_transform, val_data_transform = load_img_transforms()

    transforms_dict = {
        "train": train_data_transform,
        "val": train_data_transform,
        "test": train_data_transform
    }

    augmentation_dict = {
        "train": None,
        "val": None,
        "test": None
    }

    TRAIN_DATA_DIR = "/workspace/shared_files/Dataset_BUSI_with_GT"
    TRAIN_DATA_ANNOTATIONS_FILE = "gan_train_bus_images.csv"
    dataset = DataSet(TRAIN_DATA_DIR, TRAIN_DATA_ANNOTATIONS_FILE, TRAIN_DATA_ANNOTATIONS_FILE,
                      TRAIN_DATA_ANNOTATIONS_FILE, transforms_dict, augmentation_dict, 16, 2)

    wgan = WGAN(root_dir, param_dict, dataset, writer="tensorboard")

    wgan.train()
