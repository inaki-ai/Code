from dataset_handler import load_dataset
import torch
from gp_wgan import GP_WGAN
from image_transformations import load_img_transforms
from data_augmentation import load_data_augmentation_pipes

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

TRAIN = True

"""
- PENDIENTE:
    ** Transformaciones (BN o COLOR)
    ** Data augmentation
    * Generar el dataset (aunquesea el temporal sin BUSIS)
    * Revisar metricas y distancia de H
    ** Lr variable
    ** ADAM opt
    ** metodo test_inference (testset)
    * Descomentar linea 59 datasset_handler.py con los excel nuevos de datos
    ** weights init de los modelos
"""

def main():
    DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(f"DEVICE: {DEVICE}")

    root_dir = "/workspace/shared_files/TFM/Execution"

    transforms_dict = load_img_transforms()
    augmentation_dict = load_data_augmentation_pipes(data_aug=True)

    dataset_param_dict = {
        "data_root_dir": "/workspace/shared_files/Dataset_BUSI_with_GT",
        "train_csv_file": "train.csv",
        "val_csv_file": "val.csv",
        "test_csv_file": "gan_test_bus_images.csv",
        "transforms": transforms_dict,
        "augmentation_pipelines": augmentation_dict,
        "batchsize": 16,
        "workers": 9
    }
    dataset = load_dataset(dataset_param_dict)

    wgan_param_dict = {
        "n_epochs": 1500,
        "initial_epoch": 1,
        "generator_lr": 8e-4,
        "critic_lr": 2e-4,
        "adaptative_lr": False,
        "n_critic": 5,
        "generator": "RDAU-NET", # "RDAU-NET", "U-NET"
        "critic": "Critic",
        "device": DEVICE,
        "pretained_weights": True or not TRAIN,
        "pretained_weights_path": "pretrained_weights",
        "generator_weights_file": "generator_weights",
        "critic_weights_file": "critic_weights",
        "optimizer": "RMSprop", # RMSprop, Adam
        "log_file": "execution_log.txt",
        "random_seed": 1,
        "critic_input_type": "concatenation" # masked_img or concatenation
    }
    gan = GP_WGAN(root_dir, wgan_param_dict, dataset, writer="tensorboard")

    if TRAIN:
        gan.train()
    else:
        gan.test_segmenter()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
