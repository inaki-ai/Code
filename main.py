from dataset_handler import load_dataset
import torch
from wgan import WGAN
from image_transformations import load_img_transforms
from data_augmentation import load_data_augmentation_pipes
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

"""
- PENDIENTE:
    ** Transformaciones (BN o COLOR)
    * Data augmentation
    * Generar el dataset (aunquesea el temporal sin BUSIS)
    * Revisar metricas y distancia de H
    * Lr variable
    * revisar el merge (se puede probar a pasar la imagen mascarada o pasar solo la segmentacion)
    ** ADAM opt
    * metodo test_inference (testset)
    * Descomentar linea 59 datasset_handler.py con los excel nuevos de datos
"""

def main():
    DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    root_dir = "/workspace/shared_files/TFM/Execution"

    transforms_dict = load_img_transforms()
    augmentation_dict = load_data_augmentation_pipes()

    dataset_param_dict = {
        "data_root_dir": "/workspace/shared_files/Dataset_BUSI_with_GT",
        "train_csv_file": "gan_train_bus_images",
        "val_csv_file": "gan_val_bus_images",
        "test_csv_file": "gan_val_bus_images",
        "transforms": transforms_dict,
        "augmentation_pipelines": augmentation_dict,
        "batchsize": 16,
        "workers": 9
    }
    dataset = load_dataset(dataset_param_dict)

    wgan_param_dict = {
        "n_epochs": 750,
        "initial_epoch": 1,
        "generator_lr": 75e-4,
        "critic_lr": 75e-4,
        "n_critic": 5,
        "generator": "RDAU-NET",
        "critic": "Critic",
        "device": DEVICE,
        "pretained_weights": False,
        "pretained_weights_path": "pretrained_weights",
        "generator_weights_file": "generator_weights",
        "critic_weights_file": "critic_weights",
        "optimizer": "Adam",
        "log_file": "execution_log.txt",
        "random_seed": 42
    }
    wgan = WGAN(root_dir, wgan_param_dict, dataset, writer="tensorboard")
    wgan.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
