from PIL import Image, ImageOps
import torch
import os
import pandas as pd
import numpy as np
from torchvision import transforms
import yaml
import cv2


class Set(torch.utils.data.Dataset):

    """
    Para usar un dataset en pytorch, este tiene que heredar de torch.utils.data.Dataset. Ademas, hay que implementar
    los metodos __len__() de manera que este retorne el tamaño del dastaset y __getitem__ de manera que se pueda
    indexar (ej: dataset[i])
    """

    def __init__(self, csv_file, id, transform=None, augmentation_pipeline=None, cache="disk"):
        """
        En el constructor simplemente se almecenan los datos

        :param csv_file: archivo con las anotaciones
        :param data_root_dir: directorio de las imagenes
        :param transform: transformacion a aplicar a las imagenes
        """
        self.id = id
        self.data = pd.read_csv(csv_file)

        self.cache = cache
        self.transform = transform

        if self.cache == "ram":
            self.images = []
            for idx in range(len(self.data)):
                image = Image.open(self.data.iloc[idx, 0]).convert("RGB")
                mask = Image.open(self.data.iloc[idx, 1]).convert("L")

                self.images.append((image, mask))

        self.augmentation_pipeline = augmentation_pipeline

    def __len__(self):
        """
        En este caso cada fila del csv es una imagen
        :return: longitud del dataset
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Lee de disco y retorna la imagen dataset[idx]
        :param idx: indice a retornar
        :return: imagen y labels correspondientes al indice idx
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()


        if self.cache == "ram":
            image, mask = self.images[idx]
        else:
        # PIL images
            image = Image.open(self.data.iloc[idx, 0]).convert("RGB")
            mask = Image.open(self.data.iloc[idx, 1]).convert("L")

        if "malignant" in self.data.iloc[idx, 0]:
          Class = "malignant"
        elif "benign" in self.data.iloc[idx, 0]:
          Class = "benign"
        else:
          Class = "normal"

        filename = self.data.iloc[idx, 0]

        if self.augmentation_pipeline is not None:
            image, mask = self.augmentation_pipeline(image, mask)

        # Must be PIL images
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        to_tensor = transforms.ToTensor()

        sample = {
          "image": to_tensor(image),
          "mask": to_tensor(mask),
          "class": Class,
          "filename": filename
        }
        # En este caso como es clasificacion no hay que transformar labels, si fuera una mascara habria que retornarlos
        # juntos, como una tupla o diccionario: {'image': image, 'mask': mask}
        return sample


class DataSet():

    def __init__(self, dataset_file, transforms, augmentation_pipelines, batchsize, workers, cache):

        file = open(dataset_file, 'r')
        dataset_files = yaml.safe_load(file)

        self.trainset = Set(dataset_files["train"], "train", transforms["train"],
                            augmentation_pipelines["train"], cache)
        self.valset = Set(dataset_files["val"], "val", transforms["val"],
                            augmentation_pipelines["val"], cache)
        self.testset = Set(dataset_files["test"], "test", transforms["test"],
                            augmentation_pipelines["test"], cache)

        self.batchsize = batchsize
        self.workers = workers

        self.trainset_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batchsize, shuffle=True,
                                                           num_workers=workers)
        self.valset_loader = torch.utils.data.DataLoader(self.valset, batch_size=batchsize, shuffle=True,
                                                          num_workers=workers)
        self.testset_loader = torch.utils.data.DataLoader(self.testset, batch_size=batchsize, shuffle=True,
                                                          num_workers=workers)


def load_dataset(parameter_dict, print_info=True):

    if print_info:
        print("[I] Loading dataset")

    transforms = parameter_dict["transforms"]
    augmentation_pipelines = parameter_dict["augmentation_pipelines"]

    batchsize = parameter_dict["batch_size"]
    workers = parameter_dict["workers"]

    cache = parameter_dict["cache"]

    dataset_file = os.path.join("datasets", parameter_dict["dataset"])

    if not os.path.isfile(dataset_file) and dataset_file.endswith('.yaml'):
        raise Exception(f"Dataset file {dataset_file} does not exist")

    dataset = DataSet(dataset_file, transforms,
                      augmentation_pipelines, batchsize, workers, cache)

    if print_info:
        print("-----------------------------------------------------------------")
        print("[I] DATASET INFO:")
        print(f"\tTrain set length: {len(dataset.trainset)} images")
        print(f"\tVal set length: {len(dataset.valset)} images")
        print(f"\tTest set length: {len(dataset.testset)} images\n")
        print("\n\tMini-batches size:")
        print(f"\t\tTrain set: {len(dataset.trainset_loader)} batches")
        print(f"\t\tVal set: {len(dataset.valset_loader)} batches")
        print(f"\t\tTest set: {len(dataset.testset_loader)} batches")
        print("-----------------------------------------------------------------\n\n")

    return dataset

if __name__ == "__main__":
    from image_transformations import load_img_transforms
    from data_augmentation import load_data_augmentation_pipes
    from hyperparameters import HyperparameterReader

    DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    transforms_dict = load_img_transforms()
    augmentation_dict = load_data_augmentation_pipes(data_aug=True)

    hyperparameter_loader = HyperparameterReader("hyperparameters.yaml")
    parameter_dict = hyperparameter_loader.load_param_dict()

    parameter_dict["transforms"] = transforms_dict
    parameter_dict["augmentation_pipelines"] = augmentation_dict

    dataset = load_dataset(parameter_dict)

    trans = transforms.ToPILImage()

    save_folder = "/workspace/shared_files/aug"

    for i, batched_sample in enumerate(dataset.trainset_loader):

        images, masks, filenames = batched_sample["image"], batched_sample["mask"], batched_sample["filename"]

        for j in range(images.shape[0]):
            image, mask = images[j].to("cpu"), masks[j].to("cpu")
            name = filenames[j].split('/')[-1]

            image_save = trans(image)
            mask_save = trans(mask)

            opencv_image = np.array(image_save)
            opencv_image = opencv_image[:, :, ::-1].copy()
            opencv_gt = np.array(mask_save)

            contours_gt, _ = cv2.findContours(opencv_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(opencv_image, contours_gt, -1, (0, 255, 0), 1)

            #opencv_image = cv2.resize(opencv_image, (512, 512))

            print(os.path.join(save_folder, f"{name}"))
            cv2.imwrite(os.path.join(save_folder, f"{name}"), opencv_image)