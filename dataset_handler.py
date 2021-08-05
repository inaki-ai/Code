from PIL import Image
import torch
import os
import pandas as pd
import numpy as np


class Set(torch.utils.data.Dataset):

    """
    Para usar un dataset en pytorch, este tiene que heredar de torch.utils.data.Dataset. Ademas, hay que implementar
    los metodos __len__() de manera que este retorne el tamaño del dastaset y __getitem__ de manera que se pueda
    indexar (ej: dataset[i])
    """

    def __init__(self, data_root_dir, csv_file, id, transform=None, augmentation_pipeline=None):
        """
        En el constructor simplemente se almecenan los datos

        :param csv_file: archivo con las anotaciones
        :param data_root_dir: directorio de las imagenes
        :param transform: transformacion a aplicar a las imagenes
        """
        self.id = id
        self.data = pd.read_csv(os.path.join(data_root_dir, csv_file))
        self.data_root_dir = data_root_dir
        self.transform = transform
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

        full_image_name = os.path.join(self.data_root_dir, self.data.iloc[idx, 0])
        image = Image.open(full_image_name).convert("RGB")

        mask_image_name = os.path.join(self.data_root_dir, self.data.iloc[idx, 1])
        mask = Image.open(mask_image_name).convert("L")

        if "malignant" in full_image_name:
          Class = "malignant"
        elif "benign" in full_image_name:
          Class = "malignant"
        else:
          Class = "normal"

        filename = self.data.iloc[idx, 0]

        if self.augmentation_pipeline:
            image, ground_truth = self.augmentation_pipeline(np.array(image), np.array(ground_truth))

            image = Image.fromarray(image)
            ground_truth = Image.fromarray(ground_truth.draw(size=ground_truth.shape)[0]).convert('L')
        else:
            pass

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)


        sample = {
          "image": image,
          "mask": mask,
          "class": Class,
          "filename": filename
        }
        # En este caso como es clasificacion no hay que transformar labels, si fuera una mascara habria que retornarlos
        # juntos, como una tupla o diccionario: {'image': image, 'mask': mask}
        return sample


class DataSet():

    #transforms = [train_trans, val_trans, test_trans, idem augmentation]

    def __init__(self, data_root_dir, train_csv_file, val_csv_file, test_csv_file, transforms, augmentation_pipelines,
                 batchsize, workers):

        self.trainset = Set(data_root_dir, train_csv_file, "train", transforms["train"],
                            augmentation_pipelines["train"])
        self.valset = Set(data_root_dir, val_csv_file, "val", transforms["val"],
                            augmentation_pipelines["val"])
        self.testset = Set(data_root_dir, test_csv_file, "test", transforms["test"],
                            augmentation_pipelines["test"])

        self.batchsize = batchsize
        self.workers = workers

        self.trainset_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batchsize, shuffle=True,
                                                           num_workers=workers)
        self.valset_loader = torch.utils.data.DataLoader(self.valset, batch_size=batchsize, shuffle=True,
                                                          num_workers=workers)
        self.testset_loader = torch.utils.data.DataLoader(self.testset, batch_size=batchsize, shuffle=True,
                                                          num_workers=workers)

        print(f"Train set length: {len(self.trainset)}")
        print(f"Val set length: {len(self.valset)}")
        print(f"Test set length: {len(self.testset)}")
        print("Mini-batches size:")
        print(f"\tTrain set: {len(self.trainset_loader)}")
        print(f"\tVal set: {len(self.valset_loader)}")
        print(f"\tTest set: {len(self.testset_loader)}")