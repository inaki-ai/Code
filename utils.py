import torch
from torchvision import transforms
import PIL


def merge_images_with_masks(images, masks):

    """
    Genera las imagenes de 4 canales que se pasan al discriminador (3 de la imagen original + 1 con
    la mascara de segmentacion

    :param images: imagenes
    :param masks: mascaras de segmentacion
    :return: tensor con imagenes de 4 canales
    """

    batch_size = images.shape[0]
    img_dim = images.shape[2]
    merged = torch.rand(batch_size, 4, img_dim, img_dim)

    for i in range(batch_size):
        merged[i] = torch.cat((images[i], masks[i]))

    return merged


def load_img_transforms():

    """
    Funcion que carga las transformaciones

    :return:
    """
    train_data_transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=PIL.Image.NEAREST),
        transforms.ToTensor()
    ])

    val_data_transform = train_data_transform

    return train_data_transform, val_data_transform
