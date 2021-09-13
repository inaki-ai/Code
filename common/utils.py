import torch
import torch.nn as nn

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

def weights_init(m):

    """
    Inicializacion de los pesos de la red

    :param m: red
    :return:
    """
    classname = m.__class__.__name__

    no_init = ["OutConv", "DoubleConv"]

    if classname.find('Conv') != -1:
        if classname not in no_init:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def torch_dice_loss(pred, target, smooth = 1., adapt_values=False):

    pred = pred.contiguous()
    target = target.contiguous()

    if adapt_values:
        pred[pred >= 0.5] = 1.0
        pred[pred < 0.5] = 0.0

    loss = (1 - ((2. * (pred * target).sum(dim=2).sum(dim=2) + smooth) / \
            (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()