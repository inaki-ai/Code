from torchvision import transforms
import PIL


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
    test_data_transform = train_data_transform

    transforms_dict = {
        "train": train_data_transform,
        "val": val_data_transform,
        "test": test_data_transform
    }

    return transforms_dict