from imgaug import augmenters as iaa
import numpy as np
import PIL


class AugmentationPipeline:

    def __init__(self):
        self.spatial_aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            #iaa.Flipud(0.5),
            #iaa.Sometimes(0.15, iaa.PiecewiseAffine(scale=(0.05, 0.1))),
            iaa.Sometimes(0.5,
                          iaa.Sequential([iaa.CropToFixedSize(width=100, height=100),
                                          iaa.Resize((128, 128))])),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
        ])

        self.color_aug = iaa.Sequential([
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 00)),
            #iaa.Sometimes(0.25, iaa.MedianBlur(k=(3, 11)))
            #iaa.AddToHueAndSaturation((-50, 50))
        ])

    def __call__(self, img, mask):

        np_img = np.array(img)
        np_mask = np.array(mask).reshape(np.array(mask).shape[0], np.array(mask).shape[1], 1)
        np_img_mask = np.concatenate((np_img, np_mask), axis=2)

        np_augmented = self.spatial_aug.augment_image(np.array(np_img_mask))

        np_augmented_img = np_augmented[:, :, :3]
        np_augmented_mask = np_augmented[:, :, 3]

        np_augmented_img = self.color_aug.augment_image(np_augmented_img)

        augmented_img = PIL.Image.fromarray(np_augmented_img)
        augmented_mask = PIL.Image.fromarray(np_augmented_mask)

        return augmented_img, augmented_mask


def load_data_augmentation_pipes(data_aug=False):
    if data_aug:

        augmentation_pipe = AugmentationPipeline()

        augmentation_dict = {
            "train": augmentation_pipe,
            "val": None,
            "test": None
        }
    else:
        augmentation_dict = {
            "train": None,
            "val": None,
            "test": None
        }

    return augmentation_dict


if __name__ == "__main__":

    import PIL
    import os

    img_path = "/home/inaki/shared_files/Dataset_BUSI_with_GT/gan_train/benign/benign (200).png"
    mask_path = "/home/inaki/shared_files/Dataset_BUSI_with_GT/masks/benign (200)_mask.png"

    img = PIL.Image.open(img_path).convert("RGB")
    mask = PIL.Image.open(mask_path).convert("L")

    pipe = AugmentationPipeline()

    img = img.resize((128, 128))
    mask = mask.resize((128, 128))

    augmented_img, augmented_mask = pipe(img, mask)

    save_folder = "/home/inaki/shared_files/TFM/Execution/borrar"

    img.save(os.path.join(save_folder, "img.png"))
    mask.save(os.path.join(save_folder, "mask.png"))
    augmented_img.save(os.path.join(save_folder, "augmented_img.png"))
    augmented_mask.save(os.path.join(save_folder, "augmented_mask.png"))
