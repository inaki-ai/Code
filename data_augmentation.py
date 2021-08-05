from imgaug import augmenters as iaa
import numpy as np
import PIL


class AugmentationPipeline:

    def __init__(self):
        self.spatial_aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
        ])

        self.color_aug = iaa.Sequential([
            iaa.Sometimes(0.99,
                            iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                        iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-100, 100), per_channel=True)
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

    img_path = "/workspace/shared_files/Dataset_BUSI_with_GT/gan_train/benign/benign (200).png"
    mask_path = "/workspace/shared_files/Dataset_BUSI_with_GT/masks/benign (200)_mask.png"

    img = PIL.Image.open(img_path).convert("RGB")
    mask = PIL.Image.open(mask_path).convert("L")

    pipe = AugmentationPipeline()

    augmented_img, augmented_mask = pipe(img, mask)

    save_folder = "/workspace/shared_files/TFM/Execution/borrar"

    img.save(os.path.join(save_folder, "img.png"))
    mask.save(os.path.join(save_folder, "mask.png"))
    augmented_img.save(os.path.join(save_folder, "augmented_img.png"))
    augmented_mask.save(os.path.join(save_folder, "augmented_mask.png"))