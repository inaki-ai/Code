def load_data_augmentation_pipes(data_aug=False):
    if data_aug:
        augmentation_dict = {
            "train": None,
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
