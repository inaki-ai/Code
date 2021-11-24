import os
import random
random.seed(42)


def get_image_files(folder, dataset_abs_path_in_execution, dataset_abs_path_now):

    img_files = [os.path.join(dataset_abs_path_in_execution, 'images', folder, f) for f in os.listdir(os.path.join(dataset_abs_path_now, 'images', folder)) if '.png' in f]
    if folder == 'BUSI':
        img_files = remove_axila_imgs(img_files)

    return img_files

def remove_axila_imgs(images):
    benign_axila = [199, 205, 207, 210, 223, 225, 228, 235, 236, 243, 247, 248, 256, 257, 262, 292, 293, 294, 298, 306]
    malignant_axila = [11, 12, 13, 109, 110, 113, 148]

    filtered_images = []

    for image in images:
        image_name = image.split('/')[-1]
        append_img = True

        if 'benign' in image_name:
            for number in benign_axila:
                if f'({number})' in image_name:
                    append_img = False
                    break

        elif 'malignant' in image_name:
            for number in malignant_axila:
                if f'({number})' in image_name:
                    append_img = False
                    break

        if append_img:
            filtered_images.append(image)

    return filtered_images



if __name__ == '__main__':

    csvs_save_path = '/home/imartinez/Code/datasets/Dataset6'
    csv_file_names = ['train_dataset.csv', 'val_dataset.csv', 'test_dataset.csv']

    val_set_prop = 0.075
    test_set_prop = 0.15

    dataset_abs_path_in_execution = '/home/imartinez/Dataset_TFM/'
    dataset_abs_path_now = '/home/imartinez/Dataset_TFM/'
    image_folders = ['BUSI', 'DatasetB', 'ExpandedUnetPaper', 'BUSIS']
    #image_folders = ['BUSI', 'Dataset_paper1-12', 'ExpandedUnetPaper', 'BUSIS']
    #image_folders = ['DatasetB']

    #####
    image_files = []
    for folder in image_folders:
        image_files += get_image_files(folder, dataset_abs_path_in_execution, dataset_abs_path_now)

    random.shuffle(image_files)

    val_set_images_q = int(len(image_files) * val_set_prop)

    test_set_images_q = int(len(image_files) * test_set_prop)

    test_set_images = random.sample(image_files, test_set_images_q)
    val_set_images = random.sample(list(set(image_files) - set(test_set_images)), val_set_images_q)
    train_set_images = list(set(list(set(image_files) - set(test_set_images))) - set(val_set_images))

    print(f"Train set: {len(train_set_images)} images")
    print(f"Val set: {len(val_set_images)} images")
    print(f"Test set: {len(test_set_images)} images")
    print(f"TOTAL: {len(image_files)} images\n")

    if not os.path.isdir(csvs_save_path):
        os.mkdir(csvs_save_path)

    images_per_set = [train_set_images, val_set_images, test_set_images]

    for image_set, filename in zip(images_per_set, csv_file_names):

        with open(os.path.join(csvs_save_path, filename), 'w') as file:
            file.write("image_path,gt_path,label\n")
            for image_file in image_set:
                image_path = image_file
                gt_path = image_file.replace('images', 'gt')

                image_name = image_file.split('/')[-1]

                possible_labels = ['benign', 'malignant', 'normal']
                for label in possible_labels:
                    if label in image_name:
                        img_label = label
                        break

                file.write(f'{image_path},{gt_path},{img_label}\n')

        print(f'File {filename} successfully generated with {len(image_set)} instances')
