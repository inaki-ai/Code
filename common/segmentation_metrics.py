import numpy as np
import torch
from torchvision import transforms
import cv2
import os
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from common.utils import *
import warnings
warnings.filterwarnings("ignore")
from common.progress_logger import ProgressLogger
from hausdorff import hausdorff_distance
from common.utils import generate_output_img
import seaborn as sns
import matplotlib.pyplot as plt


class SegmentationEvaluationMetrics:

    def __init__(self, CCR, precision, recall, sensibility, specifity, f1_score,
                 jaccard, dice, roc_auc, precision_recall_auc, hausdorf_error):
        self.CCR = CCR
        self.precision = precision
        self.recall = recall
        self.sensibility = sensibility
        self.specifity = specifity
        self.f1_score = f1_score
        self.jaccard = jaccard
        self.dice = dice
        self.roc_auc = roc_auc
        self.precision_recall_auc = precision_recall_auc
        self.hausdorf_error = hausdorf_error


def compute_jaccard_dice_coeffs(mask1, mask2):
    """Calculates the dice coefficient for the images"""


    mask1 = np.asarray(mask1).astype(np.bool)
    mask2 = np.asarray(mask2).astype(np.bool)

    if mask1.shape != mask2.shape:
        raise ValueError("Shape mismatch: mask1 and mask2 must have the same shape.")

    mask1 = mask1 > 0.5
    mask2 = mask2 > 0.5

    im_sum = mask1.sum() + mask2.sum()

    if im_sum == 0:
        return 1.0, 1.0

    intersection = np.logical_and(mask1, mask2).sum()
    union = im_sum - intersection

    return intersection / union, 2. * intersection / im_sum

def compute_hausdorff_dist(im1, im2):
    """Calculates the jaccard coefficient for the images"""

    im1 = np.asarray(im1).astype(np.int32)
    im2 = np.asarray(im2).astype(np.int32)

    im1 = im1.reshape(im1.shape[1], im1.shape[2])
    im2 = im2.reshape(im2.shape[1], im2.shape[2])

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    distance = hausdorff_distance(im1, im2, distance="euclidean")
    return distance


def get_conf_mat(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    prediction = prediction > 0.5
    groundtruth = groundtruth > 0.5

    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))


    return FP, FN, TP, TN


def get_evaluation_metrics(logger, epoch, dataloader, segmentor, DEVICE, writer=None, SAVE_SEGS=False, COLOR=True,
                           N_EPOCHS_SAVE=10, folder=""):

    if not os.path.isdir(folder):
        os.mkdir(folder)

    if not epoch == -1:
        save_folder = os.path.join(folder, f"epoch_{epoch}")
    else:
        save_folder = os.path.join(folder, "segmentations")

    if SAVE_SEGS and (epoch % N_EPOCHS_SAVE == 0 or epoch == -1):
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)


    ccrs = []

    precisions = []
    recalls = []

    sensibilities = []
    specifities = []

    f1_scores = []

    jaccard_coefs = []
    dice_coeffs = []

    roc_auc_coeffs = []
    precision_recall_auc_coeffs = []

    hausdorf_errors = []

    segmentor.eval()

    with open(os.path.join(folder, 'metrics.csv'), 'w') as file:
        file.write('image_id,ccr,precision,recall,specififty,f1_score,jaccard,dsc,roc_auc,pr_auc\n')

    with torch.no_grad():

        for i, batched_sample in enumerate(dataloader):

            images, masks, filenames = batched_sample["image"].to(DEVICE), batched_sample["mask"].to(DEVICE), \
                                     batched_sample["filename"]

            hard_sigmoid = nn.Hardsigmoid()
            segmentations = hard_sigmoid(segmentor(images))
            segmentation_values = segmentations
            segmentations = torch.autograd.Variable((segmentations > 0.5).float())

            trans = transforms.ToPILImage()

            for j in range(images.shape[0]):
                image, mask = images[j].to("cpu"), masks[j].to("cpu")
                segmentation = segmentations[j].to("cpu")
                segmentation_val = segmentation_values[j].to("cpu")
                name = filenames[j].split('/')[-1]

                FP, FN, TP, TN = get_conf_mat(segmentation.numpy(), mask.numpy())

                ccr = np.divide(TP + TN, FP + FN + TP + TN)

                precision = np.divide(TP, TP + FP)
                recall = np.divide(TP, TP + FN)

                sensibility = np.divide(TP, TP + FN)
                specifity = np.divide(TN, TN + FP)

                f1_score = 2 * np.divide(precision * recall, precision + recall)

                jaccard_coef, dice_coeff = compute_jaccard_dice_coeffs(segmentation.numpy(), mask.numpy())

                mask_labels = mask.numpy().ravel().astype(np.int32)
                segmentation_labels = segmentation.numpy().ravel()
                fpr, tpr, _ = roc_curve(mask_labels, segmentation_labels)
                roc_auc = auc(fpr,tpr)

                precision_values, recall_values, _ = precision_recall_curve(mask_labels, segmentation_labels)
                precision_recall_auc = auc(recall_values, precision_values)

                hausdorf_error = compute_hausdorff_dist(segmentation.numpy(), mask.numpy())

                with open(os.path.join(folder, 'metrics.csv'), 'a') as file:
                    file.write(f'{name},{ccr},{precision},{recall},{specifity},{f1_score},{jaccard_coef},{dice_coeff},{roc_auc},{precision_recall_auc}\n')

                ccrs.append(ccr)
                precisions.append(precision)
                recalls.append(recall)
                sensibilities.append(sensibility)
                specifities.append(specifity)
                f1_scores.append(f1_score)
                jaccard_coefs.append(jaccard_coef)
                dice_coeffs.append(dice_coeff)
                roc_auc_coeffs.append(roc_auc)
                precision_recall_auc_coeffs.append(precision_recall_auc)
                hausdorf_errors.append(hausdorf_error)

                if SAVE_SEGS and (epoch % N_EPOCHS_SAVE == 0 or epoch == -1):

                    image_save = trans(image.mul_(0.5).add_(0.5))
                    mask_save = trans(mask)
                    segmentation_save = trans(segmentation)
                    segmentation_save_vals = trans(segmentation_val)
    
                    opencv_image = np.array(image_save)
                    opencv_image = opencv_image[:, :, ::-1].copy()
                    opencv_gt = np.array(mask_save)
                    opencv_segmentation = np.array(segmentation_save)
                    opencv_segmentation_vals = np.array(segmentation_save_vals)

                    if not COLOR:
                        img = np.vstack(
                          (cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY), opencv_gt, opencv_segmentation))
                        cv2.imwrite(os.path.join(save_folder, f"{name}.png"), img)

                    else:
                        """
                        opencv_image = cv2.resize(opencv_image, (512, 512))

                        opencv_gt = cv2.resize(opencv_gt, (512, 512))
                        opencv_gt = (opencv_gt > 0.5).astype(np.float32)

                        opencv_segmentation = cv2.resize(opencv_segmentation, (512, 512))
                        opencv_segmentation = (opencv_segmentation > 0.5).astype(np.float32)
                        """
                        """
                        contours_gt, _ = cv2.findContours(opencv_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        contours_seg, _ = cv2.findContours(opencv_segmentation, cv2.RETR_TREE,
                                                                  cv2.CHAIN_APPROX_SIMPLE)

                        cv2.drawContours(opencv_image, contours_gt, -1, (0, 255, 0), 1)
                        cv2.drawContours(opencv_image, contours_seg, -1, (0, 0, 255), 1)

                        opencv_image = cv2.resize(opencv_image, (512, 512))

                        cv2.imwrite(os.path.join(save_folder, f"{name}"), opencv_image)
                        
                        print(opencv_image.shape)
                        print(opencv_gt.shape)
                        print(opencv_segmentation_vals.shape)
                        print()
                        print(np.max(opencv_image))
                        print(np.max(opencv_gt))
                        print(np.max(opencv_segmentation_vals))
                        print()
                        print()
                        """

                        #opencv_image = self.un_normalizer(opencv_image)
                        #opencv_image = (opencv_image * 0.225) + 0.485

                        save_image = generate_output_img(opencv_image, opencv_gt, opencv_segmentation_vals)
                        save_image = opencv_image

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        #cv2.putText(save_image, f'DSC: {dice_coeff:.3f}', (512+50, 480), font, 1, (0,0,255), 2)
                        save_image = cv2.resize(save_image, (512, 512))

                        cv2.imwrite(os.path.join(save_folder, f"{name}"), save_image)


        ccrs = np.array(ccrs)[~np.isnan(np.array(ccrs))]
        precisions = np.array(precisions)[~np.isnan(np.array(precisions))]
        recalls = np.array(recalls)[~np.isnan(np.array(recalls))]
        sensibilities = np.array(sensibilities)[~np.isnan(np.array(sensibilities))]
        specifities = np.array(specifities)[~np.isnan(np.array(specifities))]
        f1_scores = np.array(f1_scores)[~np.isnan(np.array(f1_scores))]
        jaccard_coefs = np.array(jaccard_coefs)[~np.isnan(np.array(jaccard_coefs))]
        dice_coeffs = np.array(dice_coeffs)[~np.isnan(np.array(dice_coeffs))]
        roc_auc_coeffs = np.array(roc_auc_coeffs)[~np.isnan(np.array(roc_auc_coeffs))]
        precision_recall_auc_coeffs = np.array(precision_recall_auc_coeffs)[~np.isnan(np.array(precision_recall_auc_coeffs))]
        hausdorf_errors = np.array(hausdorf_errors)[~np.isnan(np.array(hausdorf_errors))]

        mean_ccr = np.nansum(ccrs) / len(ccrs)
        std_ccr = np.std(np.array(ccrs))

        mean_precision = np.nansum(precisions) / len(precisions)
        std_precision = np.std(np.array(precisions))

        mean_recall = np.nansum(recalls) / len(recalls)
        std_recall = np.std(np.array(recalls))

        mean_sensibility = np.nansum(sensibilities) / len(sensibilities)
        std_sensibility = np.std(np.array(sensibilities))

        mean_specifity = np.nansum(specifities) / len(specifities)
        std_specifity = np.std(np.array(specifities))

        mean_f1_score = np.nansum(f1_scores) / len(f1_scores)
        std_f1_score = np.std(np.array(f1_scores))

        mean_jaccard_coef = np.nansum(jaccard_coefs) / len(jaccard_coefs)
        std_jaccard_coef = np.std(np.array(jaccard_coefs))

        mean_dice_coeff = np.nansum(dice_coeffs) / len(dice_coeffs)
        std_dice_coeff = np.std(np.array(dice_coeffs))

        mean_roc_auc = np.nansum(roc_auc_coeffs) / len(roc_auc_coeffs)
        std_roc_auc = np.std(np.array(roc_auc_coeffs))

        precision_recall_auc = np.nansum(precision_recall_auc_coeffs) / len(precision_recall_auc_coeffs)
        std_roc_auc = np.std(np.array(precision_recall_auc_coeffs))

        mean_hausdorf_error = np.nansum(hausdorf_errors) / len(hausdorf_errors)
        std_hausdorf_error = np.std(np.array(hausdorf_errors))

        segmentor.train()

        with open(os.path.join(folder, 'avg_metrics.txt'), 'w') as file:
            file.write(f"CCR: {mean_ccr:.3f} +- {std_ccr:.3f}\n")
            file.write(f"Precision: {mean_precision:.3f} +- {std_precision:.3f}\n")
            file.write(f"Recall: {mean_recall:.3f} +- {std_recall:.3f}\n")
            file.write(f"Specifity: {mean_specifity:.3f} +- {std_specifity:.3f}\n")
            file.write(f"F1 score: {mean_f1_score:.3f} +- {std_f1_score:.3f}\n")
            file.write(f"Jaccard: {mean_jaccard_coef:.3f} +- {std_jaccard_coef:.3f}\n")
            file.write(f"DSC: {mean_dice_coeff:.3f} +- {std_dice_coeff:.3f}\n")
            file.write(f"ROC AUC: {mean_roc_auc:.3f} +- {std_roc_auc:.3f}\n")
            file.write(f"PR AUC: {precision_recall_auc:.3f} +- {std_hausdorf_error:.3f}\n")

        if writer is not None:
            writer.add_scalar("Metrics/ccr", mean_ccr, epoch)
            writer.add_scalar("Metrics/precision", mean_precision, epoch)
            writer.add_scalar("Metrics/recall", mean_recall, epoch)
            writer.add_scalar("Metrics/sensibility", mean_sensibility, epoch)
            writer.add_scalar("Metrics/specifity", mean_specifity, epoch)
            writer.add_scalar("Metrics/f1 score", mean_f1_score, epoch)
            writer.add_scalar("Metrics/jaccard idx", mean_jaccard_coef, epoch)
            writer.add_scalar("Metrics/dice coeff", mean_dice_coeff, epoch)
            writer.add_scalar("Metrics/roc-auc", mean_roc_auc, epoch)
            writer.add_scalar("Metrics/precision recall auc", precision_recall_auc, epoch)
            writer.add_scalar("Metrics/hausdorf error", mean_hausdorf_error, epoch)

        return SegmentationEvaluationMetrics(mean_ccr, mean_precision, mean_recall,
         mean_sensibility, mean_specifity, mean_f1_score, mean_jaccard_coef,
         mean_dice_coeff, mean_roc_auc, precision_recall_auc, mean_hausdorf_error)
