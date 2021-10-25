import cv2
import os
import numpy as np

#def fuzzu

def generate_output_img(image, gt, segmentation):

    GT_COLOUR = (0., 1., 0.)
    SEG_COLOUR = (0., 0., 1.)
    GT_SEG_COLOUR = (0., 1., 1.)
    ALPHA = 0.57
    ALPHA2 = 0.57

    binary_segmentation = np.zeros_like(segmentation)
    binary_segmentation[segmentation >= 0.5] = 1
    
    gt_seg_intersect_mask = gt * binary_segmentation

    paint_mask = np.zeros_like(image)
    paint_mask[gt_seg_intersect_mask == 1., :] = GT_SEG_COLOUR
    paint_mask[(gt == 1.) & (gt_seg_intersect_mask == 0.)] = GT_COLOUR
    paint_mask[(binary_segmentation == 1.) & (gt_seg_intersect_mask == 0.)] = SEG_COLOUR
    paint_mask[(paint_mask[:, :, 2] == 0.) & (binary_segmentation == 1.), :] = SEG_COLOUR

    image_painted_with_segs = np.copy(image)
    cond = (binary_segmentation == 1.) | (gt == 1.)
    image_painted_with_segs[cond, :] = ALPHA * image_painted_with_segs[cond, :] + (1-ALPHA) * paint_mask[cond]

    ########################################################################################

    segmentation = (segmentation * 255).astype(np.uint8)
    heatmap_seg = cv2.applyColorMap(segmentation, cv2.COLORMAP_JET)

    heatmap_image = ALPHA2 * image + (1-ALPHA2) * heatmap_seg

    concated_images = np.hstack([image_painted_with_segs, heatmap_image])
    
    return concated_images


if __name__ == '__main__':

    image = cv2.imread('/home/inaki/shared_files/Dataset_TFM/images/BUSI/benign (18).png')
    mask = cv2.imread('/home/inaki/shared_files/Dataset_TFM/gt/BUSI/benign (18).png')
    segmentation = cv2.imread('/home/inaki/shared_files/Dataset_TFM/gt/BUSI/benign (225).png')

    cv2.imshow('s', image)
    cv2.waitKey()

    SIZE = 500

    image = cv2.resize(image, (SIZE, SIZE)).astype(np.float32)
    mask = cv2.resize(mask, (SIZE, SIZE))
    segmentation = cv2.resize(segmentation, (SIZE, SIZE))

    cv2.imshow('s', image)
    cv2.waitKey()

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.float32)
    segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY).astype(np.float32)

    cv2.imshow('s', image)
    cv2.waitKey()

    print(type(image))

    image = image / 255.
    mask = mask / 255.
    segmentation = segmentation / 255.

    cv2.imshow('s', image)
    cv2.waitKey()

    print(type(image))

    print(image.shape)
    print(mask.shape)
    print(segmentation.shape)

    output_img = generate_output_img(image, mask, segmentation)

    cv2.imshow('s', output_img)
    cv2.waitKey()