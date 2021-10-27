import torch
import torch.nn as nn
from models.image_generators.image_generators import ImageGenerator1
import numpy as np
import cv2
import os
import PIL
from torchvision import transforms


if __name__ == '__main__':

    N_GENERATIONS = 25
    SAVE_FOLDER = 'synt_images'
    DEVICE = 'cuda:0'

    model = ImageGenerator1().to(DEVICE)
    model.load_state_dict(torch.load('experiments/exp1/weights/generator_last.pt', map_location=DEVICE))
    
    noise = torch.randn(N_GENERATIONS, 100, 1, 1).to(DEVICE)
    
    output = model(noise)
    
    n_images = output.shape[0]
    
    trans = transforms.ToPILImage()
    
    for i in range(n_images):
    
        mask = output[i, 3, :, :].detach().to('cpu')
        image = output[i, :3, :, :].detach().to('cpu')
        
        image = trans(image)
        mask = trans(mask)
        
        np_image = np.array(image)
        np_mask = np.array(mask)
        
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        
        save_img = np.hstack([np_image, np_mask])
        
        if not os.path.isdir(SAVE_FOLDER):
            os.mkdir(SAVE_FOLDER)
            
        cv2.imwrite(os.path.join(SAVE_FOLDER, f"sample_{i}.png"), save_img)