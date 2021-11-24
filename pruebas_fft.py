import torch
import torch.nn as nn
import numpy as np
import math
from PIL import Image
import PIL
from torchvision import transforms


def _spectral_crop(input, oheight, owidth):

    cutoff_freq_h = math.ceil(oheight / 2)
    cutoff_freq_w = math.ceil(owidth / 2)

    if oheight % 2 == 1:
        if owidth % 2 == 1:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):]
            bottom_left = input[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w]
            bottom_right = input[:, :, -(cutoff_freq_h-1):, -(cutoff_freq_w-1):]
        else:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            bottom_left = input[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w]
            bottom_right = input[:, :, -(cutoff_freq_h-1):, -cutoff_freq_w:]
    else:
        if owidth % 2 == 1:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):]
            bottom_left = input[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            bottom_right = input[:, :, -cutoff_freq_h:, -(cutoff_freq_w-1):]
        else:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            bottom_left = input[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            bottom_right = input[:, :, -cutoff_freq_h:, -cutoff_freq_w:]

    top_combined = torch.cat((top_left, top_right), dim=-1)
    bottom_combined = torch.cat((bottom_left, bottom_right), dim=-1)
    all_together = torch.cat((top_combined, bottom_combined), dim=-2)

    return all_together

def _spectral_pad(input, output, oheight, owidth):
    cutoff_freq_h = math.ceil(oheight / 2)
    cutoff_freq_w = math.ceil(owidth / 2)

    pad = torch.zeros_like(input)

    if oheight % 2 == 1:
        if owidth % 2 == 1:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):] = output[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):]
            pad[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w] = output[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w]
            pad[:, :, -(cutoff_freq_h-1):, -(cutoff_freq_w-1):] = output[:, :, -(cutoff_freq_h-1):, -(cutoff_freq_w-1):]
        else:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -cutoff_freq_w:] = output[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            pad[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w] = output[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w]
            pad[:, :, -(cutoff_freq_h-1):, -cutoff_freq_w:] = output[:, :, -(cutoff_freq_h-1):, -cutoff_freq_w:]
    else:
        if owidth % 2 == 1:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):] = output[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):]
            pad[:, :, -cutoff_freq_h:, :cutoff_freq_w] = output[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            pad[:, :, -cutoff_freq_h:, -(cutoff_freq_w-1):] = output[:, :, -cutoff_freq_h:, -(cutoff_freq_w-1):]
        else:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -cutoff_freq_w:] = output[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            pad[:, :, -cutoff_freq_h:, :cutoff_freq_w] = output[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            pad[:, :, -cutoff_freq_h:, -cutoff_freq_w:] = output[:, :, -cutoff_freq_h:, -cutoff_freq_w:]	

    return pad	


def DiscreteHartleyTransform(input):
    fft = torch.fft.fft(input, dim=2, norm="ortho")
    fft = torch.view_as_real(fft)
    dht = fft[:, :, :, :, -2] - fft[:, :, :, :, -1]
    return dht


def main():
    
    path = '/home/imartinez/Dataset_TFM/images/BUSI/benign (1).png'
    
    img = Image.open(path).convert("L")
    IMG_SIZE = 512
    POOLSIZE = IMG_SIZE // 2
    NCHANNELS = 3
    
    resize_t = transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=PIL.Image.NEAREST)
    to_torch_tensor_t = transforms.ToTensor()
    input = torch.rand([1, NCHANNELS, IMG_SIZE, IMG_SIZE])
    input[0, :, :, :] = to_torch_tensor_t(resize_t(img)).sub_(0.5).div_(0.5)
    

    fft = torch.(input)
    #croped_fft = fft[:, :, IMG_SIZE//4:3*IMG_SIZE//4, IMG_SIZE//4:3*IMG_SIZE//4]
    croped_fft = _spectral_crop(fft, POOLSIZE, POOLSIZE)
    pooled = DiscreteHartleyTransform(croped_fft)
    
    
    fft = torch.abs(torch.fft.fft(input, dim=2, norm="ortho"))
    
    spectrum = transforms.ToPILImage()(fft[0, :, ...])
    img_pooled = transforms.ToPILImage()(pooled[0, :, ...].mul_(0.5).add_(0.5))
    
    print(pooled.shape)
    
    resize_t(img).save('original.png')
    img_pooled.save('pooled.png')
    spectrum.save('spectrum.png')

    
    

if __name__ == '__main__':
    main()