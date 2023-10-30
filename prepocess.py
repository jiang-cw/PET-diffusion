import numpy as np
import torch
from torch.autograd import Variable
import SimpleITK as sitk
from tqdm import tqdm
import os
import random
import torch.nn as nn
# from skimage.measure import compare_psnr, compare_ssim
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = "cuda"


def save_output(outputs, id, epoch, path):
    def save_img(img, name):
        # img = SimpleITK.GetImageFromArray(img[0,0].cpu().detach().numpy())
        img = sitk.GetImageFromArray(img)

        template = sitk.ReadImage('./data/pet_chest/train_2/std_dose/1.nii.gz')
        img.SetSpacing(template.GetSpacing())
        img.SetOrigin(template.GetOrigin())
        img.SetDirection(template.GetDirection())

        sitk.WriteImage(img, path +name+'.nii.gz')

def read_image(path):
    image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image)
    image = image.astype(np.float32)

    image = np.expand_dims(np.expand_dims(image,axis=0),axis=0).astype(np.float32)
    image = torch.from_numpy(image).to(device)
    image = F.interpolate(image.float(), size=[160,256,256], mode='trilinear')
    image = image[0,0].cpu().detach().numpy()

    image_copy = image.copy()
    array_max, array_min = np.percentile(image, [99.9, 0.1])
    image = (image - array_min) / (array_max - array_min)
    return image, image_copy, array_max, array_min





template = sitk.ReadImage('data/PET/train/std_dose/1.nii.gz')
root = os.path.join(os.getcwd(),'data/PET/train/std_dose/')
paths = os.listdir(root)
for path in paths:
    print(path)
    image = sitk.ReadImage('data/PET/train/std_dose/'+ path )
    image = sitk.GetArrayFromImage(image)
    image = image.astype(np.float32)

    image = np.expand_dims(np.expand_dims(image,axis=0),axis=0).astype(np.float32)
    image = torch.from_numpy(image).to(device)
    image = F.interpolate(image.float(), size=[160,256,256], mode='trilinear')
    image = image[0,0].cpu().detach().numpy()


    image = sitk.GetImageFromArray(image)
            
    image.SetSpacing(template.GetSpacing())
    image.SetOrigin(template.GetOrigin())
    image.SetDirection(template.GetDirection())

    sitk.WriteImage(image,'data/PET/train/std_dose_1/'+ path)  