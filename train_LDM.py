import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
from torch.utils.data import DataLoader
from utils.utils import *
from LDM.diffusion import *
from init import Options
from utils import *
import pandas as pd
import numpy as np
from thop import profile
from torchsummary import summary
from skimage.metrics import mean_squared_error, normalized_root_mse, peak_signal_noise_ratio, structural_similarity
from autoencoder.autoencoder import *
from autoencoder.discriminators import *
from init import Options

print(torch.cuda.is_available())
print(torch.__version__)  #

opt = Options().parse()
min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))

if opt.gpu_ids != '-1':
    num_gpus = len(opt.gpu_ids.split(','))
else:
    num_gpus = 0
print('number of GPU:', num_gpus)

# -----  Loading the list of data -----
train_list = create_list_CT(opt.data_path)
val_list = create_list_CT(opt.val_path)
# SECT_list = create_SECT_list(opt.val_path)

for i in range(opt.increase_factor_data):  # augment the data list for training

    train_list.extend(train_list)
    val_list.extend(val_list)

print('Number of training patches per epoch:', len(train_list))
print('Number of validation patches per epoch:', len(val_list))

# -----  Transformation and Augmentation process for the data  -----
trainTransforms = [
            # NiftiDataset.Resample(opt.new_resolution, opt.resample),
            # NiftiDataset.Augmentation(),
            # NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
            NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
            ]

train_set = NifitDataSet_CT(train_list, direction=opt.direction, transforms=trainTransforms, train=True)    # define the dataset and loader
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)  # Here are then fed to the network with a defined batch size
# -------------------------------------

autoencoder = build_netG(opt)
autoencoder_CT = build_netG(opt)

diffusion = Diffusion(device='cuda')
Unet = UNet()

criterionMSE = nn.MSELoss()  # nn.MSELoss()

# -----  Use Single GPU or Multiple GPUs -----
if (opt.gpu_ids != -1) & torch.cuda.is_available():
    use_gpu = True
    Unet.cuda()
    autoencoder.cuda()
    autoencoder_CT.cuda()
    criterionMSE.cuda()

    if num_gpus > 1:
        Unet = nn.DataParallel(Unet)
        autoencoder = nn.DataParallel(autoencoder)
        autoencoder_CT = nn.DataParallel(autoencoder_CT)

optim_Unet = optim.Adam(Unet.parameters(), betas=(0.5,0.999), lr=opt.learning_rate)
Unet_scheduler = get_scheduler(optim_Unet, opt)

autoencoder.load_state_dict(torch.load('result/autoencoder/chk_8/g_epoch_8.pth'))
autoencoder.eval()  

autoencoder_CT.load_state_dict(torch.load('result/autoencoder_CT/chk_100/g_epoch_100.pth'))
autoencoder_CT.eval() 



# -----  Training Cycle -----
print('Start training :) ')


log_name = opt.task
print("log_name: ", log_name)
f = open(os.path.join('result/' + opt.task + '/', log_name + ".txt"), "w")

for epoch in range(opt.epoch):

    for batch_idx, (data, label, CT, filename) in enumerate(train_loader):
  
        label = label.cuda()
        CT = CT.cuda()
        # data = data.cuda()
        target = autoencoder.module.encoder(label)
        CT_encoded = autoencoder_CT.module.encoder(CT)

        t = diffusion.sample_timesteps(target.shape[0]).cuda()
        target_t = diffusion.noise_images(target, t)
        target_out = Unet(target_t, t, CT_encoded)
        target_image = autoencoder.module.decoder(target_out)


        optim_Unet.zero_grad()
        MSE_loss = criterionMSE(target_out, target)
        MSE_loss_image = criterionMSE(target_image, label)
        MSE = MSE_loss + MSE_loss_image
        MSE.backward()

        optim_Unet.step()


        update_learning_rate(Unet_scheduler, optim_Unet)
        
 


    if epoch % opt.save_fre == 0:
        save_path = os.path.join('result/' + opt.task + '/', 'chk_'+str(epoch))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        ##### Logger ######

        valTransforms = [
            # NiftiDataset.Resample(opt.new_resolution, opt.resample),
            # NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
            # NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
        ]

        val_set = NifitDataSet_CT(val_list, direction=opt.direction, transforms=valTransforms, test=True)
        val_loader = DataLoader(val_set, batch_size= 1, shuffle=False, num_workers=opt.workers)



        # test 
        name = list()
        count = 0
        for batch in val_loader:
            data, target, CT, filename = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3]


        
            data = data.cuda()

            CT = CT.cuda()
            target = autoencoder.module.encoder(data)
            CT_encoded = autoencoder_CT.module.encoder(CT)

            t = diffusion.sample_timesteps(target.shape[0]).cuda()
            target_t = diffusion.noise_images(target, t)
            prediction = Unet(target_t, t, CT_encoded)
            prediction = autoencoder.module.decoder(prediction)
         
            
            
            data = data[0,0].cpu().detach().numpy()
      
            prediction = prediction[0,0].cpu().detach().numpy()
     

            target = target[0,0].cpu().detach().numpy()
            data = (data * 127.5) + 127.5
            prediction = (prediction * 127.5) + 127.5
         
            target = (target * 127.5) + 127.5

            if  epoch / opt.save_fre == 1:
                save_result(data, prediction, target, index = filename[0][0:-7], path = save_path)
            else:
                save_result(prediction = prediction, index = filename[0][0:-7], path = save_path)
            count = count + 1


        torch.save(Unet.state_dict(), '%s/g_epoch_{}.pth'.format(epoch) % save_path)

f.close()