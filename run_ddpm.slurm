#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --partition=bme_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -t 5-00:00:00


#SBATCH --output=result/out/%j.out
#SBATCH --error=result/out/%j.err



python train_ddpm.py model=ddpm dataset=default model.results_folder_postfix='own_dataset' model.autoencoder_ckpt='result/checkpoints/autoencoder/DEFAULT/own_dataset/lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt' model.diffusion_img_size=32 model.diffusion_depth_size=32 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=10 model.gpus=1