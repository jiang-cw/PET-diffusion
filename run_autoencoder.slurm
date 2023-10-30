#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --partition=bme_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -t 5-00:00:00


#SBATCH --output=result/out/%j.out
#SBATCH --error=result/out/%j.err



PL_TORCH_DISTRIBUTED_BACKEND=gloo python train_autoencoder.py dataset=default dataset.root_dir='data/PET/train/std_dose/' model=autoencoder model.gpus=1 model.default_root_dir_postfix='own_dataset' model.precision=16 model.embedding_dim=8 model.n_hiddens=16 model.downsample=[2,2,2] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 model.discriminator_iter_start=10000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 model.gan_feat_weight=4 model.batch_size=1 model.n_codes=16384 model.accumulate_grad_batches=1 