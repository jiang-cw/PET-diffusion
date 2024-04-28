#!/bin/bash
#SBATCH --job-name=autoencoder
#SBATCH --partition=bme_gpu4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=40G
#SBATCH -N 1
#SBATCH -t 5-00:00:00
#SBATCH --output=result/out/%j.out
#SBATCH --error=result/out/%j.err

python train_autoencoder.py