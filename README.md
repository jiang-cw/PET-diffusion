
## PET-diffusion: Unsupervised PET Enhancement based on the Latent Diffusion Model (see https://link.springer.com/chapter/10.1007/978-3-031-43907-0_1)
This repo contains the supported pytorch code and configuration files of our work


![Overview of the framework](img/fig1_V3.png)

# System Requirements
This code has been tested on Ubuntu 20.04 and an NVIDIA Tesla A100 GPU. Furthermore it was developed using Python v3.8. 


# Model Training
The entire model training is divided into two stages. In the first stage, the `train_autoencoder.py` script is used to train an autoencoder for compressing PET images. In the second stage, the `train_LDM.py` script is used to train the LDM model. At this stage, the autoencoder trained in the first stage is called to compress PET images. 

It's important to note that if CT image-guided denoising is required in the second stage, a separate autoencoder (`autoencoder_CT`) needs to be pre-trained specifically for compressing CT images.



## Citation
```
@inproceedings{jiang2023pet,
  title={PET-Diffusion: Unsupervised PET Enhancement Based on the Latent Diffusion Model},
  author={Jiang, Caiwen and Pan, Yongsheng and Liu, Mianxin and Ma, Lei and Zhang, Xiao and Liu, Jiameng and Xiong, Xiaosong and Shen, Dinggang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={3--12},
  year={2023},
  organization={Springer}
}
```

