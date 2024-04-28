import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler




def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),)



class Encoder3D(nn.Module):
    def __init__(self, input_channels, num_filters =64):
        super(Encoder3D, self).__init__()

        self.in_dim = input_channels
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()

    def forward(self, x):
        down_1 = self.down_1(x) # -> [1, 4, 128, 128, 128]
        pool_1 = self.pool_1(down_1) # -> [1, 4, 64, 64, 64]
        down_2 = self.down_2(pool_1) # -> [1, 8, 64, 64, 64]
        pool_2 = self.pool_2(down_2) # -> [1, 8, 32, 32, 32]
        down_3 = self.down_3(pool_2) # -> [1, 16, 32, 32, 32]
        pool_3 = self.pool_3(down_3) # -> [1, 16, 16, 16, 16]
        
        down_4 = self.down_4(pool_3) # -> [1, 32, 16, 16, 16]
        pool_4 = self.pool_4(down_4) # -> [1, 32, 8, 8, 8]

        return pool_4


class Decoder3D(nn.Module):
    def __init__(self, output_channels, num_filters=64):
        super(Decoder3D, self).__init__()

        self.out_dim = output_channels
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 4, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 2, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 1, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 1, self.num_filters * 1, activation)
        # Output
        self.out = conv_block_3d(self.num_filters, self.out_dim, activation)

    def forward(self, x):
        # Up sampling
        trans_1 = self.trans_1(x) # -> [1, 128, 8, 8, 8]
        trans_2 = self.trans_2(trans_1) # -> [1, 64, 16, 16, 16]
        trans_3 = self.trans_3(trans_2) # -> [1, 32, 32, 32, 32]
        trans_4 = self.trans_4(trans_3) # -> [1, 16, 64, 64, 64]
  
        # Output
        out = self.out(trans_4) # -> [1, 3, 128, 128, 128]
        return out

class Autoencoder3D(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Autoencoder3D, self).__init__()
        self.encoder = Encoder3D(input_channels)
        self.decoder = Decoder3D(output_channels)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class PerceptualLoss3D(nn.Module):
    def __init__(self):
        super(PerceptualLoss3D, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = self.feature_extractor(x)
        y_features = self.feature_extractor(y)
        loss = torch.mean((x_features - y_features) ** 2)
        return loss

class PatchDiscriminator3D(nn.Module):
    def __init__(self):
        super(PatchDiscriminator3D, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class PatchGANLoss3D(nn.Module):
    def __init__(self, discriminator):
        super(PatchGANLoss3D, self).__init__()
        self.discriminator = discriminator

    def forward(self, real, fake):
        real_loss = torch.mean((self.discriminator(real) - 1) ** 2)
        fake_loss = torch.mean((self.discriminator(fake)) ** 2)
        return real_loss + fake_loss



def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>



def build_netG(opt):
   
    generator = Autoencoder3D(opt.img_channel, opt.img_channel)

    init_weights(generator, init_type='normal')

    return generator