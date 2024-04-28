import torch
import torch.nn as nn
import numpy as np
import functools
import random
from torch.nn import init



lrelu = nn.LeakyReLU(0.2)


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

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def conv_block(ndf,stage):
    module = []
    module += [
        nn.Conv3d(ndf*(2**stage), ndf*(2**stage), kernel_size=4, stride=2, padding=1),
        nn.BatchNorm3d(ndf*(2**stage)),
        nn.Dropout3d(p=0.5),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv3d(ndf*(2**stage), ndf*(2**(stage+1)), kernel_size=4, stride=1, padding=1),
        nn.BatchNorm3d(ndf*(2**(stage+1))),
        nn.Dropout3d(p=0.5),
        nn.LeakyReLU(negative_slope=0.2)
    ]
    return module


class PixelDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PixelDiscriminator, self).__init__()
        self.input_nc = 2
        self.ndf = opt.ndf  # number of filters
        self.norm_layer = nn.BatchNorm3d

        if type(self.norm_layer) == functools.partial:
            use_bias = self.norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = self.norm_layer == nn.InstanceNorm3d

        self.net = [
            nn.Conv3d(self.input_nc,  self.ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(self.ndf,  self.ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            self.norm_layer( self.ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d( self.ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, opt):
        super(NLayerDiscriminator, self).__init__()
        self.input_nc = 2
        self.ndf = opt.ndf  # number of filters
        self.n_layers = 3
        self.norm_layer = nn.BatchNorm3d

        if type(self.norm_layer) == functools.partial:  # no need to use bias as BatchNorm3d has affine parameters
            use_bias = self.norm_layer.func == nn.BatchNorm3d
        else:
            use_bias = self.norm_layer == nn.BatchNorm3d

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(self.input_nc, self.ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1

        for n in range(1, self.n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(self.ndf * nf_mult_prev, self.ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                self.norm_layer(self.ndf * nf_mult),
                nn.Dropout(0.2),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** self.n_layers, 8)

        sequence += [
            nn.Conv3d(self.ndf * nf_mult_prev, self.ndf * nf_mult, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            self.norm_layer(self.ndf * nf_mult),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(self.ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=1)]  # output 1 channel prediction map

        self.model = nn.Sequential(*sequence)

    def forward(self, img_input):
        """Standard forward."""
        result = self.model(img_input)
        return result


def build_netD(opt):

    if opt.netD == 'PatchGAN':
        discriminator = NLayerDiscriminator(opt)
    elif opt.netD == 'PixelGAN':
        discriminator = PixelDiscriminator(opt)
    else:
        raise NotImplementedError

    init_weights(discriminator, init_type='normal')

    return discriminator


def smooth_positive_labels(y):
    output = y - 0.3 + (np.random.random(y) * 0.5)

    if output >= 1:

        return 1
    else:
        return output

def smooth_negative_labels(y):
    return 0 + np.random.random(y) * 0.3


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=float(smooth_positive_labels(1)), target_fake_label=float(smooth_negative_labels(1))):
        super(GANLoss, self).__init__()
        if random.randint(0, 100) >= 7:  # noisy labels
            self.register_buffer('real_label', torch.tensor(target_real_label))
            self.register_buffer('fake_label', torch.tensor(target_fake_label))
        else:
            self.register_buffer('real_label', torch.tensor(target_fake_label))
            self.register_buffer('fake_label', torch.tensor(target_real_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        # print(target_tensor),
        # print(target_tensor.shape)
        return self.loss(input, target_tensor)



if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    from torchsummaryX import summary

    from init import Options
    opt = Options().parse()

    torch.cuda.set_device(0)
    discriminator = build_netD(opt)
    net = discriminator.cuda().eval()

    data= Variable(torch.randn(opt.batch_size, opt.img_channel, opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])).cuda()

    data = torch.cat((data, data), 1)
    out = net(data)

    summary(net, data)
    print("out size: {}".format(out.size()))

