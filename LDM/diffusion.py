import math
import torch
from torch import nn
import numpy as np
from torch.nn import init
from torch.nn import functional as F
# import config
from inspect import isfunction
from abc import abstractmethod
from numbers import Number
from tqdm import tqdm
import clip
from skimage.transform import radon, iradon



device = torch.device("cuda:6")

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

def timestep_embedding(gammas, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=gammas.device)
    args = gammas[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels):
    return GroupNorm32(16, channels)

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=3, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2] * 2, x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=3, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (2, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=True,
        dims=3,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = conv_nd(1, channels, channels, 1)

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class CrossAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads, num_head_channels, condition_dim):
        super(CrossAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.scale = (num_head_channels // num_heads) ** -0.5
        
        self.to_qk = nn.Linear(condition_dim, 2 * num_heads * num_head_channels, bias=False)
        self.to_v = nn.Linear(channels, num_heads * num_head_channels, bias=False)
        
        self.unifyheads = nn.Linear(num_heads * num_head_channels, channels)
        
    def forward(self, x, condition):
        b, t, _, h = *x.shape, self.num_heads
        
        qk = self.to_qk(condition).chunk(2, dim=-1)
        q, k = map(lambda t: t.reshape(b, t.shape[1], h, -1).transpose(1, 2), qk)
        v = self.to_v(x).reshape(b, t, h, -1).transpose(1, 2)
        
        q *= self.scale
        dots = torch.einsum('bhid,bhjd->bhij', q, k)
        attn = F.softmax(dots, dim=-1)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.unifyheads(out)

class UNet(nn.Module):
    def __init__(
        self,
        image_size = 128,
        in_channels = 1,
        model_channels = 16,
        out_channels = 1,
        num_res_blocks = 1,
        attention_resolutions = (8,),
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        num_classes=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.cross_attention = CrossAttentionBlock(channels=ch, num_heads=num_heads, num_head_channels=num_head_channels, condition_dim=time_embed_dim)


        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # if self.num_classes is not None:
        #     self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            # CrossAttention(ch),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            # normalization(ch),
            # nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, timesteps, condition):
        # assert (y is not None) == (self.num_classes is not None)

        x = self.cross_attention(x, condition)

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # if self.num_classes is not None:
        #     assert y.shape == (x.shape[0],)
        #     emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        for module in self.middle_block:
            # if isinstance(module, CrossAttention):
            #     h = module(h, y)
            if isinstance(module, AttentionBlock):
                h = module(h)
            else:
                h = module(h,emb)
        #     hs.append(h)
        # h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class Diffusion:
    def __init__(self, noise_steps=15, min_noise_level=0.04, etas_end=0.999, kappa= 0.1, power=0.3, device='cpu'):
        self.device = torch.device(device)
        self.sqrt_etas = self.get_named_eta_schedule(noise_steps,min_noise_level,etas_end,kappa,power)
        self.etas = self.sqrt_etas **2
        self.kappa = kappa
        self.noise_steps = noise_steps
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev
        self.base_scale = 0.01

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = kappa**2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(self.posterior_variance[1], self.posterior_variance[1:])

        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas

    def add_poisson_noise(self, images, scale=1.0, theta=None):
        """
        Add Poisson noise to images after applying Radon transform.
        :param images: Original images, a torch Tensor of shape (N, H, W).
        :param scale: Scaling factor for adjusting noise intensity.
        :param theta: The angles at which to compute the Radon transform.
        :return: Noisy images after inverse Radon transform.
        """
        # Apply Radon transform
        images_np = images.detach().cpu().numpy()
        sinograms = np.array([radon(image, theta=theta) for image in images_np])

        # Ensure non-negativity
        sinograms_positive = sinograms - np.min(sinograms)

        # Apply Poisson noise
        noisy_sinograms = np.random.poisson(sinograms_positive * scale) / scale
        noise_sinograms = noisy_sinograms - sinograms_positive + sinograms

        # Apply inverse Radon transform to the noise
        noisy_images_np = np.array([iradon(noise_sinogram, theta=theta, filter=None) for noise_sinogram in noise_sinograms])
        noisy_images = torch.from_numpy(noisy_images_np).float()

        return noisy_images
 

    def add_poisson_noise_in_projection(self, volumes, scale=1.0, theta=None):
        """
        Add Poisson noise to the projections of 3D volumes.
        :param volumes: Original 3D volumes, a torch Tensor of shape (N, C, D, H, W).
        :param scale: Scaling factor for adjusting noise intensity.
        :param theta: The angles at which to compute the Radon transform.
        :return: Noisy 3D volumes after inverse Radon transform.
        """
        noisy_volumes = []
        device = volumes.device
        volumes = volumes.cpu().detach().numpy()  # Convert to numpy for processing
        scale = scale.cpu().numpy()
        for volume in volumes:
            noisy_volume_slices = []
            for i in range(volume.shape[1]):  # Iterate over each depth slice
                slice_np = volume[0, i, :, :]  # Assuming single-channel data
                print(slice_np.shape)
                if theta is None:
                    theta = np.linspace(0., 180., max(slice_np.shape), endpoint=False)
                sinogram = radon(slice_np, theta=theta)
                # 修正负值或 NaN
                sinogram[sinogram < 0] = 0  # 将负值置为0
                sinogram = np.nan_to_num(sinogram)  # 将 NaN 替换为0
                sinogram_noisy = np.random.poisson(sinogram * scale) / scale
                reconstructed_slice = iradon(sinogram_noisy, theta=theta, filter_name='ramp', output_size=slice_np.shape[0])
                noisy_volume_slices.append(reconstructed_slice)            
            # Stack slices back into a volume
            noisy_volume = np.stack(noisy_volume_slices, axis=0)
            noisy_volumes.append(noisy_volume)
        
        noisy_volumes_np = np.array(noisy_volumes)
        noisy_volumes_tensor = torch.from_numpy(noisy_volumes_np).float().to(device).unsqueeze(1)  # Add channel dimension back

        return noisy_volumes_tensor


    def get_named_eta_schedule(self,noise_steps,min_noise_level,etas_end,kappa,power):
        etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        increaser = math.exp(1/((noise_steps-1))*math.log(etas_end/etas_start))
        base = np.ones([noise_steps, ]) * increaser
        power_timestep = np.linspace(0, 1, noise_steps, endpoint=True)**power
        power_timestep *= (noise_steps-1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
        return sqrt_etas
    
    def _scale_input(self, inputs, t):
        std = torch.sqrt(_extract_into_tensor(self.etas, t, inputs.shape) * self.kappa**2 + 1)
        inputs_norm = inputs / std
        return inputs_norm

    def noise_images(self, x_start, t, noise=None):
        # noise = torch.randn_like(x_start)
        noise_scale = (t + 1) * self.base_scale
        noise_image = self.add_poisson_noise_in_projection(x_start, noise_scale)
        return noise_image

    def sample_timesteps(self, n):
        return torch.randint(low=0, high=self.noise_steps, size=(n,))
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        return posterior_mean 
    
    def p_mean_variance(self, model, x_t, y, t, cond, clip_denoised=True):
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        #FDG_output = model(self._scale_input(x_t, t), t)
        FDG_output = model(x_t, t, cond)
        def process_xstart(x):
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        pred_xstart = process_xstart(FDG_output)
        model_mean = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)
        return model_mean, model_log_variance

    def prior_sample(self, y, noise=None):
        #Generate samples from the prior distribution, i.e., q(x_T|x_0) ~= N(x_T|y, ~)
        if noise is None:
            noise = torch.randn_like(y)
        t = torch.tensor([self.noise_steps-1,] * y.shape[0], device=y.device).long()
        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise
    
    def p_sample(self, model, x, y, t, cond, clip_denoised=True):
        mean, log_variance = self.p_mean_variance(model, x, y, t, cond, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))  # no noise when t == 0
        sample = mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise
        return sample

    def sample(self, model, y, cond, clip_denoised=True):
        # generating noise
        noise = torch.randn_like(y)
        y_sample = self.prior_sample(y, noise)
        indices = list(range(self.noise_steps))[::-1]
        for i in tqdm(indices):
            t = torch.tensor([i] * y.shape[0], device= device)
            with torch.no_grad():
                y_sample = self.p_sample(model, y_sample, y, t, cond, clip_denoised=clip_denoised)
        return y_sample

#CLIP
def get_optim_params(model_name: str):
    if model_name in ['ViT-B/32', 'ViT-B/16']:
        return ['visual.transformer.resblocks.11.attn.in_proj_weight',
                'visual.transformer.resblocks.11.attn.in_proj_bias',
                'visual.transformer.resblocks.11.attn.out_proj.weight',
                'visual.transformer.resblocks.11.attn.out_proj.bias',
                'visual.transformer.resblocks.11.ln_1.weight',
                'visual.transformer.resblocks.11.ln_1.bias',
                'visual.transformer.resblocks.11.mlp.c_fc.weight',
                'visual.transformer.resblocks.11.mlp.c_fc.bias',
                'visual.transformer.resblocks.11.mlp.c_proj.weight',
                'visual.transformer.resblocks.11.mlp.c_proj.bias',
                'visual.transformer.resblocks.11.ln_2.weight',
                'visual.transformer.resblocks.11.ln_2.bias',
                'visual.ln_post.weight',
                'visual.ln_post.bias',
                'visual.proj',
                'transformer.resblocks.11.attn.in_proj_weight',
                'transformer.resblocks.11.attn.in_proj_bias',
                'transformer.resblocks.11.attn.out_proj.weight',
                'transformer.resblocks.11.attn.out_proj.bias',
                'transformer.resblocks.11.ln_1.weight',
                'transformer.resblocks.11.ln_1.bias',
                'transformer.resblocks.11.mlp.c_fc.weight',
                'transformer.resblocks.11.mlp.c_fc.bias',
                'transformer.resblocks.11.mlp.c_proj.weight',
                'transformer.resblocks.11.mlp.c_proj.bias',
                'transformer.resblocks.11.ln_2.weight',
                'transformer.resblocks.11.ln_2.bias',
                'ln_final.weight',
                'ln_final.bias',
                'text_projection']
    elif model_name in ['ViT-L/14', 'ViT-L/14@336px']:
        return ['visual.transformer.resblocks.23.attn.in_proj_weight',
                'visual.transformer.resblocks.23.attn.in_proj_bias',
                'visual.transformer.resblocks.23.attn.out_proj.weight',
                'visual.transformer.resblocks.23.attn.out_proj.bias',
                'visual.transformer.resblocks.23.ln_1.weight',
                'visual.transformer.resblocks.23.ln_1.bias',
                'visual.transformer.resblocks.23.mlp.c_fc.weight',
                'visual.transformer.resblocks.23.mlp.c_fc.bias',
                'visual.transformer.resblocks.23.mlp.c_proj.weight',
                'visual.transformer.resblocks.23.mlp.c_proj.bias',
                'visual.transformer.resblocks.23.ln_2.weight',
                'visual.transformer.resblocks.23.ln_2.bias',
                'visual.ln_post.weight',
                'visual.ln_post.bias',
                'visual.proj',
                'transformer.resblocks.11.attn.in_proj_weight',
                'transformer.resblocks.11.attn.in_proj_bias',
                'transformer.resblocks.11.attn.out_proj.weight',
                'transformer.resblocks.11.attn.out_proj.bias',
                'transformer.resblocks.11.ln_1.weight',
                'transformer.resblocks.11.ln_1.bias',
                'transformer.resblocks.11.mlp.c_fc.weight',
                'transformer.resblocks.11.mlp.c_fc.bias',
                'transformer.resblocks.11.mlp.c_proj.weight',
                'transformer.resblocks.11.mlp.c_proj.bias',
                'transformer.resblocks.11.ln_2.weight',
                'transformer.resblocks.11.ln_2.bias',
                'ln_final.weight',
                'ln_final.bias',
                'text_projection']
    else:
        print(f"no {model_name}")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model, self.preprocess = clip.load('ViT-B/32', 'cpu')

        self.optim_params = get_optim_params('ViT-B/32')

        for name, param in self.model.named_parameters():
            if name not in self.optim_params:
                param.requires_grad = False

    def forward(self, text):
        text_features = self.model.encode_text(text)
        return text_features

if __name__ == "__main__":
    #x = torch.randn((1, 1, 40, 48, 40))
    diffusion =  Diffusion()
    FDG_path = "D:/Caiwen/data/train/target/686130_1.nii"
    FDG = nifti_to_numpy(FDG_path)
    FDG = np.expand_dims(FDG, axis=0)
    FDG = np.expand_dims(FDG, axis=1)
    FDG = torch.tensor(FDG)
    FDG = FDG.to(config.device)

    t1_path = "D:/Caiwen/data/train/input/686130_0.nii" 
    t1 = nifti_to_numpy(t1_path)
    t1 = np.expand_dims(t1, axis=0)
    t1 = np.expand_dims(t1, axis=1)
    t1 = torch.tensor(t1)
    t1 = t1.to(config.device)

    image = nib.load(config.path)

    indices = list(range(diffusion.noise_steps))[::-1]
    for i in tqdm(indices):
        t = torch.tensor([i] * FDG.shape[0], device=config.device)
        noise = torch.randn_like(FDG)
        FDG_t = diffusion.noise_images(FDG, t1, t, noise)
        FDG_out = FDG_t.detach().cpu().numpy()
        FDG_out = np.squeeze(FDG_out)
        FDG_out = FDG_out.astype(np.float32)
        FDG_out=nib.Nifti1Image(FDG_out,image.affine)
        nib.save(FDG_out,"./view/"+str(i)+"_686130")
