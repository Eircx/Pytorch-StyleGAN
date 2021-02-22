import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os

class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super(ApplyNoise, self).__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise):
        if noise is not None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)


class ApplyStyle(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(ApplyStyle, self).__init__()
        self.linear = FC(latent_size, channels * 2, gain=1.0, use_wscale=use_wscale)
    
    def forward(self, x, latent):
        style = self.linear(latent)
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)
        x = x * (style[:, 0] + 1.0) + style[:, 1]
        return x


class Blur2d(nn.Module):
    def __init__(self, f=[1,2,1], normalize=True, stride=1, flip=False):
        super(Blur2d, self).__init__()
        if f is not None:
            f = torch.Tensor(f)
            f = f[:,None] * f[None,:]
            f = f[None, None]
            if normalize:
                f = f / f.sum()
            if flip:
                f = torch.flip(f, [2,3])
            self.f = f
        else:
            self.f = None
        self.stride = stride

    def forward(self, x):
        if self.f is not None:
            kernel = self.f.expand(x.size(1), -1, -1, -1).to(x.device)
            x = F.conv2d(x, kernel, stride=self.stride, 
                    padding=int(self.f.size(2)//2), groups=x.size(1))
        return x    
            

class FC(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            gain=2**0.5,
            lrmul=1.0,
            use_wscale=False,
            bias=True):
        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None
            self.b_lrmul = 0
        
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels) * init_std)

    def forward(self, x):
        # weight = self.weight.to(x.device)
        # bias = self.bias.to(x.device)
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out
            

class Conv2d(nn.Module):
    def __init__(self,
        input_channels,
        output_channels,
        kernel_size,
        gain=2**0.5,
        use_wscale=False,
        lrmul=1.0,
        bias=True):
        super(Conv2d, self).__init__()
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)
        self.kernel_size = kernel_size
        
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std
        )

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None
            self.b_lrmul = None

    def forward(self, x):
        if self.bias is not None:
            return F.conv2d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, padding=self.kernel_size // 2)
        else:
            return F.conv2d(x, self.weight * self.w_lrmul, padding=self.kernel_size // 2)


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)
        return x * tmp


class Upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super(Upscale2d, self).__init__()
        self.gain = gain
        self.factor = factor
    
    def forward(self, x):
        if self.gain != 1:
            x *= self.gain
        if self.factor > 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, self.factor, -1, self.factor)
            x = x.contiguous().view(shape[0], shape[1], self.factor * shape[2], self.factor * shape[3])
        return x


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, dim=(2, 3), keepdim=True)
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, dim=(2, 3), keepdim=True) + self.epsilon)
        return x * tmp


class LayerEpilogue(nn.Module):
    def __init__(self,
                 channels, 
                 dlatent_size,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 use_style):
        super(LayerEpilogue, self).__init__()

        if use_noise:
            self.noise = ApplyNoise(channels)
        else:
            self.noise = None

        if use_pixel_norm:
            self.pixel_norm = PixelNorm()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = InstanceNorm()
        else:
            self.instance_norm = None
        
        if use_style:
            self.style = ApplyStyle(dlatent_size, channels, use_wscale)
        else:
            self.style = None

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, noise, dlatents_in_slice=None):
        if self.noise is not None:
            x = self.noise(x, noise)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        if self.style is not None:
            x = self.style(x, dlatents_in_slice)

        return x


class GBlock(nn.Module):
    def __init__(self,
                 res,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 noise_input,
                 dlatent_size=128,
                 use_style=True,
                 f=[1,2,1],
                 factor=2,
                 fmap_base=2048,
                 fmap_decay=1.0,
                 fmap_max=128,
                 ):
        super(GBlock, self).__init__()
        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        self.res = res
        self.blur = Blur2d(f)
        self.noise_input = noise_input
        if res < 6:
            self.up_sample = Upscale2d(factor)
        else:
            self.up_sample = nn.ConvTranspose2d(self.nf(res-3), self.nf(res-2), 4, stride=2, padding=1)

        self.adaIn1 = LayerEpilogue(self.nf(res-2), dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style) 
        self.conv1 = Conv2d(self.nf(res-2), self.nf(res-2), kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf(res-2), dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style) 

    def forward(self, x, dlatent):
        x = self.up_sample(x)
        x = self.adaIn1(x, self.noise_input[self.res*2-4], dlatent[:, self.res*2-4])
        x = self.conv1(x)
        x = self.adaIn2(x, self.noise_input[self.res*2-3], dlatent[:, self.res*2-3])
        return x


class G_mapping(nn.Module):
    def __init__(self, 
        mapping_fmaps=128,
        dlatent_size=128,
        resolution=256,
        normalize_latents=True,
        use_wscale=True,
        lrmul=0.01,
        gain=2**0.5
        ):
        super(G_mapping, self).__init__()
        self.mapping_fmaps=mapping_fmaps
        self.layers = nn.Sequential(
            FC(self.mapping_fmaps, dlatent_size, gain, lrmul, use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul, use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul, use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul, use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul, use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul, use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul, use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul, use_wscale)
        )
        self.normalize_latents = normalize_latents
        self.log2_res = int(np.log2(resolution))
        self.num_layers = self.log2_res * 2 - 2
        self.pixel_norm = PixelNorm()

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.layers(x)
        return out, self.num_layers


class G_synthesis(nn.Module):
    def __init__(self,
                 dlatent_size,
                 resolution=256,
                 fmap_base=2048,
                 num_channels=3,
                 fmap_max=128,
                 fmap_decay=1.0,
                 f=[1,2,1],
                 use_pixel_norm = False,
                 use_instance_norm = True,
                 use_wscale = True,
                 use_noise = True,
                 use_style=True
                 ):
        super(G_synthesis, self).__init__()
        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        self.log2_res = int(np.log2(resolution))
        num_layers = self.log2_res * 2 - 2
        self.num_layers = num_layers

        self.noise_inputs = []
        for layer_idx in range(num_layers):
            res = layer_idx // 2 + 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noise_inputs.append(torch.randn(*shape).to('cuda'))
        
        self.blur = Blur2d(f)

        self.channel_shrinkage = Conv2d(input_channels=self.nf(self.log2_res-2),
                                        output_channels=self.nf(self.log2_res),
                                        kernel_size=3,
                                        use_wscale=use_wscale)
        self.torgb = Conv2d(self.nf(self.log2_res), num_channels, kernel_size=1, gain=1, use_wscale=use_wscale)

        self.const_input = nn.Parameter(torch.ones(1, self.nf(1), 4, 4))
        self.bias = nn.Parameter(torch.ones(self.nf(1)))
        self.adaIn1 = LayerEpilogue(self.nf(1), dlatent_size, use_wscale, use_noise, use_pixel_norm,
                                    use_instance_norm, use_style)
        self.conv1 = Conv2d(self.nf(1), self.nf(1), 3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf(1), dlatent_size, use_wscale, use_noise, use_pixel_norm,
                                    use_instance_norm, use_style)
        self.GBlocks = nn.ModuleList()
        for res in range(3, 9):
            self.GBlocks.append(GBlock(res, use_wscale, use_noise, use_pixel_norm,
                                use_instance_norm, self.noise_inputs))

    def forward(self, dlatent):
        images_out = None
        x = self.const_input.expand(dlatent.size(0), -1, -1, -1)
        x = x + self.bias.view(1, -1, 1, 1)
        x = self.adaIn1(x, self.noise_inputs[0], dlatent[:, 0])
        x = self.conv1(x)
        x = self.adaIn2(x, self.noise_inputs[1], dlatent[:, 1])

        # 4 x 4 -> 256 x 256
        for block in self.GBlocks:
            x = block(x, dlatent)
        
        x = self.channel_shrinkage(x)
        img_out = self.torgb(x)
        return img_out

        
class StyleGenerator(nn.Module):
    def __init__(self, 
                 resolution=256,
                 truncation_psi=0.7,
                 truncation_cutoff=6,
                 style_mixing_prob=0.9,
                 mapping_fmaps=128,
                 **kwargs
                 ):
        super(StyleGenerator, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff
        self.style_mixing_prob = style_mixing_prob
        self.mapping = G_mapping(self.mapping_fmaps, **kwargs)
        self.synthesis = G_synthesis(self.mapping_fmaps, **kwargs)

    def forward(self, latents1):
        dlatents1, num_layers =self.mapping(latents1)
        dlatents1 = dlatents1.unsqueeze(1)
        dlatents1 = dlatents1.expand(-1, num_layers, -1)

        #TODO 样式混合技巧

        #截断技巧
        if self.truncation_cutoff and self.truncation_psi:
            coefs = np.ones([1, num_layers, 1], dtype=np.float32)
            for i in range(num_layers):
                if i < self.truncation_cutoff:
                    coefs[:, i, :] *= self.truncation_psi
                
            dlatents1 = dlatents1 * torch.Tensor(coefs).to(dlatents1.device) #dlatents_avg_face = zeros
            
        img = self.synthesis(dlatents1)
        return img


class StyleDiscriminator(nn.Module):
    def __init__(self,
                 num_channels=3,
                 resolution=256,
                 fmap_base=2048,
                 fmap_max=128,
                 fmap_decay=1.0,
                 f=[1,2,1]
                 ):
        super(StyleDiscriminator, self).__init__()
        self.log2_res = int(np.log2(resolution))

        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        self.fromrgb = nn.Conv2d(num_channels, self.nf(self.log2_res-1), 1)
        self.blur2d = Blur2d(f)

        res = self.log2_res
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(self.nf(self.log2_res-1), self.nf(self.log2_res-1), kernel_size=3, padding=1))
        for i in range(5):
            self.convs.append(nn.Conv2d(self.nf(self.log2_res-1-i), self.nf(self.log2_res-2-i), kernel_size=3, padding=1))

        self.downs = nn.ModuleList()
        for i in range(3): # 256 -> 64 avgplooing2d to shrink images, else conv
            self.downs.append(nn.AvgPool2d(2))
        for i in range(3):
            self.downs.append(nn.Conv2d(self.nf(self.log2_res-4-i), self.nf(self.log2_res-4-i), kernel_size=2, stride=2))

        self.conv_last = nn.Conv2d(self.nf(self.log2_res-6), self.nf(1), kernel_size=3, padding=1)
        self.dense0 = nn.Linear(fmap_base, self.nf(0))
        self.dense1 = nn.Linear(self.nf(0), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = F.leaky_relu(self.fromrgb(input), 0.2, inplace=True)
        res = self.log2_res
        # 256 -> 128
        for i in range(6):
            x = F.leaky_relu(self.convs[i](x), 0.2, inplace=True)
            x = F.leaky_relu(self.downs[i](self.blur2d(x)), 0.2, inplace=True)

        # 4 x 4 -> point
        x = F.leaky_relu(self.conv_last(x), 0.2, inplace=True)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.dense0(x), 0.2, inplace=True)
        x = F.leaky_relu(self.dense1(x), 0.2, inplace=True)
        return x


if __name__ == "__main__":
    D = StyleDiscriminator()
    G = StyleGenerator()
    print(D)