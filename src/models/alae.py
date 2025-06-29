from src.models.costume_layers import *

"""
All the sub_modules here are sub_modules devised in https://arxiv.org/abs/2004.04467
"""


class AlaeStyleEncoderBlock(nn.Module):
    def __init__(self, latent_dim, in_channels, out_channels, downsample=False, is_last_block=False):
        super(AlaeStyleEncoderBlock, self).__init__()
        assert not (is_last_block and downsample), "You should not downscale after last block"
        self.downsample = downsample
        self.is_last_block = is_last_block
        self.conv1 = Lreq_Conv2d(in_channels, in_channels, 3, 1)
        self.lrelu = nn.LeakyReLU(0.2)
#         self.instance_norm_1 = StyleInstanceNorm2d(in_channels)
        self.pixel_norm_1 = StylePixelNorm2d()
        self.c_1 = LREQ_FC_Layer(2 * in_channels, latent_dim)
        if is_last_block:
            self.conv2 = Lreq_Conv2d(in_channels, out_channels, STARTING_DIM, 0)
            self.c_2 = LREQ_FC_Layer(out_channels, latent_dim)
        else:
            scale = 2 if downsample else 1
            self.conv2 = torch.nn.Sequential(LearnablePreScaleBlur(in_channels),
                                             Lreq_Conv2d(in_channels, out_channels, 3, 1),
                                             torch.nn.AvgPool2d(scale, scale))
#             self.instance_norm_2 = StyleInstanceNorm2d(out_channels)
            self.pixel_norm_2 = StylePixelNorm2d()
            self.c_2 = LREQ_FC_Layer(2 * out_channels, latent_dim)

        self.name = f"EncodeBlock({latent_dim}, {in_channels}, {out_channels}, is_last_block={is_last_block}, downsample={downsample})"

    def __str__(self):
        return self.name

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)

#         x, style_1 = self.instance_norm_1(x)
        x, style_1 = self.pixel_norm_1(x)
        w1 = self.c_1(style_1.squeeze(3).squeeze(2))

        x = self.conv2(x)
        x = self.lrelu(x)
        if self.is_last_block:
            w2 = self.c_2(x.squeeze(3).squeeze(2))
        else:
#             x, style_2 = self.instance_norm_2(x)
            x, style_2 = self.pixel_norm_2(x)
            w2 = self.c_2(style_2.squeeze(3).squeeze(2))

        return x, w1, w2


class AlaeEncoder(nn.Module):
    """
    The Style version of ALAE encoder
    """
    def __init__(self, latent_dim, progression):
        """
        progression: A list of tuples (<out_res>, <out_channels>) that describes the Encoding blocks this module should have
        """
        super().__init__()
        assert progression[0][0] == STARTING_DIM, f"Last module should note downscale so first out_dim should be {STARTING_DIM}"
        self.latent_size = latent_dim
        self.from_rgbs = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        # Parse the module description given in "progression"
        for i in range(len(progression)):
            self.from_rgbs.append(Lreq_Conv2d(3, progression[i][1], 1, 0))
        for i in range(len(progression) - 1, -1, -1):
            if i == 0:
                self.conv_blocks.append(AlaeStyleEncoderBlock(latent_dim, progression[i][1], STARTING_CHANNELS,
                                                              is_last_block=True))
            else:
                downsample = progression[i][0] / 2 == progression[i - 1][0]
                self.conv_blocks.append(AlaeStyleEncoderBlock(latent_dim, progression[i][1], progression[i - 1][1],
                                                              downsample=downsample))


        assert(len(self.conv_blocks) == len(self.from_rgbs))
        self.n_layers = len(self.conv_blocks)
        
        self.linear = nn.Linear(self.latent_size, 1)

    def __str__(self):
        name = "Style-Encoder:\n"
        name += "\tfromRgbs\n"
        for i in range(len(self.from_rgbs)):
            name += f"\t {self.from_rgbs[i]}\n"
        name += "\tStyleEncoderBlocks\n"
        for i in range(len(self.conv_blocks)):
            name += f"\t {self.conv_blocks[i]}\n"
        return name

    def forward(self, image, final_resolution_idx, alpha=1):
        latent_vector = torch.zeros(image.shape[0], self.latent_size).to(image.device)

        feature_maps = self.from_rgbs[final_resolution_idx](image)

        first_layer_idx = self.n_layers - final_resolution_idx - 1
        for i in range(first_layer_idx, self.n_layers):
            feature_maps, w1, w2 = self.conv_blocks[i](feature_maps)
            latent_vector += w1 + w2

            # If this is the first conv block to be run and this is not the last one the there is an already stabilized
            # previous scale layers : Alpha blend the output of the unstable new layer with the downscaled putput
            # of the previous one
            if i == first_layer_idx and i != self.n_layers - 1 and alpha < 1:
                if self.conv_blocks[i].downsample:
                    image = downscale_2d(image)
                skip_first_block_feature_maps = self.from_rgbs[final_resolution_idx - 1](image)
                feature_maps = alpha * feature_maps + (1 - alpha) * skip_first_block_feature_maps
                
        output = self.linear(latent_vector)

        return output