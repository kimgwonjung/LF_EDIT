import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Network(nn.Module):
    def __init__(self, imgh, imgw, patch_size, embed_dim, features=[256,256,256,256], dropout = 0.):
        super(Network, self).__init__()

        self.ori_encoder = Encoder(imgh,imgw,patch_size,embed_dim,features,dropout)
        self.tar_encoder = Encoder(imgh,imgw,patch_size,embed_dim,features,dropout)

        self.decoder_1 = Decoder(features[0], features[0])
        self.decoder_2 = Decoder(features[1], features[0])
        self.decoder_3 = Decoder(features[2], features[1])
        self.decoder_4 = Decoder(features[3], features[2])

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, ori, tar):
        ori_layer_1_rn,ori_layer_2_rn,ori_layer_3_rn,ori_layer_4_rn = self.ori_encoder(ori)

        tar_layer_1_rn,tar_layer_2_rn,tar_layer_3_rn,tar_layer_4_rn = self.tar_encoder(tar)
        decoder_4 = self.decoder_4(ori_layer_4_rn, tar_layer_4_rn)
        decoder_3 = self.decoder_3(ori_layer_3_rn, tar_layer_3_rn, decoder_4)
        decoder_2 = self.decoder_2(ori_layer_2_rn, tar_layer_2_rn, decoder_3)
        decoder_1 = self.decoder_1(ori_layer_1_rn, tar_layer_1_rn, decoder_2)
        out = self.out_conv(decoder_1)

        return (out + 1) / 2


class Encoder(nn.Module):
    def __init__(self,imgh, imgw, patch_size, embed_dim, features=[256,256,256,256], dropout = 0.):
        super(Encoder, self).__init__()
        num_patches = (imgh // patch_size) * (imgw // patch_size)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        readout_oper = get_readout_oper(embed_dim, features, use_readout="project", start_index=1)

        self.encoder_1 = Transformer(dim=embed_dim, depth=2, heads=4, dim_head=embed_dim // 4,
                                         mlp_dim=embed_dim * 4)
        self.encoder_2 = Transformer(dim=embed_dim, depth=4, heads=4, dim_head=embed_dim // 4,
                                         mlp_dim=embed_dim * 4)
        self.encoder_3 = Transformer(dim=embed_dim, depth=3, heads=4, dim_head=embed_dim // 4,
                                         mlp_dim=embed_dim * 4)
        self.encoder_4 = Transformer(dim=embed_dim, depth=3, heads=4, dim_head=embed_dim // 4,
                                         mlp_dim=embed_dim * 4)

        self.post_processing1 = nn.Sequential(
            readout_oper[0],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([imgh // patch_size, imgw // patch_size])),
            nn.Conv2d(in_channels=embed_dim, out_channels=features[0], kernel_size=1, stride=1, padding=0),
            nn.ConvTranspose2d(in_channels=features[0], out_channels=features[0], kernel_size=6, stride=6, padding=0)
        )
        self.post_processing2 = nn.Sequential(
            readout_oper[1],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([imgh // patch_size, imgw // patch_size])),
            nn.Conv2d(in_channels=embed_dim, out_channels=features[1], kernel_size=1, stride=1, padding=0),
            nn.ConvTranspose2d(in_channels=features[1], out_channels=features[1], kernel_size=3, stride=3, padding=0)
        )
        self.post_processing3 = nn.Sequential(
            readout_oper[2],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([imgh // patch_size, imgw // patch_size])),
            nn.Conv2d(in_channels=embed_dim, out_channels=features[2], kernel_size=1, stride=1, padding=0),
            Interpolate(scale_factor=1.5, mode="bilinear")
        )
        self.post_processing4 = nn.Sequential(
            readout_oper[3],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([imgh // patch_size, imgw // patch_size])),
            nn.Conv2d(in_channels=embed_dim, out_channels=features[3], kernel_size=1, stride=1, padding=0),
            Interpolate(scale_factor=0.75, mode="bilinear")
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        layer_1 = self.encoder_1(x)
        layer_1_rn = self.post_processing1(layer_1)
        layer_2 = self.encoder_2(layer_1)
        layer_2_rn = self.post_processing2(layer_2)
        layer_3 = self.encoder_3(layer_2)
        layer_3_rn = self.post_processing3(layer_3)
        layer_4 = self.encoder_4(layer_3)
        layer_4_rn = self.post_processing4(layer_4)

        return [layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn]


class Decoder(nn.Module):
    def __init__(self, features, out_features, bn=True):
        super(Decoder, self).__init__()
        self.proj = nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=1, stride=1, padding=0)

        self.resUnit1 = ResidualConvUnit(features, bn=bn)
        self.resUnit2 = ResidualConvUnit(features, bn=bn)

        self.out_conv = nn.Conv2d(in_channels=features, out_channels=out_features, kernel_size=1, stride=1, padding=0)

    def forward(self, *xs):
        ori = xs[0]
        tar = xs[1]
        output = self.proj(torch.cat((ori, tar), dim=1))
        if len(xs) == 3:
            res = self.resUnit1(xs[2])
            output = output + res
        output = self.resUnit2(output)

        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        output = self.out_conv(output)
        return output
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x
    
class ResidualConvUnit(nn.Module):
    def __init__(self, features, bn=False):
        super(ResidualConvUnit, self).__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()

        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        return out + x



if __name__ == '__main__':
    model = Network(384, 384, 12, 432)
    ori = torch.randn((1, 3, 384, 384), dtype=torch.float)
    tar = torch.randn((1, 3, 384, 384), dtype=torch.float)
    output = model(ori, tar)
    print(output.shape)



