"""SoFoNO (Spectral operator + Fourier Neural Operator) super-resolution model.

Ported from the standalone SoFoNO reference implementation (SoFoNO/models/sofono.py,
SoFoNO/models/block.py, SoFoNO/models/edsr.py) so it plugs into this project's
model_factory / centralized train.py the same way SRCNN, PRUSR and DSCMS do.

The original SoFoNO is a continuous (arbitrary-scale) implicit operator: it takes a
low-res image plus a query coordinate grid + cell size, and predicts values at those
coordinates. This project's dataset always returns LR/HR pairs at the same fixed
resolution, so the coord/cell query grid is built internally from the input shape
(matching a scale=1 query in the original codebase) and `forward(x)` returns an
image of the same shape as `x`, matching the interface every other model here uses.

Default hyperparameters (width=256, T=3, ranges=[-1, 1], local_branch='Conv',
init_s=0.0, edsr-baseline encoder with n_resblocks=16/n_feats=64) match the
`train_edsr-SoFoNO_vinicius*.yaml` configs used for the reference training runs
under SoFoNO/save/, so pretrained checkpoints from those runs load with matching
module names/shapes.
"""

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for n in shape:
        v0, v1 = (-1, 1) if ranges is None else ranges
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


# --- EDSR-baseline encoder (feature extractor, no upsampling) ---

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super().__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        super().__init__(*m)


class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super().__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        m_body = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size),
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        return x


def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1, scale=2,
                        no_upsampling=True, rgb_range=1, n_colors=2):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.rgb_range = rgb_range
    args.n_colors = n_colors
    return EDSR(args)


# --- SoFoNO spectral/local mixing block ---

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out = (x - mean) / (std + self.eps)
        return self.weight * out + self.bias


class AdaIN(nn.Module):
    """https://github.com/CellEight/Pytorch-Adaptive-Instance-Normalization"""

    def mu(self, x):
        return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])

    def sigma(self, x):
        return torch.sqrt(
            (torch.sum((x.permute([2, 3, 0, 1]) - self.mu(x)).permute([2, 3, 0, 1]) ** 2, (2, 3)) + 0.000000023)
            / (x.shape[2] * x.shape[3])
        )

    def forward(self, x, y):
        return (self.sigma(y) * ((x.permute([2, 3, 0, 1]) - self.mu(x)) / self.sigma(x)) + self.mu(y)).permute(
            [2, 3, 0, 1]
        )


class SoFoNOBlock(nn.Module):
    def __init__(self, width=256, local_branch='Conv', blocks=16):
        super().__init__()
        self.width = width
        self.local_branch = local_branch

        if self.local_branch == 'Conv':
            self.local_b = nn.Sequential(nn.Conv2d(self.width, self.width, 1), nn.GELU())
        elif self.local_branch == 'Attn':
            self.blocks = blocks
            self.headc = self.width // self.blocks
            self.heads = self.blocks
            self.midc = self.width

            self.qkv_proj = nn.Conv2d(self.width, 3 * self.width, 1)
            self.kln = LayerNorm((self.heads, 1, self.headc))
            self.vln = LayerNorm((self.heads, 1, self.headc))
            self.o_proj1 = nn.Conv2d(self.midc, self.midc, 1)
            self.o_proj2 = nn.Conv2d(self.midc, self.midc, 1)
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown local_branch '{local_branch}', expected 'Conv' or 'Attn'.")

        self.cross_g = nn.Sequential(nn.Conv2d(self.width // 2, self.width // 2, 1), nn.GELU())
        self.cross_l = nn.Sequential(nn.Conv2d(self.width // 2, self.width // 2, 1), nn.GELU())
        self.adain = AdaIN()
        self.cat_conv = nn.Sequential(
            nn.Conv2d(self.width, self.width, 1),
            nn.GELU(),
            nn.Conv2d(self.width, self.width, 1),
            nn.GELU(),
        )

    def sofono_layer_norm(self, x, eps=1e-5, dim=1):
        mean = x.mean(dim, keepdim=True)
        std = x.std(dim, keepdim=True)
        return (x - mean) / (std + eps)

    def spectral_transform(self, x, domain):
        f_x = torch.fft.fft2(x)
        f_x = torch.fft.fftshift(f_x, dim=(-2, -1))
        f_x = f_x * (domain / domain.mean(dim=(-2, -1)))
        out_ft = torch.fft.ifftshift(f_x, dim=(-2, -1))
        return torch.fft.ifft2(out_ft, s=(out_ft.size(-2), out_ft.size(-1)), dim=(-2, -1)).real

    def forward(self, x, s, idx, T):
        prev_x = x.clone()
        n_y = x.size(-1)
        n_x = x.size(-2)

        ind_x = torch.arange(-n_x / 2, n_x / 2, device=x.device) * (2 * torch.pi / n_x)
        ind_y = torch.arange(-n_y / 2, n_y / 2, device=x.device) * (2 * torch.pi / n_y)
        xy_domain = (1 + (ind_y ** 2) + (ind_x[:, None] ** 2)) ** (s / 2)

        tran_x = self.spectral_transform(x, xy_domain)

        if self.local_branch == 'Conv':
            x = self.local_b(self.sofono_layer_norm(x))
        else:  # Attn
            b, c, h, w = x.shape
            prev_x = x.clone()
            qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(b, h * w, self.heads, 3 * self.headc)
            qkv = qkv.permute(0, 2, 1, 3)
            q, k, v = qkv.chunk(3, dim=-1)

            v = self.vln(v)
            k = self.kln(k)
            v = torch.matmul(k.transpose(-2, -1), v) / (h * w)
            v = v.permute(0, 3, 1, 2)
            v = torch.matmul(q, v)
            v = v.permute(0, 2, 1, 3).reshape(b, h, w, c)

            ret = v.permute(0, 3, 1, 2) + prev_x
            x = self.o_proj2(self.act(self.o_proj1(ret))) + prev_x

        x, cross_x = x.chunk(2, dim=1)
        tran_x, cross_tran_x = tran_x.chunk(2, dim=1)
        cross_local = self.cross_l(cross_x)
        cross_global = self.cross_g(cross_tran_x)
        x = self.adain(x, cross_global)
        tran_x = self.adain(tran_x, cross_local)

        cat_x = torch.cat((x, tran_x), dim=1) + prev_x
        x = self.cat_conv(cat_x)

        return x + prev_x


class SoFoNO(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, width=256, T=3,
                 ranges=(-1, 1), local_branch='Conv', init_s=0.0,
                 encoder_n_resblocks=16, encoder_n_feats=64):
        super().__init__()
        self.width = width
        self.T = T
        self.ranges = ranges
        self.out_channels = out_channels

        self.encoder = make_edsr_baseline(
            n_resblocks=encoder_n_resblocks,
            n_feats=encoder_n_feats,
            no_upsampling=True,
            n_colors=in_channels,
        )
        self.conv00 = nn.Conv2d((encoder_n_feats + 2) * 4 + 2, self.width, 1)

        self.local_branch = local_branch
        self.s = nn.Parameter(torch.tensor(float(init_s), dtype=torch.float32))
        self.fno_kernels = nn.ModuleList(
            [SoFoNOBlock(width=self.width, local_branch=self.local_branch) for _ in range(T)]
        )

        self.fc1 = nn.Conv2d(self.width, self.width, 1)
        self.fc2 = nn.Conv2d(self.width, out_channels, 1)
        # The original model adds a bilinear resample of the input (a global
        # skip). At scale 1 that resample is exactly the identity, so
        # create_model(residual=...) turns it off and adds x itself, giving
        # every architecture the same skip.
        self.input_skip = True

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell):
        feat = self.feat
        device = feat.device

        pos_lr = (
            make_coord(feat.shape[-2:], flatten=False, ranges=self.ranges)
            .to(device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(feat.shape[0], 2, *feat.shape[-2:])
        )

        domain_min, domain_max = (-1, 1) if self.ranges is None else self.ranges
        rx = (domain_max - domain_min) / feat.shape[-2] / 2
        ry = (domain_max - domain_min) / feat.shape[-1] / 2
        eps_shift = 1e-6

        rel_coords, feat_s, areas = [], [], []
        for vx in (-1, 1):
            for vy in (-1, 1):
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(domain_min + 1e-6, domain_max - 1e-6)

                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)

                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)
                rel_coords.append(rel_coord)
                feat_s.append(feat_)

        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-2]
        rel_cell[:, 1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)
        areas[0], areas[3] = areas[3], areas[0]
        areas[1], areas[2] = areas[2], areas[1]

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)

        grid = torch.cat(
            [*rel_coords, *feat_s, rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, coord.shape[1], coord.shape[2])],
            dim=1,
        )

        x = self.conv00(grid)
        for i, kernel in enumerate(self.fno_kernels):
            x = kernel(x, self.s, i, self.T)

        ret = self.fc2(F.gelu(self.fc1(x)))
        if self.input_skip:
            ret = ret + F.grid_sample(
                self.inp, coord.flip(-1), mode='bilinear', padding_mode='border', align_corners=False
            )
        return ret

    def _query_grid_for(self, x):
        """Build a full-resolution (coord, cell) query grid matching x's own H, W.

        This project's dataset always pairs LR/HR at the same fixed resolution, so
        querying at the input's own grid reproduces a scale=1 query in the original
        SoFoNO training pipeline.
        """
        b, _, h, w = x.shape
        coord = make_coord((h, w), ranges=self.ranges, flatten=False).to(x.device)
        coord = coord.unsqueeze(0).expand(b, h, w, 2).clone()

        domain_min, domain_max = (-1, 1) if self.ranges is None else self.ranges
        cell = torch.tensor(
            [(domain_max - domain_min) / h, (domain_max - domain_min) / w],
            device=x.device, dtype=x.dtype,
        ).unsqueeze(0).expand(b, 2).clone()

        return coord, cell

    def forward(self, x):
        coord, cell = self._query_grid_for(x)
        self.gen_feat(x)
        return self.query_rgb(coord, cell)
