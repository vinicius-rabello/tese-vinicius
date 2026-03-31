import os
import math
import glob
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# =============================================================================
# MODEL — EDSR encoder (2-channel)
# =============================================================================

def default_conv(in_ch, out_ch, kernel_size, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size,
                     padding=kernel_size // 2, bias=bias)

def make_coord(shape):
    """
    Returns a (H, W, 2) grid of normalised coordinates in [-1, 1].
    Inlined from utils.make_coord — no external dependency needed.
    """
    H, W = shape
    ys = torch.linspace(-1 + 1/H, 1 - 1/H, H)
    xs = torch.linspace(-1 + 1/W, 1 - 1/W, W)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([grid_y, grid_x], dim=-1)  # (H, W, 2)


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            default_conv(n_feats, n_feats, kernel_size),
            nn.ReLU(inplace=True),
            default_conv(n_feats, n_feats, kernel_size),
        )

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class EDSREncoder(nn.Module):
    """
    EDSR without the upsampling tail — outputs feature maps (B, n_feats, H, W).
    Adapted for 2-channel (u, v) input instead of RGB.
    """
    def __init__(self, n_resblocks=16, n_feats=64, res_scale=1.0, n_colors=2):
        super().__init__()
        self.out_dim = n_feats
        self.head = default_conv(n_colors, n_feats, 3)
        self.body = nn.Sequential(
            *[ResBlock(n_feats, 3, res_scale) for _ in range(n_resblocks)],
            default_conv(n_feats, n_feats, 3),
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        return x + res


# =============================================================================
# MODEL — Galerkin attention (copied verbatim, minus the @register decorator)
# =============================================================================

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias   = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1,  keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class GalerkinAttn(nn.Module):
    """simple_attn from galerkin.py, minus the @register decorator."""
    def __init__(self, midc, heads):
        super().__init__()
        self.headc = midc // heads
        self.heads = heads
        self.midc  = midc

        self.qkv_proj = nn.Conv2d(midc, 3 * midc, 1)
        self.o_proj1  = nn.Conv2d(midc, midc, 1)
        self.o_proj2  = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2, -1), v) / (H * W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

        ret  = v.permute(0, 3, 1, 2) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        return bias


# =============================================================================
# MODEL — SRONET (adapted for 2-channel output)
# =============================================================================

class SRONET(nn.Module):
    """
    SRNO from sronet.py, with:
      - encoder injected directly (no models.make registry)
      - fc2 output changed from 3 → n_out (2 for u,v)
      - make_coord inlined
      - 'name' arg removed from GalerkinAttn.forward()
    """
    def __init__(self, encoder, width=256, blocks=16, n_out=2):
        super().__init__()
        self.width   = width
        self.encoder = encoder
        enc_feats    = encoder.out_dim   # 64

        # 4 neighbours × (feat_channels + 2 rel_coords) + 2 rel_cell = (64+2)*4+2
        self.conv00 = nn.Conv2d((enc_feats + 2) * 4 + 2, width, 1)
        self.conv0  = GalerkinAttn(width, blocks)
        self.conv1  = GalerkinAttn(width, blocks)

        self.fc1 = nn.Conv2d(width,  256,   1)
        self.fc2 = nn.Conv2d(256,    n_out, 1)

    def gen_feat(self, inp):
        self.inp  = inp
        self.feat = self.encoder(inp)

    def query(self, coord, cell):
        feat = self.feat
        B    = feat.shape[0]

        # coordinate grid of the LR feature map
        pos_lr = make_coord(feat.shape[-2:]).to(feat.device)   # (H, W, 2)
        pos_lr = pos_lr.permute(2, 0, 1).unsqueeze(0).expand(B, 2, *feat.shape[-2:])

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        eps_shift = 1e-6

        rel_coords, feat_s, areas = [], [], []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[..., 0] += vx * rx + eps_shift
                coord_[..., 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                feat_ = F.grid_sample(feat, coord_.flip(-1),
                                      mode='nearest', align_corners=False)
                old_coord = F.grid_sample(pos_lr, coord_.flip(-1),
                                          mode='nearest', align_corners=False)

                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0] *= feat.shape[-2]
                rel_coord[:, 1] *= feat.shape[-1]

                area = torch.abs(rel_coord[:, 0] * rel_coord[:, 1])
                areas.append(area + 1e-9)
                rel_coords.append(rel_coord)
                feat_s.append(feat_)

        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-2]
        rel_cell[:, 1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(0)
        # swap areas so weights are correctly assigned to opposite neighbours
        areas[0], areas[3] = areas[3], areas[0]
        areas[1], areas[2] = areas[2], areas[1]

        for i, area in enumerate(areas):
            feat_s[i] = feat_s[i] * (area / tot_area).unsqueeze(1)

        rel_cell_spatial = rel_cell[:, :, None, None].expand(
            -1, -1, coord.shape[1], coord.shape[2])

        grid = torch.cat([*rel_coords, *feat_s, rel_cell_spatial], dim=1)

        x   = self.conv00(grid)
        x   = self.conv0(x)
        x   = self.conv1(x)
        ret = self.fc2(F.gelu(self.fc1(x)))

        # bilinear residual from the LR input
        ret = ret + F.grid_sample(self.inp, coord.flip(-1),
                                  mode='bilinear', padding_mode='border',
                                  align_corners=False)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query(coord, cell)


# =============================================================================
# SMOKE TEST
# =============================================================================

def test():
    batch_size = 16
    lr_size    = 8    # LR input spatial size
    hr_size    = 32   # HR target spatial size (what coord/cell describe)
    in_ch      = 2     # u, v velocity channels
    out_ch     = 2

    # --- inputs ---
    inp   = torch.randn(batch_size, in_ch, lr_size, lr_size)
    coord = make_coord((hr_size, hr_size)) \
                .unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, H, W, 2)
    cell  = torch.tensor([[2.0 / hr_size, 2.0 / hr_size]]) \
                .expand(batch_size, -1)                        # (B, 2)

    print('=== Inputs ===')
    print(f'  inp   : {tuple(inp.shape)}')
    print(f'  coord : {tuple(coord.shape)}')
    print(f'  cell  : {tuple(cell.shape)}')

    # --- build model ---
    encoder = EDSREncoder(n_resblocks=16, n_feats=64, res_scale=1.0, n_colors=in_ch)
    model   = SRONET(encoder=encoder, width=256, blocks=16, n_out=out_ch)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n=== Model ===')
    print(f'  EDSREncoder  out_dim : {encoder.out_dim}')
    print(f'  SRONET       n_out   : {out_ch}')
    print(f'  Total params         : {n_params:,}')

    # --- forward pass ---
    model.eval()
    with torch.no_grad():
        pred = model(inp, coord, cell)

    print(f'\n=== Output ===')
    print(f'  pred  : {tuple(pred.shape)}')   # expect (B, 2, hr_size, hr_size)
    assert pred.shape == (batch_size, out_ch, hr_size, hr_size), \
        f'Unexpected output shape: {pred.shape}'
    print(f'  ✓ shape is correct: (B={batch_size}, C={out_ch}, H={hr_size}, W={hr_size})')
    print(f'  pred  min={pred.min():.4f}  max={pred.max():.4f}  mean={pred.mean():.4f}')


if __name__ == '__main__':
    test()