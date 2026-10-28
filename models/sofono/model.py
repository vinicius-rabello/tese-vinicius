import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.sofono.config as config

# =============================================================================
# SHARED HELPERS
# =============================================================================

def default_conv(in_ch, out_ch, kernel_size, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size,
                     padding=kernel_size // 2, bias=bias)


def make_coord(shape, ranges=None):
    """
    Returns a (H, W, 2) grid of normalised coordinates.

    Args:
        shape:  (H, W) tuple
        ranges: (min, max) coordinate range — defaults to (-1, 1) if None
    """
    H, W = shape
    lo, hi = (-1, 1) if ranges is None else ranges
    ys = torch.linspace(lo + (hi - lo) / H, hi - (hi - lo) / H, H)
    xs = torch.linspace(lo + (hi - lo) / W, hi - (hi - lo) / W, W)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([grid_y, grid_x], dim=-1)          # (H, W, 2)


# =============================================================================
# ENCODER — EDSR (2-channel)
# =============================================================================

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
# BLOCKS — Galerkin attention  &  SoFoNO block
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
    """
    simple_attn from galerkin.py — inlined, @register decorator removed.
    """
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

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H * W, self.heads, 3 * self.headc)
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


class AdaIN(nn.Module):
    # https://github.com/CellEight/Pytorch-Adaptive-Instance-Normalization/blob/master/AdaIN.py
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

    def forward(self, x, y):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        return (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])

class SoFoNO_block(nn.Module):
    def __init__(self, width=256, local_branch='Conv', blocks=16):
        super().__init__()
        self.width = width

        self.local_branch = local_branch
        if self.local_branch=='Conv':
            self.local_b = nn.Sequential(
                nn.Conv2d(self.width, self.width, 1),
                nn.GELU(),
            )
        elif self.local_branch=='Attn':
            self.blocks = blocks
            self.headc = self.width // self.blocks
            self.heads = self.blocks
            self.midc = self.width

            self.qkv_proj = nn.Conv2d(self.width, 3*self.width, 1)

            self.kln = LayerNorm((self.heads, 1, self.headc))
            self.vln = LayerNorm((self.heads, 1, self.headc))
            self.o_proj1 = nn.Conv2d(self.midc, self.midc, 1)
            self.o_proj2 = nn.Conv2d(self.midc, self.midc, 1)

            self.act = nn.GELU()

        self.cross_g = nn.Sequential(
                nn.Conv2d(self.width//2, self.width//2, 1),
                nn.GELU()
            )
        self.cross_l = nn.Sequential(
                nn.Conv2d(self.width//2, self.width//2, 1),
                nn.GELU()
            )

        self.adain = AdaIN()

        self.cat_conv = nn.Sequential(
                nn.Conv2d(self.width, self.width, 1),
                nn.GELU(),
                nn.Conv2d(self.width, self.width, 1),
                nn.GELU(),
            )

    def SoFoNO_LayerNorm(self, x, eps=1e-5, dim=1):
        mean = x.mean(dim, keepdim=True) 
        std = x.std(dim, keepdim=True)

        return (x - mean) / (std + eps)

    def our_transform(self, x, domain):
        F_x = torch.fft.fft2(x)
        F_x = torch.fft.fftshift(F_x, dim=(-2, -1))

        F_x *= (domain / domain.mean(dim=(-2, -1))) 

        out_ft = torch.fft.ifftshift(F_x, dim=(-2, -1))
        tran_x = torch.fft.ifft2(out_ft, s=(out_ft.size(-2), out_ft.size(-1)), dim=(-2, -1)).real

        return tran_x

    def forward(self, x, s, idx, T):
        prev_x = x.clone()
        N_y = x.size(-1)
        N_x = x.size(-2)

        ind_x = torch.arange(-N_x/2,N_x/2).to(config.DEVICE) * (2 * torch.pi / N_x)
        ind_y = torch.arange(-N_y/2,N_y/2).to(config.DEVICE) * (2 * torch.pi / N_y)

        xy_domain =  (1 + (ind_y**2) + (ind_x[:,None]**2))**(s/2)

        tran_x = self.our_transform(x, xy_domain)

        if self.local_branch=='Conv':
            x = self.local_b(self.SoFoNO_LayerNorm(x))

        elif self.local_branch=='Attn':
            B, C, H, W = x.shape
            
            prev_x = x.clone()
            qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc)
            qkv = qkv.permute(0, 2, 1, 3) # (B, self.heads, H*W, 3*self.headc)
            q, k, v = qkv.chunk(3, dim=-1)

            v = self.vln(v)
            k = self.kln(k)

            v = torch.matmul(k.transpose(-2,-1), v) / (H*W)

            v = v.permute(0, 3, 1, 2)
            v = torch.matmul(q, v)
            v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

            ret = v.permute(0, 3, 1, 2) + prev_x
            x = self.o_proj2(self.act(self.o_proj1(ret))) + prev_x

        x, cross_x = x.chunk(2, dim=1)
        tran_x, cross_tran_x = tran_x.chunk(2, dim=1)
        cross_local = self.cross_l(cross_x)
        cross_global = self.cross_g(cross_tran_x)
        x = self.adain(x, cross_global); tran_x = self.adain(tran_x, cross_local)

        cat_x = torch.cat((x, tran_x), dim=1) + prev_x

        x = self.cat_conv(cat_x)
        
        return x + prev_x


# =============================================================================
# MODEL — SoFoNO  (adapted for arbitrary n_out and encoder injection)
# =============================================================================

class SoFoNO(nn.Module):
    """
    SoFoNO from sofono.py, adapted to match model.py conventions:

      - @register decorator removed
      - encoder injected directly (no models.make registry)
      - SoFoNO_block / simple_attn / make_coord inlined (no external deps)
      - hard-coded .cuda() replaced with .to(feat.device)
      - fc2 output generalised from 3 (RGB) → n_out (default 2 for u, v)
      - show_feature_map debug call dropped
      - full smoke test added at the bottom
    """
    def __init__(self, encoder, width=256, T=2, ranges=None,
                 local_branch='Conv', init_s=0, n_out=2):
        """
        Args:
            encoder      : any nn.Module with an `out_dim` attribute
            width        : channel width inside the network
            T            : number of SoFoNO blocks
            ranges       : coordinate range tuple (min, max); None → (-1, 1)
            local_branch : 'Conv' or 'Attn' for the local branch in each block
            init_s       : initial value of the learnable gate scalar s
            n_out        : number of output channels (2 for u, v velocity)
        """
        super().__init__()
        self.width  = width
        self.T      = T
        self.ranges = ranges

        self.encoder = encoder
        enc_feats    = encoder.out_dim     # e.g. 64

        # 4 neighbours × (feat_channels + 2 rel_coords) + 2 rel_cell
        self.conv00 = nn.Conv2d((enc_feats + 2) * 4 + 2, width, 1)

        # learnable gate scalar shared across all SoFoNO blocks
        self.s = nn.Parameter(torch.tensor(float(init_s)))

        self.fno_blocks = nn.ModuleList([
            SoFoNO_block(width=width, local_branch=local_branch)
            for _ in range(T)
        ])

        self.fc1 = nn.Conv2d(width,  width,  1)
        self.fc2 = nn.Conv2d(width,  n_out,  1)

    # ------------------------------------------------------------------
    def gen_feat(self, inp):
        self.inp  = inp
        self.feat = self.encoder(inp)

    # ------------------------------------------------------------------
    def query(self, coord, cell):
        feat   = self.feat
        device = feat.device

        # coordinate grid of the LR feature map — use device of feat, not .cuda()
        pos_lr = make_coord(feat.shape[-2:], ranges=self.ranges).to(device)
        pos_lr = pos_lr.permute(2, 0, 1).unsqueeze(0).expand(
            feat.shape[0], 2, *feat.shape[-2:])

        lo, hi = (-1, 1) if self.ranges is None else self.ranges
        rx = (hi - lo) / feat.shape[-2] / 2
        ry = (hi - lo) / feat.shape[-1] / 2
        eps_shift = 1e-6

        domain_min = lo + 1e-6
        domain_max = hi - 1e-6

        rel_coords, feat_s, areas = [], [], []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[..., 0] += vx * rx + eps_shift
                coord_[..., 1] += vy * ry + eps_shift
                coord_.clamp_(domain_min, domain_max)

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
        # swap areas so weights are assigned to the opposite neighbours
        areas[0], areas[3] = areas[3], areas[0]
        areas[1], areas[2] = areas[2], areas[1]

        for i, area in enumerate(areas):
            feat_s[i] = feat_s[i] * (area / tot_area).unsqueeze(1)

        rel_cell_spatial = rel_cell[:, :, None, None].expand(
            -1, -1, coord.shape[1], coord.shape[2])

        grid = torch.cat([*rel_coords, *feat_s, rel_cell_spatial], dim=1)

        x = self.conv00(grid)
        for i, block in enumerate(self.fno_blocks):
            x = block(x, self.s, i, self.T)

        ret = self.fc2(F.gelu(self.fc1(x)))

        # bilinear residual from the LR input
        ret = ret + F.grid_sample(self.inp, coord.flip(-1),
                                  mode='bilinear', padding_mode='border',
                                  align_corners=False)
        return ret

    # ------------------------------------------------------------------
    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query(coord, cell)


# =============================================================================
# SMOKE TEST
# =============================================================================

def test():
    batch_size = 16
    lr_size    = 8     # LR input spatial size
    hr_size    = 32    # HR target spatial size (what coord / cell describe)
    in_ch      = 2     # u, v velocity channels
    out_ch     = 2

    # --- inputs ---
    inp   = torch.randn(batch_size, in_ch, lr_size, lr_size)
    coord = make_coord((hr_size, hr_size)) \
                .unsqueeze(0).expand(batch_size, -1, -1, -1)   # (B, H, W, 2)
    cell  = torch.tensor([[2.0 / hr_size, 2.0 / hr_size]]) \
                .expand(batch_size, -1)                         # (B, 2)

    print('=== Inputs ===')
    print(f'  inp   : {tuple(inp.shape)}')
    print(f'  coord : {tuple(coord.shape)}')
    print(f'  cell  : {tuple(cell.shape)}')

    for branch in ('Conv', 'Attn'):
        print(f'\n=== SoFoNO  local_branch={branch!r} ===')
        encoder = EDSREncoder(n_resblocks=16, n_feats=64, res_scale=1.0, n_colors=in_ch)
        model   = SoFoNO(encoder=encoder, width=256, T=2,
                         local_branch=branch, init_s=0, n_out=out_ch)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Total params : {n_params:,}')

        model.eval()
        with torch.no_grad():
            pred = model(inp, coord, cell)

        assert pred.shape == (batch_size, out_ch, hr_size, hr_size), \
            f'Unexpected output shape: {pred.shape}'
        print(f'  pred  : {tuple(pred.shape)}  ✓')
        print(f'  pred  min={pred.min():.4f}  max={pred.max():.4f}  mean={pred.mean():.4f}')


if __name__ == '__main__':
    test()