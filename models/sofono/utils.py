import torch

def make_coord(shape):
    """
    Returns a (H, W, 2) grid of normalised coordinates in [-1, 1].
    Equivalent to utils.make_coord from the original repo.
    """
    H, W = shape
    ys = torch.linspace(-1 + 1/H, 1 - 1/H, H)
    xs = torch.linspace(-1 + 1/W, 1 - 1/W, W)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([grid_y, grid_x], dim=-1)  # (H, W, 2)