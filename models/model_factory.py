import torch.nn as nn

from models.DSCMS.model import DSCMS
from models.SRCNN.model import SRCNN
from models.PRUSR.model import PRUSR
from models.SoFoNO.model import SoFoNO
from models.SRNO.model import SRNO


class GlobalResidual(nn.Module):
    """y = x + f(x) when enabled, y = f(x) otherwise."""

    def __init__(self, net: nn.Module, enabled: bool):
        super().__init__()
        self.net = net
        self.enabled = enabled

    def forward(self, x):
        out = self.net(x)
        return x + out if self.enabled else out


def create_model(model_name: str, in_channels=2, out_channels=2, residual=None):
    """
    Returns the instantiated model based on the string name.

    residual=None keeps each architecture exactly as originally implemented
    (SRNO/SoFoNO with their built-in bilinear input skip, the CNNs without
    one) -- that is what older checkpoints were trained with. residual=True
    or False gives every architecture the same treatment: the built-in skip
    is removed and the network is wrapped in GlobalResidual.
    """
    name = model_name.lower()

    if name == 'dscms':
        model = DSCMS(in_channels, out_channels)
    elif name == 'srcnn':
        model = SRCNN(in_channels, out_channels)
    elif name == 'prusr':
        model = PRUSR(in_channels, out_channels)
    elif name == 'sofono':
        model = SoFoNO(in_channels, out_channels)
    elif name == 'srno':
        model = SRNO(in_channels, out_channels)
    else:
        raise ValueError(f"Model '{model_name}' is not supported! Choose from: dscms, srcnn, prusr, sofono, srno.")

    if residual is None:
        return model
    if hasattr(model, 'input_skip'):
        model.input_skip = False
    return GlobalResidual(model, enabled=bool(residual))
