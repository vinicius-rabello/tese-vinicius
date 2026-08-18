from models.DSCMS.model import DSCMS
from models.SRCNN.model import SRCNN
from models.PRUSR.model import PRUSR
from models.SoFoNO.model import SoFoNO
from models.SRNO.model import SRNO

def create_model(model_name: str, in_channels=2, out_channels=2):
    """
    Returns the instantiated model based on the string name.
    """
    name = model_name.lower()

    if name == 'dscms':
        return DSCMS(in_channels, out_channels)
    elif name == 'srcnn':
        return SRCNN(in_channels, out_channels)
    elif name == 'prusr':
        return PRUSR(in_channels, out_channels)
    elif name == 'sofono':
        return SoFoNO(in_channels, out_channels)
    elif name == 'srno':
        return SRNO(in_channels, out_channels)
    else:
        raise ValueError(f"Model '{model_name}' is not supported! Choose from: dscms, srcnn, prusr, sofono, srno.")
