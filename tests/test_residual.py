"""Checks for the global residual option (stage 4.3). Run from the repo root:

    python tests/test_residual.py

The reference models are loaded from the commit *before* the change
(REFERENCE_COMMIT) straight out of git, so "identical to before" really means
identical to the code the old runs used, not to a path of the new code.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_factory import create_model  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402

REFERENCE_COMMIT = "6c5e0ff"
ARCHS = {  # name -> (module path, class name, output layer)
    "srcnn": ("models/SRCNN/model.py", "SRCNN", "conv3"),
    "dscms": ("models/DSCMS/model.py", "DSCMS", "final_layers"),
    "prusr": ("models/PRUSR/model.py", "PRUSR", "final_layers.1"),
    "srno": ("models/SRNO/model.py", "SRNO", "fc2"),
    "sofono": ("models/SoFoNO/model.py", "SoFoNO", "fc2"),
}
CNNS = ("srcnn", "dscms", "prusr")
OPERATORS = ("srno", "sofono")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def reference_model(name):
    path, cls, _ = ARCHS[name]
    source = subprocess.check_output(["git", "show", f"{REFERENCE_COMMIT}:{path}"]).decode()
    tmp = tempfile.NamedTemporaryFile("w", suffix=f"_{name}_ref.py", delete=False)
    tmp.write(source)
    tmp.close()
    spec = importlib.util.spec_from_file_location(f"ref_{name}", tmp.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.unlink(tmp.name)
    return getattr(module, cls)(2, 2)


def zero_output_layer(model, name):
    layer = model.get_submodule(ARCHS[name][2])
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()


def inputs():
    g = torch.Generator().manual_seed(0)
    return torch.randn(2, 2, 128, 128, generator=g).to(DEVICE)


def run(model, x):
    model.to(DEVICE).eval()
    with torch.no_grad():
        return model(x)


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    return ok


def main():
    set_seed(0)
    x = inputs()
    results = []

    for name in ARCHS:
        # (e) residual=true, output layer zeroed -> y = x exactly
        model = create_model(name, residual=True)
        zero_output_layer(model.net, name)
        y = run(model, x)
        results.append(check(f"{name}: residual=true, zeroed output layer -> y == x", torch.equal(y, x)))

        # residual=false, output layer zeroed -> y = 0: no hidden skip is left
        model = create_model(name, residual=False)
        zero_output_layer(model.net, name)
        y = run(model, x)
        results.append(check(f"{name}: residual=false, zeroed output layer -> y == 0",
                             torch.count_nonzero(y).item() == 0))

    for name in ARCHS:
        ref = reference_model(name)
        ref_out = run(ref, x)

        # residual=None must still be the untouched original (old checkpoints, notebooks)
        legacy = create_model(name)
        legacy.load_state_dict(ref.state_dict())
        results.append(check(f"{name}: residual=None identical to the pre-change model",
                             torch.equal(run(legacy, x), ref_out)))

        # CNNs: residual=false is the original model (the ablation compares against it).
        # Operators: residual=true (identity skip) reproduces the original bilinear skip.
        residual = name in OPERATORS
        new = create_model(name, residual=residual)
        new.net.load_state_dict(ref.state_dict())
        new_out = run(new, x)
        diff = (new_out - ref_out).abs().max().item()
        results.append(check(f"{name}: residual={str(residual).lower()} with the same weights identical to the pre-change model",
                             torch.equal(new_out, ref_out), f"(max abs diff {diff:.3g})"))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
