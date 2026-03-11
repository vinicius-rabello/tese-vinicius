# Super-Resolution for HYCON Dataset

Deep learning framework for super-resolution of HYCON dataset. Implements multiple state-of-the-art models in a modular architecture.

## Project Structure

Each model is an independent module under `models/`:

```
models/
├── DSCMS/          # Downsampled Skip-Connection Multi-Scale
├── PRUSR/          # Progressive Residual U-Net
├── SRCNN/          # Super-Resolution CNN
└── CycleGAN/       # Cycle-Consistent GAN

datasets/           # Dataset loaders
metrics/            # Evaluation metrics (MSE, SSIM, PSNR, LPIPS, FSIM, EPI, TKE)
```

## Installation

```bash
python -m venv venv
source ./venv/bin/activate  # Windows: ./venv/Scripts/activate
pip install -r requirements.txt
```

## Training

Train any model:

```bash
python main.py --model <model_name> --training-strategy default
```

Models: `dscms`, `prusr`, `srcnn`, `cyclegan`

For SRCNN multistage training:

```bash
python main.py --model srcnn --training-strategy multistage
```

Configuration: Edit `models/<MODEL>/config.py` for hyperparameters.

## Evaluation

Evaluate trained models or baselines:

```bash
python evaluate_model_updated.py --model <model_name>
```

Models: `DSCMS`, `PRUSR`, `bilinear`, `bicubic`

Results saved to `models/<model_name>/results.json`

## Visualization

Plot training losses:

```bash
python plot_losses.py --model <model_name>  # Single model
python plot_losses.py --model all           # Compare all models
```

## Data Format

Place `.npy` files in `data/` directory with shape `(N, H, W, C)`:
- Training: `data/100/window_2003.npy`, etc.
- Testing: `data/100/window_2023.npy`

## Output Structure

```
models/<MODEL>/output/
├── weights/        # Model checkpoints
├── logs/           # Training logs
└── images/         # Sample predictions
```
