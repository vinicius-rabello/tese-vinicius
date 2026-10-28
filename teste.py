# validate_sampler.py
from collections import defaultdict
from torch.utils.data import random_split, DataLoader
from datasets.sr_tiny_dataset_srno_sampler import SRTinyDataset
from datasets.samplers import SameSizeBatchSampler  # ajuste o import

def validate(loader, name, n_batches=50):
    shapes_seen = set()
    errors = 0

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        inp = batch['inp']  # (B, C, H, W)
        B, C, H, W = inp.shape

        # Checa se todos do batch têm o mesmo shape (já garantido pelo stack, mas confirma)
        shapes_seen.add((H, W))

        # Checa que gt e coord também são consistentes
        assert batch['gt'].shape[0]    == B, f"gt batch size mismatch"
        assert batch['coord'].shape[0] == B, f"coord batch size mismatch"

    print(f"\n[{name}]")
    print(f"  Batches verificados : {min(i+1, n_batches)}")
    print(f"  Shapes LR distintos : {len(shapes_seen)}  ← deve ser > 1 para confirmar variação")
    print(f"  Shapes vistos       : {sorted(shapes_seen)[:10]} {'...' if len(shapes_seen) > 10 else ''}")
    print(f"  Erros de shape      : {errors}  ← deve ser 0")


if __name__ == "__main__":
    import models.sofono.config as config

    dataset    = SRTinyDataset(config.HR_FILES)
    train_size = int((1 - config.VAL_SPLIT) * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_sampler = SameSizeBatchSampler(train_ds, batch_size=config.BATCH_SIZE, drop_last=False)
    val_sampler   = SameSizeBatchSampler(val_ds,   batch_size=config.BATCH_SIZE, drop_last=False)

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              num_workers=config.NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler,
                              num_workers=config.NUM_WORKERS, pin_memory=False)

    validate(train_loader, "TRAIN")
    validate(val_loader,   "VAL")