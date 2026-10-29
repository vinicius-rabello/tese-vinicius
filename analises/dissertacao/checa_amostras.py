"""Scans every 128x128 sample (2001-2024) for all-zero, constant or NaN fields and
dates each index from the MT axis of the monthly netCDF files (same order as
create_dataset.py). Read only."""
import json, os
import numpy as np
from netCDF4 import Dataset, num2date
ROOT = '/home/lcadame/Vinicius_Thesis/Data/hycom/100'
out = {'zerados': [], 'constantes': [], 'com_nan': [], 'zeros_parciais': [], 'datas_faltantes': {}, 'n_por_ano': {}}
for y in range(2001, 2025):
    dates = []
    for m in range(1, 13):
        p = f'{ROOT}/{y}/hycom_gom_{y}-{m:02d}.nc'
        if os.path.exists(p):
            with Dataset(p) as ds:
                dates += [str(d)[:10] for d in num2date(ds['MT'][:], ds['MT'].units)]
    a = np.load(f'hycom_numpy/hycom_{y}.npy', mmap_mode='r')
    assert len(a) == len(dates), (y, len(a), len(dates))
    out['n_por_ano'][y] = len(a)
    import datetime as dt
    full = [str(dt.date(y, 1, 1) + dt.timedelta(days=i)) for i in range((dt.date(y + 1, 1, 1) - dt.date(y, 1, 1)).days)]
    missing = sorted(set(full) - set(dates))
    if missing:
        out['datas_faltantes'][y] = [missing[0], missing[-1], len(missing)] if len(missing) > 3 else missing
    flat = np.asarray(a).reshape(len(a), -1, 2)
    split = 'treino' if y <= 2019 else ('validacao' if y <= 2021 else 'teste')
    for i in range(len(a)):
        f = flat[i]
        tag = {'ano': y, 'indice_no_ano': i, 'data': dates[i], 'conjunto': split}
        if np.isnan(f).any():
            out['com_nan'].append(tag)
        if (f == 0).all():
            out['zerados'].append(tag)
        elif (f.max(0) == f.min(0)).any():
            out['constantes'].append(tag)
        elif ((f == 0).all(1)).mean() > 0:
            out['zeros_parciais'].append({**tag, 'fracao_px_zero': float(((f == 0).all(1)).mean())})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'amostras_suspeitas.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'n_por_ano'}, indent=1, ensure_ascii=False))
