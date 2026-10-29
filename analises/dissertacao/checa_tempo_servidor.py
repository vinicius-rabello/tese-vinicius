"""Reads only the MT axis of each yearly GOMb0.01 reanalysis file on the HYCOM
THREDDS server and checks that index day*24 + 12 falls at 12:00 UTC."""
import json, os, sys
import netCDF4 as nc, numpy as np
out = {}
for y in range(2001, 2025):
    try:
        ds = nc.Dataset(f"https://tds.hycom.org/thredds/dodsC/GOMb0.01/reanalysis/{y}")
        t = ds['MT']; v = np.asarray(t[:]); d = nc.num2date(v, t.units)
        steps = np.round(np.diff(v) * 24, 3)
        off = [str(d[k]) for k in range(12, len(d), 24) if d[k].hour != 12 or (d[k] - d[0]).days != k // 24][:5]
        out[y] = {'n': len(v), 'inicio': str(d[0]), 'fim': str(d[-1]), 'lacunas': int((steps != 1).sum()),
                  'indices_fora_12h': off, 'experiment': ds.getncattr('experiment')}
        ds.close()
    except Exception as e:
        out[y] = {'erro': repr(e)[:200]}
    print(y, out[y], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tempo_servidor.json'), 'w'), indent=1)
