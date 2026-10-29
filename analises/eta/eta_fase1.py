"""ETA of what is left of fase1 (SoFoNO A seed 0, then block B), weekday windows."""
from datetime import datetime, timedelta
SEC = {'sofono0': 243.3, 'srcnn1': 31.9, 'srno1': 206.6}
def windows(start, weekend):
    yield start, start.replace(hour=16, minute=45)
    day = (start + timedelta(days=1)).date()
    while True:
        dt = datetime(day.year, day.month, day.day)
        if dt.weekday() < 5:
            s, e = dt.replace(hour=9, minute=5), dt.replace(hour=16, minute=45)
            if weekend and dt.weekday() == 4:
                e = (dt + timedelta(days=3)).replace(hour=8, minute=45)
                yield s, e; day = (dt + timedelta(days=3)).date(); continue
            yield s, e
        day += timedelta(days=1)
def sim(runs, weekend, start):
    w = windows(start, weekend); _, we = next(w); t = start; out = []; n = 1
    for name, sec, ep in runs:
        done = 0; t += timedelta(seconds=90)
        while done < ep:
            if t + timedelta(seconds=sec) <= we: t += timedelta(seconds=sec); done += 1
            else: ws, we = next(w); t = ws + timedelta(seconds=90); n += 1
        out.append((name, t, n))
    return out
start = datetime(2026, 10, 5, 10, 37)
sofono_done = 16
for label, so, sc, sn in (('otimista (SoFoNO 90, B-SRCNN 213, B-SRNO 90)', 90, 213, 90),
                          ('central (SoFoNO 150, B-SRCNN 213, B-SRNO 150)', 150, 213, 150),
                          ('pior caso (tudo 300)', 300, 300, 300)):
    runs = [('A_sofono', SEC['sofono0'], so - sofono_done)] + [('B_srcnn', SEC['srcnn1'], sc)] * 4 + [('B_srno', SEC['srno1'], sn)] * 4
    gh = sum(s * e for _, s, e in runs) / 3600
    for wk in (False, True):
        r = sim(runs, wk, start)
        print(f"{label:48s} {gh:5.1f} GPU-h | {'+fds' if wk else 'úteis'}: SoFoNO {r[0][1]:%a %d/%m %H:%M}, fim B-SRCNN {r[4][1]:%a %d/%m}, fim fase1 {r[-1][1]:%a %d/%m %H:%M} ({r[-1][2]} janelas)")
