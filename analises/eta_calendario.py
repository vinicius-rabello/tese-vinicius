from datetime import datetime, timedelta
S = {('srcnn',0):6.3,('srcnn',1):31.9,('dscms',0):20.6,('dscms',1):46.1,('prusr',0):102.7,('prusr',1):128.1,
     ('srno',0):179.4,('srno',1):206.6,('sofono',0):242.9,('sofono',1):276.4}
LOAD = {'srcnn':60,'dscms':60,'prusr':60,'srno':90,'sofono':90}
E=300; A=('srcnn','dscms','prusr','srno','sofono')
def runs(arch, lam, n=1, epochs=E, factor=1.0): return [(arch, S[(arch,lam)]*factor, epochs)]*n
phases = {
 'fase1 (resto: PRUSR, SRNO, SoFoNO; B)': runs('prusr',0,epochs=E-19)+runs('srno',0)+runs('sofono',0)+runs('srcnn',1,4)+runs('srno',1,4),
 'fase2 (C semente 0)': runs('dscms',1)+runs('prusr',1)+runs('sofono',1),
 'fase3 (D: SRCNN, DSC/MS, SRNO)': runs('srcnn',0)+runs('dscms',0)+runs('srno',0),
 'fase4 (A sementes 1-2)': [r for _ in range(2) for a in A for r in runs(a,0)],
 'fase5 (C sementes 1-2)': [r for _ in range(2) for a in A for r in runs(a,1)],
 'fase6 (E, pior caso SoFoNO)': [r for n in (2,3,4) for r in runs('sofono',0,factor=1+0.38*(n-1))]+runs('sofono',0,epochs=2*E),
 'fase7 (regime B, 13)': [r for a in A for r in runs(a,0)]+[r for a in A for r in runs(a,1)]+runs('srcnn',1,3),
}
def windows(start, weekend):
    d = start
    yield start, start.replace(hour=17, minute=0)            # hoje até a máquina desligar
    day = (start + timedelta(days=1)).date()
    while True:
        dt = datetime(day.year, day.month, day.day)
        if dt.weekday() < 5:
            s, e = dt.replace(hour=9, minute=5), dt.replace(hour=16, minute=45)
            if weekend and dt.weekday() == 4:                  # sexta 9h05 -> segunda 8h45, contínuo
                e = (dt + timedelta(days=3)).replace(hour=8, minute=45)
                yield s, e; day = (dt + timedelta(days=3)).date(); continue
            yield s, e
        day += timedelta(days=1)
def simulate(weekend, start):
    w = windows(start, weekend); ws, we = next(w); t = ws; first = True; out = {}
    pending = [(name, list(rs)) for name, rs in phases.items()]
    cur = None; days = 1
    for name, rs in pending:
        for arch, sec, ep in rs:
            done = 0; need_load = not first
            first = False
            while done < ep:
                if need_load: t += timedelta(seconds=LOAD[arch]); need_load = False
                if t + timedelta(seconds=sec) <= we:
                    t += timedelta(seconds=sec); done += 1
                else:
                    ws, we = next(w); t = ws; need_load = True; days += 1
        out[name] = (t, days)
    return out
start = datetime(2026, 10, 1, 16, 1)
for weekend in (False, True):
    print('\n== ' + ('com janela longa toda sexta 17h -> segunda 9h' if weekend else 'só dias úteis, 9h05-16h45'))
    for name, (t, days) in simulate(weekend, start).items():
        print(f"  {name:40s} termina {t:%a %d/%m %H:%M}  (janela nº {days})")
