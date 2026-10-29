"""Calendar ETA of the remaining queues (weekday windows 09:05-16:45, holidays off)."""
from datetime import datetime, timedelta, date
HOLIDAYS = {date(2026, 11, 2), date(2026, 11, 20), date(2026, 12, 25), date(2027, 1, 1)}
def windows(start):
    yield start, start.replace(hour=16, minute=45)
    d = start.date() + timedelta(days=1)
    while True:
        if d.weekday() < 5 and d not in HOLIDAYS:
            yield datetime(d.year, d.month, d.day, 9, 5), datetime(d.year, d.month, d.day, 16, 45)
        d += timedelta(days=1)
def run(plan, start):
    w = windows(start); ws, we = next(w); t = start; out = []
    for name, hours in plan:
        left = hours * 3600
        while left > 0:
            avail = (we - t).total_seconds()
            if avail >= left:
                t += timedelta(seconds=left); left = 0
            else:
                left -= max(avail, 0); ws, we = next(w); t = ws + timedelta(minutes=2)
        out.append((name, t))
    return out
start = datetime(2026, 10, 29, 9, 35)
for label, sofono_h in (('SoFoNO para em ~20-40 épocas (central)', 2.3), ('SoFoNO até 300 épocas (pior caso)', 14.6)):
    plan = [('fase2 (C_sofono)', sofono_h), ('fase3', 6.5), ('fase4', 47.5), ('fase5', 66.1), ('fase6', 85.5), ('fase7', 61.9)]
    print(label)
    for n, t in run(plan, start):
        print(f"  {n:18s} termina {t:%a %d/%m %H:%M}")
