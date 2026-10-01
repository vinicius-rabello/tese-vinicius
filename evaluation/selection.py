"""A priori rule for choosing lambda (DECISIONS.md, "Selection rules"), applied
on the validation set, never on the test set.

For one sweep (one architecture, lambda = 0 from block A seed 0 and the
lambda > 0 runs of block B):
  1. candidates: lambda > 0 with validation MSE <= 1.10 x the MSE of lambda = 0;
  2. among them, the lowest eps_hf (mean of eps(k) over k >= D0, validation);
  3. every candidate whose eps_hf is within 2% of that lowest value ties with
     it, and the smallest lambda among the tied ones is chosen.
SRCNN and SRNO sweeps: the same lambda -> it holds for all architectures;
different -> per family (SRCNN's for SRCNN, DSC/MS, PRUSR; SRNO's for SRNO,
SoFoNO).
"""

MSE_TOLERANCE = 0.10
EPS_HF_TIE = 0.02
FAMILIES = {'cnn': ('srcnn', 'dscms', 'prusr'), 'op': ('srno', 'sofono')}
FAMILY_OF_SWEEP = {'srcnn': 'cnn', 'srno': 'op'}

TABLE_COLS = ['sweep_arch', 'run_id', 'lambda', 'mse_val', 'mse_val_ratio', 'mse_within_10pct', 'eps_hf_val',
              'eps_hf_vs_lowest', 'candidate', 'tied_with_lowest', 'chosen', 'rule']


def select_lambda(arch, runs):
    """runs: [{'run_id', 'lambda', 'mse_val', 'eps_hf_val'}], one with lambda = 0.
    Returns (table rows, chosen lambda or None, explanation)."""
    base = [r for r in runs if r['lambda'] == 0]
    if len(base) != 1:
        raise ValueError(f"{arch}: need exactly one lambda = 0 run with validation metrics, got {len(base)}")
    mse0 = base[0]['mse_val']
    rows = []
    for r in sorted(runs, key=lambda r: r['lambda']):
        ratio = r['mse_val'] / mse0
        rows.append({'sweep_arch': arch, 'run_id': r['run_id'], 'lambda': r['lambda'], 'mse_val': r['mse_val'],
                     'mse_val_ratio': ratio, 'mse_within_10pct': ratio <= 1 + MSE_TOLERANCE,
                     'eps_hf_val': r['eps_hf_val'],
                     'candidate': r['lambda'] > 0 and ratio <= 1 + MSE_TOLERANCE})
    candidates = [r for r in rows if r['candidate']]
    lowest = min((r['eps_hf_val'] for r in candidates), default=None)
    for r in rows:
        r['eps_hf_vs_lowest'] = r['eps_hf_val'] / lowest - 1 if lowest else ''
        r['tied_with_lowest'] = bool(r['candidate'] and r['eps_hf_val'] <= lowest * (1 + EPS_HF_TIE))
    tied = [r for r in rows if r['tied_with_lowest']]
    chosen = min(r['lambda'] for r in tied) if tied else None
    if chosen is None:
        why = f"no lambda > 0 has validation MSE within {MSE_TOLERANCE:.0%} of lambda = 0: no choice"
    else:
        why = (f"candidates (MSE <= {1 + MSE_TOLERANCE:g} x lambda=0): "
               + ", ".join(f"{r['lambda']:g}" for r in candidates)
               + f"; lowest eps_hf {lowest:.5g}; within {EPS_HF_TIE:.0%}: "
               + ", ".join(f"{r['lambda']:g}" for r in tied) + f"; chosen (smallest): {chosen:g}")
    for r in rows:
        r['chosen'] = r['lambda'] == chosen
        r['rule'] = why
    return rows, chosen, why


def family_lambdas(chosen):
    """chosen: {'srcnn': lambda, 'srno': lambda} -> (per-architecture lambdas,
    make_block_c.py arguments)."""
    if chosen.get('srcnn') is None or chosen.get('srno') is None:
        return None, None
    if chosen['srcnn'] == chosen['srno']:
        lam = chosen['srcnn']
        return {a: lam for f in FAMILIES.values() for a in f}, f"--lambda {lam:g}"
    per_arch = {a: chosen[sweep] for sweep, fam in FAMILY_OF_SWEEP.items() for a in FAMILIES[fam]}
    return per_arch, f"--lambda-cnn {chosen['srcnn']:g} --lambda-op {chosen['srno']:g}"
