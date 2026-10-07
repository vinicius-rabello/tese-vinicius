"""A priori rule for choosing lambda (DECISIONS.md, "Selection rules" and the
2026-10-07 amendment), applied on the validation set, never on the test set.

For one sweep (one architecture, lambda = 0 from block A seed 0 and the
lambda > 0 runs of block B):
  1. candidates: lambda > 0 with validation MSE <= 1.10 x the MSE of lambda = 0;
  2. among them, the lowest eps_w_10_40 (eps weighted by target power over the
     shells 10 <= k < 40, validation);
  3. every candidate whose eps_w_10_40 is within 2% of that lowest value ties
     with it, and the smallest lambda among the tied ones is chosen;
  4. no candidate -> no choice: the decision goes back to the user.
The original criterion, eps_hf (plain mean of eps(k) over k >= 10), is applied
the same way as a sensitivity analysis only: its choice is reported, never used.
SRCNN and SRNO sweeps: the same lambda -> it holds for all architectures;
different -> per family (SRCNN's for SRCNN, DSC/MS, PRUSR; SRNO's for SRNO,
SoFoNO).
"""

MSE_TOLERANCE = 0.10
EPS_HF_TIE = 0.02          # tie tolerance (name kept from the original rule)
FAMILIES = {'cnn': ('srcnn', 'dscms', 'prusr'), 'op': ('srno', 'sofono')}
FAMILY_OF_SWEEP = {'srcnn': 'cnn', 'srno': 'op'}

TABLE_COLS = ['sweep_arch', 'run_id', 'lambda', 'mse_val', 'mse_val_ratio', 'mse_within_10pct',
              'eps_w_10_40_val', 'eps_w_vs_lowest', 'candidate', 'tied_with_lowest', 'chosen',
              'eps_hf_val (info)', 'chosen_by_eps_hf (info)', 'rule', 'rule_eps_hf (info)']


def _apply(rows, key):
    """Steps 2-4 on the candidate rows with criterion `key`.
    Returns (lowest, tied rows, chosen lambda or None)."""
    candidates = [r for r in rows if r['candidate']]
    lowest = min((r[key] for r in candidates), default=None)
    tied = [r for r in candidates if r[key] <= lowest * (1 + EPS_HF_TIE)] if candidates else []
    return lowest, tied, (min(r['lambda'] for r in tied) if tied else None)


def _explain(rows, key, name, lowest, tied, chosen):
    if chosen is None:
        return f"no lambda > 0 has validation MSE within {MSE_TOLERANCE:.0%} of lambda = 0: no choice"
    return (f"candidates (MSE <= {1 + MSE_TOLERANCE:g} x lambda=0): "
            + ", ".join(f"{r['lambda']:g}" for r in rows if r['candidate'])
            + f"; lowest {name} {lowest:.5g}; within {EPS_HF_TIE:.0%}: "
            + ", ".join(f"{r['lambda']:g}" for r in tied) + f"; chosen (smallest): {chosen:g}")


def select_lambda(arch, runs):
    """runs: [{'run_id', 'lambda', 'mse_val', 'eps_w_val'[, 'eps_hf_val']}], one
    with lambda = 0. Returns (table rows, chosen lambda or None, explanation)."""
    base = [r for r in runs if r['lambda'] == 0]
    if len(base) != 1:
        raise ValueError(f"{arch}: need exactly one lambda = 0 run with validation metrics, got {len(base)}")
    mse0 = base[0]['mse_val']
    rows = []
    for r in sorted(runs, key=lambda r: r['lambda']):
        ratio = r['mse_val'] / mse0
        rows.append({'sweep_arch': arch, 'run_id': r['run_id'], 'lambda': r['lambda'], 'mse_val': r['mse_val'],
                     'mse_val_ratio': ratio, 'mse_within_10pct': ratio <= 1 + MSE_TOLERANCE,
                     'eps_w_10_40_val': r['eps_w_val'], 'eps_hf_val (info)': r.get('eps_hf_val', ''),
                     'candidate': r['lambda'] > 0 and ratio <= 1 + MSE_TOLERANCE})
    lowest, tied, chosen = _apply(rows, 'eps_w_10_40_val')
    why = _explain(rows, 'eps_w_10_40_val', 'eps_w_10_40', lowest, tied, chosen)
    # Sensitivity only: what the original eps_hf criterion would choose.
    if all(r['eps_hf_val (info)'] != '' for r in rows):
        hf_low, hf_tied, hf_chosen = _apply(rows, 'eps_hf_val (info)')
        why_hf = _explain(rows, 'eps_hf_val (info)', 'eps_hf', hf_low, hf_tied, hf_chosen)
    else:
        hf_chosen, why_hf = None, 'eps_hf not available'
    for r in rows:
        r['eps_w_vs_lowest'] = r['eps_w_10_40_val'] / lowest - 1 if lowest else ''
        r['tied_with_lowest'] = r in tied
        r['chosen'] = r['lambda'] == chosen
        r['chosen_by_eps_hf (info)'] = hf_chosen is not None and r['lambda'] == hf_chosen
        r['rule'] = why
        r['rule_eps_hf (info)'] = why_hf
    agree = 'same lambda' if hf_chosen == chosen else f'eps_hf would choose {hf_chosen}'
    return rows, chosen, f"{why} [sensitivity, eps_hf: {agree}]"


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
