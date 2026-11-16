"""Print running-window statistics in the form the manuscript quotes them.

The significance tests are computed by the notebook that draws the figure
(``peri_sac.test_pairwise_conditions``, ``test_condition_vs_baseline``, ``apply_fdr_correction``).
This module only summarises that output: which windows survived FDR correction, grouped into
intervals, the direction and size of each effect, and the extremes of the mean curves.

Windows are grouped as in ``03_figures_2_4``'s REPORTED VALUES cells - a gap of more than
two steps starts a new interval - so the intervals compare with those already in the text.
Intervals are window centres, as the manuscript quotes them ("275–425 ms").
"""

import numpy as np
import pandas as pd

from peri_sac import calculate_baseline_metrics_from_trials

CONDS = ('consistent', 'inconsistent')
COND_LABELS = {'consistent': 'congruent', 'inconsistent': 'incongruent'}


def sig_clusters(rows, alpha=0.05, step=25, max_gap_steps=2):
    """Group the FDR-significant windows of one test into intervals.

    rows : list of dicts with 'window_center' and 'p_value_corrected'.
    Returns dicts with start / end (window centres, ms), n_windows, p_min / p_max (FDR p over the
    interval's significant windows) and centres.
    """
    df = pd.DataFrame(rows)
    if df.empty or 'p_value_corrected' not in df:
        return []
    sig = df[df['p_value_corrected'] < alpha].sort_values('window_center')
    if sig.empty:
        return []
    groups, current = [], [sig.iloc[0]]
    for _, r in sig.iloc[1:].iterrows():
        if r['window_center'] - current[-1]['window_center'] > step * max_gap_steps:
            groups.append(current)
            current = []
        current.append(r)
    groups.append(current)
    out = []
    for g in groups:
        g = pd.DataFrame(g)
        out.append({
            'start': float(g['window_center'].min()), 'end': float(g['window_center'].max()),
            'n_windows': len(g),
            'p_min': float(g['p_value_corrected'].min()), 'p_max': float(g['p_value_corrected'].max()),
            'centres': [float(c) for c in g['window_center']],
        })
    return out


def subject_baselines(data, filters, metric, baseline_range=(-500, 0),
                      reference='time_from_sound_to_sacc_start'):
    """Per-subject pre-sound value of `metric` per condition: what the vs-baseline test compares
    each window with. filters = {'consistent': mask, 'inconsistent': mask}."""
    out = {}
    for cond in CONDS:
        bd = calculate_baseline_metrics_from_trials(data[filters[cond]], baseline_range, reference)
        out[cond] = {s: v.get(metric, np.nan) for s, v in bd.items()}
    return out


def _subject_means(res, metric, centres):
    return res[res['window_center'].isin(centres)].groupby('id_subject')[metric].mean()


def _fmt_ms(c):
    return f"{c['start']:.0f} ms" if c['start'] == c['end'] else f"{c['start']:.0f}–{c['end']:.0f} ms"


def _fmt_p(c):
    f = lambda p: '< 0.001' if p < 0.001 else f'{p:.3f}'
    if c['p_max'] < 0.001:
        return 'FDR p < 0.001'
    if c['p_min'] == c['p_max']:
        return f"FDR p = {f(c['p_min'])}"
    return f"FDR p {f(c['p_min'])} to {f(c['p_max'])}"


def report(sig, metric, curves, title, baselines=None, alpha=0.05, step=25,
           extremes=(('min', (0, 450)), ('max', (0, 450))), percent=None, report_range=None):
    """Print one metric's statistics for one experiment and alignment; return them as a frame.

    sig       : FDR-corrected dict for this metric ('<metric>_cond0_vs_cond1',
                '<metric>_cond0_vs_baseline', '<metric>_cond1_vs_baseline');
                cond0 = congruent, cond1 = incongruent.
    curves    : {'consistent': results_df, 'inconsistent': results_df} (per-subject windows).
    baselines : reference for "above / below baseline" and its size, per condition:
                {cond: {subject: value}} (exact, see subject_baselines), {cond: float}, one float,
                or None = 0 (right for the % change count metrics).
    extremes  : (('min' | 'max', (t0, t1)), ...) of each condition's mean curve.
    percent   : format values as % change (default: True for count metrics).
    report_range : (t0, t1) - only windows centred in this range are reported (e.g. up to the
                median RT, so that nothing after the response is quoted). The FDR correction is
                not redone: p values stay those of the full family, as drawn in the figure.
    """
    if report_range is not None:
        keep = lambda rows: [r for r in rows if report_range[0] <= r['window_center'] <= report_range[1]]
        sig = {k: keep(v) for k, v in sig.items()}
        curves = {c: df[df['window_center'].between(*report_range)] for c, df in curves.items()}
    if percent is None:
        percent = 'count' in metric
    fv = (lambda v: f'{v:+.1f} %') if percent else (lambda v: f'{v:.3f}')
    fd = (lambda v: f'{v:+.1f} pp') if percent else (lambda v: f'{v:+.3f}')

    def base_of(cond):
        if baselines is None:
            return 0.0
        return baselines[cond] if isinstance(baselines, dict) else baselines

    def base_level(cond):
        b = base_of(cond)
        return float(np.nanmean(list(b.values()))) if isinstance(b, dict) else float(b)

    def delta_vs_base(cond, centres):
        m = _subject_means(curves[cond], metric, centres)
        b = base_of(cond)
        if isinstance(b, dict):
            b = pd.Series(b).reindex(m.index)
        return float((m - b).mean())

    n = curves['consistent']['id_subject'].nunique()
    print(f"\n{'=' * 78}\n{title}   [{metric}, n = {n}]\n{'=' * 78}")
    if baselines is not None:
        print('  pre-sound level:  ' + ',  '.join(f'{COND_LABELS[c]} {base_level(c):.3f}' for c in CONDS))

    rows = []
    tests = [(f'{metric}_cond0_vs_baseline', 'congruent vs baseline', 'consistent'),
             (f'{metric}_cond1_vs_baseline', 'incongruent vs baseline', 'inconsistent'),
             (f'{metric}_cond0_vs_cond1', 'congruent vs incongruent', None)]
    for key, label, cond in tests:
        if key not in sig:
            continue
        cl = sig_clusters(sig[key], alpha, step)
        if not cl:
            p = pd.DataFrame(sig[key])
            pmin = float(p['p_value_corrected'].min()) if 'p_value_corrected' in p else np.nan
            print(f'  {label:26s}  none significant (smallest FDR p = {pmin:.3f})')
            rows.append({'test': label, 'p_min': pmin})
            continue
        for c in cl:
            if cond is None:
                d = float((_subject_means(curves['consistent'], metric, c['centres'])
                           - _subject_means(curves['inconsistent'], metric, c['centres'])).mean())
                direction = 'congruent higher' if d > 0 else 'incongruent higher'
            else:
                d = delta_vs_base(cond, c['centres'])
                direction = 'above baseline' if d > 0 else 'below baseline'
            print(f'  {label:26s}  {_fmt_ms(c):>14s}  {direction:18s}  mean Δ {fd(d):>9s}   {_fmt_p(c)}')
            rows.append({'test': label, **{k: c[k] for k in ('start', 'end', 'n_windows', 'p_min', 'p_max')},
                         'direction': direction, 'delta': d})

    for kind, (t0, t1) in extremes or ():
        for cond in CONDS:
            mc = curves[cond].groupby('window_center')[metric].mean()
            mc = mc[(mc.index >= t0) & (mc.index <= t1)].dropna()
            if mc.empty:
                continue
            t, v = (mc.idxmin(), mc.min()) if kind == 'min' else (mc.idxmax(), mc.max())
            extra = '' if baselines is None else f'   (Δ from pre-sound {fd(v - base_level(cond))})'
            print(f'  {COND_LABELS[cond]:11s} {kind} in {t0:.0f}–{t1:.0f} ms:  t = {t:.0f} ms,  {fv(v)}{extra}')
            rows.append({'test': f'{COND_LABELS[cond]} {kind}', 'start': t, 'end': t, 'value': v})

    out = pd.DataFrame(rows)
    out.insert(0, 'metric', metric)
    out.insert(0, 'title', title)
    return out
