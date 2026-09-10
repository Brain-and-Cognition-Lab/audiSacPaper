"""
Heatmap-style peri-saccadic analyses.

Provides plot_amplitude_time_heatmap(), which shows a chosen running-window
metric as a 2-D heat map: time on x-axis, saccade amplitude on y-axis,
colour on z-axis.  Fully analogous to the line-plot analyses in sections 3-5,
including the same binning methods and vertical reference lines.

Data preparation and plotting are separated:
  - prepare_amplitude_time_heatmap() → dict  (slow; cache-friendly)
  - plot_amplitude_time_heatmap(..., precomputed=dict)  (fast; display only)
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def _make_3_ticks(vmin, vmax):
    """Return [bottom, middle, top] ticks rounded to nice values inside [vmin, vmax].

    Uses ceil for the bottom and floor for the top so ticks never exceed the
    norm's range (important for TwoSlopeNorm colorbars).
    """
    vmid = (vmin + vmax) / 2
    span = abs(vmax - vmin)
    if span == 0:
        return [vmin]
    if max(abs(vmin), abs(vmax)) <= 1:
        lo  = math.ceil(vmin * 100) / 100
        hi  = math.floor(vmax * 100) / 100
        mid = round(vmid, 2)
    else:
        if span >= 50:
            scale = 10
        elif span >= 10:
            scale = 5
        elif span >= 1:
            scale = 1
        elif span >= 0.1:
            scale = 0.1
        else:
            scale = 0.05
        lo  = math.ceil(vmin / scale) * scale
        hi  = math.floor(vmax / scale) * scale
        mid = round(vmid / scale) * scale
    return sorted({lo, mid, hi})


_CBAR_LABELS = {
    'saccade_count_normalized':      'Sac Count',
    'saccade_count_normalized_spec': 'Sac Count',
    'saccade_count':                 'Saccade Count',
    'saccade_rate_hz':               'Saccade Rate (Hz)',
    'rt_median':                     'Median RT (ms)',
    'p_correct_response':            'P(Correct)',
    'p_wrong_response':              'P(Error)',
    'p_omission':                    'P(Omission)',
    'p_correct_choice':              'P(Correct Choice)',
    'p_response_given':              'P(Response)',
}

# Per-metric defaults for vlim and vcenter used when the caller passes vlim=None.
# vlim=None means auto-scale from the data matrix (plain Normalize).
_METRIC_DEFAULTS = {
    'saccade_count_normalized':      {'vlim': 45,      'vcenter': 0},
    'saccade_count_normalized_spec': {'vlim': 45,      'vcenter': 0},
    'p_correct_response':            {'vlim': (0, 1),  'vcenter': 0.5},
    'p_wrong_response':              {'vlim': (0, 1),  'vcenter': 0.5},
    'p_omission':                    {'vlim': (0, 1),  'vcenter': 0.5},
    'p_correct_choice':              {'vlim': (0, 1),  'vcenter': 0.5},
    'p_response_given':              {'vlim': (0, 1),  'vcenter': 0.5},
}


def prepare_amplitude_time_heatmap(
    data,
    analysis_filter,
    amplitude_col,
    time_reference,
    time_range,
    baseline_range,
    consistency=None,
    window_size=100,
    step_size=25,
    # ----- binning -----
    n_bins=12,
    binning_method='quantile_presound',
    per_subject=True,
    presound_threshold=0,
    # ----- metric -----
    metric='saccade_count_normalized',
    # ----- colour scale -----
    vlim=None,
    vcenter=None,
):
    """Compute heatmap matrix and colour-scale parameters without plotting.

    Returns a dict that can be passed directly to plot_amplitude_time_heatmap()
    via its ``precomputed`` argument, or serialised to disk (e.g. pickle) and
    reloaded later to skip recomputation.

    Parameters match those of plot_amplitude_time_heatmap() exactly.

    Returns
    -------
    dict with keys:
        matrix, x_edges, y_edges   — ready for pcolormesh
        vmin, vmax, vcenter        — colour-scale limits
        cbar_label                 — string for the colorbar title
        metric                     — metric name (for reference)
    """
    from binning import assign_quantile_bins_presound, assign_quantile_bins
    from peri_sac import calculate_running_window_metrics

    # ------------------------------------------------------------------
    # 1. Apply filters and assign amplitude bins
    # ------------------------------------------------------------------
    data_filtered = data[analysis_filter].copy()

    if consistency is not None:
        data_filtered = data_filtered[data_filtered['consistent'] == consistency]

    if binning_method == 'quantile_presound':
        data_binned = assign_quantile_bins_presound(
            data_filtered,
            n_bins=n_bins,
            value_col=amplitude_col,
            per_subject=per_subject,
            presound_threshold=presound_threshold,
        )
    elif binning_method == 'quantile':
        data_binned = assign_quantile_bins(
            data_filtered,
            n_bins=n_bins,
            value_col=amplitude_col,
            per_subject=per_subject,
        )
    else:
        raise ValueError(
            f"Unknown binning_method: {binning_method!r}. "
            "Use 'quantile_presound' (used by the paper) or 'quantile'."
        )

    data_binned = data_binned.rename(columns={'quantile_bin': '_amp_bin'})

    # ------------------------------------------------------------------
    # 2. Compute running window metrics for each amplitude bin
    # ------------------------------------------------------------------
    bin_ids = sorted(data_binned['_amp_bin'].dropna().unique())
    bin_amp_means = {}
    bin_results   = {}

    for bin_id in bin_ids:
        bin_mask = data_binned['_amp_bin'] == bin_id
        res = calculate_running_window_metrics(
            data_binned,
            analysis_filter=bin_mask,
            time_reference=time_reference,
            window_size=window_size,
            step_size=step_size,
            time_range=time_range,
            baseline_range=baseline_range,
        )
        bin_results[bin_id]   = res
        bin_amp_means[bin_id] = data_binned.loc[bin_mask, amplitude_col].median()

    # ------------------------------------------------------------------
    # 3. Build 2-D matrix: rows = amplitude bins, cols = time points
    # ------------------------------------------------------------------
    time_points = np.arange(time_range[0], time_range[1] + step_size, step_size)
    matrix = np.full((len(bin_ids), len(time_points)), np.nan)

    for row_idx, bin_id in enumerate(bin_ids):
        res = bin_results[bin_id]
        if res is None or len(res) == 0 or metric not in res.columns:
            continue
        group = res.groupby('window_center')[metric].mean()
        for col_idx, t in enumerate(time_points):
            if t in group.index:
                matrix[row_idx, col_idx] = group[t]

    # ------------------------------------------------------------------
    # 4. Resolve colour scale
    # ------------------------------------------------------------------
    _defaults        = _METRIC_DEFAULTS.get(metric, {})
    resolved_vlim    = vlim    if vlim    is not None else _defaults.get('vlim',    None)
    resolved_vcenter = vcenter if vcenter is not None else _defaults.get('vcenter', None)

    if resolved_vlim is None:
        finite = matrix[np.isfinite(matrix)]
        vmin = float(finite.min()) if len(finite) else 0.0
        vmax = float(finite.max()) if len(finite) else 1.0
    else:
        if isinstance(resolved_vlim, tuple):
            vmin, vmax = resolved_vlim
        else:
            vmin, vmax = -resolved_vlim, resolved_vlim

    # ------------------------------------------------------------------
    # 5. Build pcolormesh edges
    # ------------------------------------------------------------------
    y_values = np.array([bin_amp_means[b] for b in bin_ids])
    if len(y_values) > 1:
        dy = np.diff(y_values)
        y_edges = np.concatenate([
            [y_values[0] - dy[0] / 2],
            (y_values[:-1] + y_values[1:]) / 2,
            [y_values[-1] + dy[-1] / 2],
        ])
    else:
        y_edges = np.array([y_values[0] - 1, y_values[0] + 1])

    x_edges = np.concatenate([
        [time_points[0] - step_size / 2],
        (time_points[:-1] + time_points[1:]) / 2,
        [time_points[-1] + step_size / 2],
    ])

    return {
        'matrix':     matrix,
        'x_edges':    x_edges,
        'y_edges':    y_edges,
        'vmin':       vmin,
        'vmax':       vmax,
        'vcenter':    resolved_vcenter,
        'cbar_label': _CBAR_LABELS.get(metric, metric),
        'metric':     metric,
    }


def plot_amplitude_time_heatmap(
    data=None,
    analysis_filter=None,
    amplitude_col=None,
    time_reference=None,
    time_range=None,
    baseline_range=None,
    consistency=None,
    window_size=100,
    step_size=25,
    # ----- binning -----
    n_bins=12,
    binning_method='quantile_presound',
    per_subject=True,
    presound_threshold=0,
    # ----- metric shown as colour -----
    metric='saccade_count_normalized',
    # ----- colour scale -----
    vlim=None,
    vcenter=None,
    cmap='RdBu_r',
    # ----- precomputed data (skips all computation) -----
    precomputed=None,
    # ----- layout -----
    figsize=(14, 5),
    title=None,
    amplitude_unit_label='px',
    # ----- reference lines (same as line-plot analyses) -----
    median_rt_line=None,
    rt_q1=None,
    rt_q3=None,
    ax=None,
    cax=None,
):
    """
    Plot a 2-D heatmap of *metric* over time × amplitude.

    Pass ``precomputed=prepare_amplitude_time_heatmap(...)`` to skip all data
    computation and go straight to rendering — useful when iterating on display
    settings without changing the underlying data or parameters.

    Parameters
    ----------
    precomputed : dict or None
        If given, must be the dict returned by prepare_amplitude_time_heatmap().
        All data-computation arguments (data, analysis_filter, …) are ignored.
    (all other parameters are the same as before)

    Returns
    -------
    fig, ax
    """
    # ------------------------------------------------------------------
    # Data: use precomputed or compute now
    # ------------------------------------------------------------------
    if precomputed is not None:
        p = precomputed
    else:
        if data is None or analysis_filter is None:
            raise ValueError(
                "Either precomputed= or (data, analysis_filter, …) must be provided."
            )
        p = prepare_amplitude_time_heatmap(
            data=data,
            analysis_filter=analysis_filter,
            amplitude_col=amplitude_col,
            time_reference=time_reference,
            time_range=time_range,
            baseline_range=baseline_range,
            consistency=consistency,
            window_size=window_size,
            step_size=step_size,
            n_bins=n_bins,
            binning_method=binning_method,
            per_subject=per_subject,
            presound_threshold=presound_threshold,
            metric=metric,
            vlim=vlim,
            vcenter=vcenter,
        )

    matrix      = p['matrix']
    x_edges     = p['x_edges']
    y_edges     = p['y_edges']
    vmin        = p['vmin']
    vmax        = p['vmax']
    resolved_vcenter = p.get('vcenter')
    # Resolve the label from the current table via the metric recorded in `p`,
    # so a renamed label applies to precomputed/cached data without a recompute.
    _cbar_metric = p.get('metric', metric)
    cbar_label   = _CBAR_LABELS.get(_cbar_metric,
                                    p.get('cbar_label', _cbar_metric))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if resolved_vcenter is not None:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=resolved_vcenter, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    pcm = ax.pcolormesh(x_edges, y_edges, matrix, cmap=cmap, norm=norm, shading='flat')

    if cax is not None:
        cbar = fig.colorbar(pcm, cax=cax)
    else:
        cbar = fig.colorbar(pcm, ax=ax)
    cbar.ax.set_title(cbar_label, pad=8, loc='center')
    cbar.set_ticks(_make_3_ticks(vmin, vmax))

    # Reference lines
    ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    if rt_q1 is not None and rt_q3 is not None:
        ax.axvspan(rt_q1, rt_q3, color='gray', alpha=0.15)
    if median_rt_line is not None:
        ax.axvline(median_rt_line, color='gray', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_xlabel('Time relative to sound onset (ms)')
    ax.set_ylabel(f'Saccade amplitude ({amplitude_unit_label})')
    if title:
        ax.set_title(title)

    return fig, ax
