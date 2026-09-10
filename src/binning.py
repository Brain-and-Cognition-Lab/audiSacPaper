"""
Quantile binning of a continuous saccade property.

Used to split saccade amplitude into bins before running the peri-saccadic
analyses:

- the amplitude x time heatmaps in ``02_figures_main.ipynb``
  (10 bins, global quantiles)
- the 2 x 3 repeated-measures ANOVAs in ``04_stats_anova.ipynb``
  (3 bins, per-subject quantiles)

``assign_quantile_bins_presound()`` is the one the paper uses: bin edges are
derived from pre-sound saccades only, so the binning cannot be influenced by
the sound-evoked change in saccade amplitude, and are then applied to all
saccades.

Both functions add a ``quantile_bin`` column numbered 1..n_bins.
"""

import pandas as pd
import numpy as np


def assign_quantile_bins(data, n_bins=3, value_col='amplitude_deg', per_subject=True):
    """
    Assign quantile bins based on all observations.

    Parameters
    ----------
    data : pd.DataFrame
        Data with an ``id_subject`` column and ``value_col``.
    n_bins : int
        Number of quantile bins (default: 3).
    value_col : str
        Name of the column to bin (default: 'amplitude_deg').
    per_subject : bool
        If True, compute quantiles within each subject. If False, use global
        quantiles across all subjects (default: True).

    Returns
    -------
    pd.DataFrame
        Copy of ``data`` with an added ``quantile_bin`` column (1..n_bins).
    """

    data_copy = data.copy()
    valid = data_copy[value_col].notna()

    if per_subject:
        data_copy.loc[valid, 'quantile_bin'] = (
            data_copy[valid]
            .groupby('id_subject')[value_col]
            .transform(lambda x: pd.qcut(x, q=n_bins, labels=False, duplicates='drop') + 1)
        )
    else:
        data_copy.loc[valid, 'quantile_bin'] = (
            pd.qcut(data_copy.loc[valid, value_col],
                    q=n_bins, labels=False, duplicates='drop') + 1
        )

    return data_copy


def assign_quantile_bins_presound(data, n_bins=3, value_col='amplitude_deg',
                                  per_subject=True, presound_threshold=0):
    """
    Assign quantile bins using ONLY pre-sound observations to set the edges.

    1. Restrict to saccades starting before sound onset
       (``time_from_sound_to_sacc_start < presound_threshold``).
    2. Compute quantile edges from those observations.
    3. Apply those edges to ALL observations.

    This keeps the bin definition independent of the sound-evoked change in
    the binned property.

    Parameters
    ----------
    data : pd.DataFrame
        Data with ``id_subject``, ``time_from_sound_to_sacc_start`` and
        ``value_col`` columns.
    n_bins : int
        Number of quantile bins (default: 3).
    value_col : str
        Name of the column to bin (default: 'amplitude_deg').
    per_subject : bool
        If True, compute edges within each subject. If False, use global edges
        (default: True).
    presound_threshold : float
        Time in ms; observations with
        ``time_from_sound_to_sacc_start < presound_threshold`` count as
        pre-sound (default: 0, i.e. the saccade started before sound onset).

    Returns
    -------
    pd.DataFrame
        Copy of ``data`` with an added ``quantile_bin`` column (1..n_bins).
        Subjects with too few distinct pre-sound values get NaN.
    """

    data_copy = data.copy()
    valid = data_copy[value_col].notna()

    presound_mask = data_copy['time_from_sound_to_sacc_start'] < presound_threshold
    presound_data = data_copy[valid & presound_mask].copy()

    if len(presound_data) == 0:
        print(f"WARNING: No pre-sound observations found with threshold {presound_threshold}ms!")
        print("Falling back to standard quantile binning.")
        return assign_quantile_bins(data, n_bins=n_bins,
                                    value_col=value_col,
                                    per_subject=per_subject)

    print(f"Calculating quantiles based on {len(presound_data)} pre-sound observations")
    print(f"  (out of {valid.sum()} total valid observations)")

    if per_subject:

        def assign_bins_for_subject(subject_data):
            subject_id = subject_data.name

            subject_presound = presound_data[presound_data['id_subject'] == subject_id]

            if len(subject_presound) == 0:
                # No pre-sound data for this subject - skip binning
                return pd.Series(np.nan, index=subject_data.index)

            quantiles = [i / n_bins for i in range(n_bins + 1)]
            bin_edges = subject_presound[value_col].quantile(quantiles).values

            # Handle duplicate edges
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < n_bins + 1:
                # Not enough unique values - skip this subject
                return pd.Series(np.nan, index=subject_data.index)

            # Apply these edges to ALL data from this subject
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf

            return pd.cut(subject_data[value_col], bins=bin_edges,
                          labels=False, include_lowest=True) + 1

        data_copy.loc[valid, 'quantile_bin'] = (
            data_copy[valid]
            .groupby('id_subject', group_keys=False)
            .apply(assign_bins_for_subject)
            .values
        )

    else:
        quantiles = [i / n_bins for i in range(n_bins + 1)]
        bin_edges = presound_data[value_col].quantile(quantiles).values

        # Handle duplicate edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < n_bins + 1:
            print(f"WARNING: Only {len(bin_edges)-1} unique bin edges found (requested {n_bins})")
            print("This may be due to limited variability in pre-sound data.")

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        print(f"Bin edges (from pre-sound data): {bin_edges}")

        data_copy.loc[valid, 'quantile_bin'] = (
            pd.cut(data_copy.loc[valid, value_col],
                   bins=bin_edges, labels=False, include_lowest=True) + 1
        )

    return data_copy
