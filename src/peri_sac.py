"""
Peri-saccadic analysis module: metric calculation and plotting.

This module provides comprehensive functions for analyzing peri-saccadic data:
- Metric calculation from raw trial data (running window analysis)
- Baseline normalization and significance testing
- Line/significance-bar primitives that the figure notebooks compose into panels

Key Functions:
- calculate_baseline_metrics_from_trials(): Per-subject baseline values
- calculate_running_window_metrics(): Calculate all metrics (count, rate, RT, accuracy, normalized)
- test_condition_vs_baseline() / test_pairwise_conditions(): Wilcoxon / Kruskal-Wallis tests
- apply_fdr_correction(): Benjamini-Hochberg correction across windows
- plot_single_condition() / plot_significance_bars(): Drawing primitives

Available Metrics:
- saccade_count: Raw saccade count
- saccade_rate_hz: Saccade rate in Hz (per-trial average)
- saccade_rate_kernel: Kernel-smoothed saccade rate (τ=50ms, Rolfs et al., 2008)
- saccade_count_normalized: Normalized count (% of baseline) - auto-calculated if baseline_range provided
- saccade_count_normalized_spec: Special normalized count using FIXED baseline
  (time_from_sound_to_sacc_start, response_given==1) - auto-calculated if baseline_range provided
- rt_median: Median reaction time
- amplitude_median: Median saccade amplitude (in degrees)
- duration_median: Median saccade duration (ms) — from sacc_dur column
- latency_median: Median preceding-fixation duration (ms) — from prev_fix_duration column
- velocity_median: Median saccade peak velocity (deg/s) — from peak_velocity column
- p_wrong_response: P(omission or wrong choice)
- p_wrong_choice: P(wrong choice | response given)
- p_omission: P(no response)
- p_correct_response: P(correct response) = 1 - p_wrong_response
- p_correct_choice: P(correct choice | response given) = 1 - p_wrong_choice
- p_response_given: P(response) = 1 - p_omission
- d_prime: SDT sensitivity d' (responded trials only; signal/noise split via sound_direction or sound_pitch)
- criterion_c: SDT response bias c (responded trials only)
- p_response_right: P(response == 'right' | response given) — uses 'response' col if present, else infers from acc + sound_direction
- p_response_left: P(response == 'left' | response given) = 1 - p_response_right
- p_response_high: P(response == 'high' | response given) — pitch tasks only ('up' arrow); NaN elsewhere
- p_response_low: P(response == 'low' | response given) = 1 - p_response_high

Features:
- Significance testing vs baseline (Wilcoxon test) - colored bars below
- Pairwise significance testing between conditions - grey bars at top
- Optional FDR Benjamini-Hochberg correction
- Automatic baseline normalization during metric calculation
- Flexible metric selection

The full-figure helpers of the original module (plot_peri_saccadic,
plot_peri_saccadic_grid, plot_difference, plot_rt_distribution) are not part of
this repository: the paper figures are laid out cell-by-cell in the notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

DEFAULT_METRICS = {
    'saccade_count': {'label': 'Saccade Count', 'ylim': None},
    'saccade_rate_hz': {'label': 'Saccade Rate (Hz)', 'ylim': None},
    'saccade_rate_kernel': {'label': 'Saccade Rate - Kernel (Hz)', 'ylim': None},
    'saccade_count_normalized': {'label': 'Saccade Count (% change)', 'ylim': None},
    'saccade_count_normalized_spec': {'label': 'Saccade count (% change, sound-ref)', 'ylim': None},
    'rt_median': {'label': 'Median RT (ms)', 'ylim': None},
    'amplitude_median': {'label': 'Median Amplitude (deg)', 'ylim': None},
    'duration_median': {'label': 'Median Saccade Duration (ms)', 'ylim': None},
    'latency_median': {'label': 'Median Preceding Fixation (ms)', 'ylim': None},
    'velocity_median': {'label': 'Median Peak Velocity (deg/s)', 'ylim': None},
    'p_wrong_response': {'label': 'P(Error)', 'ylim': (0, 1)},
    'p_wrong_choice': {'label': 'P(Wrong Choice)', 'ylim': (0, 1)},
    'p_omission': {'label': 'P(Omission)', 'ylim': (0, 1)},
    'p_correct_response': {'label': 'P(Correct)', 'ylim': (0, 1)},
    'p_correct_choice': {'label': 'P(Correct Choice)', 'ylim': (0, 1)},
    'p_response_given': {'label': 'P(Response)', 'ylim': (0, 1)},
    'd_prime': {'label': "d'", 'ylim': None},
    'criterion_c': {'label': 'Criterion c', 'ylim': None},
    'p_response_right': {'label': 'P(Response Right)', 'ylim': (0, 1)},
    'p_response_left': {'label': 'P(Response Left)', 'ylim': (0, 1)},
    'p_response_high': {'label': 'P(Response High)', 'ylim': (0, 1)},
    'p_response_low': {'label': 'P(Response Low)', 'ylim': (0, 1)},
}

# Color palettes for different numbers of conditions
COLOR_PALETTES = {
    1: ['#2E86AB'],  # Blue
    2: ['#06A77D', '#F77F00'],  # Green, Orange
    3: ['#2E4057', '#048A81', '#F18F01'],  # Dark blue, teal, orange
    4: ['#06A77D', '#F77F00', '#D62828', '#9B59B6']  # Green, Orange, Red, Purple
}

# COLOR_PALETTES = {
#     1: ['#2E86AB'],  # Blue
#     2: ['#3CAF97', '#962516'],  # Green, Orange
#     3: ['#2E4057', '#048A81', '#F18F01'],  # Dark blue, teal, orange
#     4: ['#06A77D', '#F77F00', '#D62828', '#9B59B6']  # Green, Orange, Red, Purple
# }

# # Test all three options
# PALETTES = {
#     'Option 1: Teal-Blue vs Orange': (
#         ['#7fcdbb', '#41b6c4', '#1d91c0'],
#         ['#fdae6b', '#fd8d3c', '#e6550d']
#     ),
#     'Option 2: Blue-Green vs Warm Orange': (
#         ['#a1dab4', '#41b6c4', '#225ea8'],
#         ['#feb24c', '#fd8d3c', '#d7301f']
#     ),
#     'Option 3: Cyan-Green vs Amber': (
#         ['#66c2a4', '#2ca25f', '#006d2c'],
#         ['#ffeda0', '#fc4e2a', '#bd0026']
#     )
# }

# Grey shades for pairwise comparisons (darkest to lightest)
GREY_SHADES = ['#2D2D2D', '#4D4D4D', '#6D6D6D', '#8D8D8D', '#ADADAD', '#CDCDCD']


# ============================================================================
# HELPER FUNCTIONS - BASELINE CALCULATION
# ============================================================================

def calculate_baseline_metrics_from_trials(data, baseline_range, time_reference='time_to_sacc_start',
                                            response_filter=False, sdt_column=None,
                                            sdt_min_correction_n=0):
    """
    Calculate baseline metrics from raw trial data within a specified time range.

    Parameters:
    -----------
    data : pd.DataFrame
        Long format data (already filtered by condition if needed)
    baseline_range : tuple
        (min_time, max_time) for baseline period in ms
    time_reference : str
        Time reference column (e.g., 'time_to_sacc_start')
    response_filter : bool
        If True, only include trials where response_given == 1
    sdt_column : str or None
        Column used to split signal vs. noise for SDT baseline metrics.
        Should match what is passed to calculate_running_window_metrics.
        If None, d_prime and criterion_c baseline values are NaN.

    Returns:
    --------
    dict : {subject_id: {metric_name: value}}
        Dictionary containing baseline values for each subject and metric:
        - saccade_count: Number of unique saccades in baseline period
        - saccade_rate_hz: Saccade rate in Hz (per-trial average)
        - saccade_rate_kernel: NaN (kernel rate has no single baseline value)
        - saccade_count_normalized: Always 1.0 (baseline normalized to itself = 100%)
        - p_omission: Proportion of trials with no response
        - p_wrong_choice: Proportion of wrong responses among responses given
        - p_wrong_response: Proportion with omission or wrong choice
        - rt_median: Median reaction time for responded trials
        - d_prime: SDT sensitivity in baseline period
        - criterion_c: SDT response bias in baseline period
    """
    min_time, max_time = baseline_range
    baseline_dict = {}

    # Resolve SDT positive value once.
    # sdt_column comes from the caller (e.g. config['sdt_column']); None → SDT metrics are NaN.
    if sdt_column is not None:
        sdt_positive = _SDT_POSITIVE_VALUES.get(sdt_column)
        if sdt_positive is None:
            raise ValueError(
                f"sdt_column='{sdt_column}' has no known positive value. "
                f"Add it to _SDT_POSITIVE_VALUES."
            )
    else:
        sdt_positive = None

    for subject in data['id_subject'].unique():
        subj_data = data[data['id_subject'] == subject]

        # Get all trials/sounds that fall within the baseline time range
        baseline_data = subj_data[
            (subj_data[time_reference] >= min_time) &
            (subj_data[time_reference] < max_time)
        ]

        baseline_dict[subject] = {}

        if len(baseline_data) > 0:
            # Calculate metrics for this baseline window
            n_sounds = len(baseline_data)
            n_responded = (baseline_data['response_given'] == 1).sum()
            n_omissions = (baseline_data['response_given'] == 0).sum()
            n_wrong_choice = ((baseline_data['response_given'] == 1) & (baseline_data['acc'] == 0)).sum()

            # Saccade count
            n_saccades = baseline_data['id_sacc'].nunique()

            # Saccade rate (Hz) - per-trial average rate
            baseline_duration_sec = (max_time - min_time) / 1000.0  # Convert to seconds
            saccades_per_trial = baseline_data.groupby('id_sound')['id_sacc'].nunique()
            trial_rates = saccades_per_trial / baseline_duration_sec
            saccade_rate_hz = trial_rates.mean() if len(saccades_per_trial) > 0 else np.nan

            # Calculate metrics
            p_omission = n_omissions / n_sounds if n_sounds > 0 else np.nan
            p_wrong_choice = n_wrong_choice / n_responded if n_responded > 0 else np.nan
            n_wrong_response = n_omissions + n_wrong_choice
            p_wrong_response = n_wrong_response / n_sounds if n_sounds > 0 else np.nan

            responded_data = baseline_data[baseline_data['response_given'] == 1]
            rt_median = responded_data['rt'].median() if len(responded_data) > 0 else np.nan

            # Calculate inverse metrics
            p_correct_response = 1 - p_wrong_response if not np.isnan(p_wrong_response) else np.nan
            p_correct_choice = 1 - p_wrong_choice if not np.isnan(p_wrong_choice) else np.nan
            p_response_given = 1 - p_omission if not np.isnan(p_omission) else np.nan

            # SDT metrics (responded trials only)
            sdt = _compute_sdt_metrics(responded_data, sdt_column, sdt_positive,
                                       min_correction_n=sdt_min_correction_n) \
                if sdt_column is not None else {'d_prime': np.nan, 'criterion_c': np.nan}

            # Response direction probabilities (responded trials only)
            resp_dir = _compute_response_direction_probs(responded_data)
            resp_pitch = _compute_response_pitch_probs(responded_data)

            # Store baseline metrics
            baseline_dict[subject]['saccade_count'] = n_saccades
            baseline_dict[subject]['saccade_rate_hz'] = saccade_rate_hz
            baseline_dict[subject]['saccade_rate_kernel'] = np.nan  # Kernel rate is time-dependent, no single baseline value
            baseline_dict[subject]['saccade_count_normalized'] = 1.0  # Normalized baseline is always 1.0 (100%)
            baseline_dict[subject]['saccade_count_normalized_spec'] = 1.0  # Special normalized baseline is always 1.0 (100%)
            baseline_dict[subject]['p_omission'] = p_omission
            baseline_dict[subject]['p_wrong_choice'] = p_wrong_choice
            baseline_dict[subject]['p_wrong_response'] = p_wrong_response
            baseline_dict[subject]['p_correct_response'] = p_correct_response
            baseline_dict[subject]['p_correct_choice'] = p_correct_choice
            baseline_dict[subject]['p_response_given'] = p_response_given
            baseline_dict[subject]['rt_median'] = rt_median
            baseline_dict[subject]['d_prime'] = sdt['d_prime']
            baseline_dict[subject]['criterion_c'] = sdt['criterion_c']
            baseline_dict[subject]['p_response_right'] = resp_dir['p_response_right']
            baseline_dict[subject]['p_response_left'] = resp_dir['p_response_left']
            baseline_dict[subject]['p_response_high'] = resp_pitch['p_response_high']
            baseline_dict[subject]['p_response_low'] = resp_pitch['p_response_low']

            # Saccade property medians
            baseline_dict[subject]['amplitude_median'] = (
                baseline_data['amplitude_deg'].median()
                if 'amplitude_deg' in baseline_data.columns else np.nan
            )
            baseline_dict[subject]['duration_median'] = (
                baseline_data['sacc_dur'].median()
                if 'sacc_dur' in baseline_data.columns else np.nan
            )
            baseline_dict[subject]['latency_median'] = (
                baseline_data['prev_fix_duration'].median()
                if 'prev_fix_duration' in baseline_data.columns else np.nan
            )
            baseline_dict[subject]['velocity_median'] = (
                baseline_data['peak_velocity'].median()
                if 'peak_velocity' in baseline_data.columns else np.nan
            )
        else:
            # No data in baseline range - set to NaN
            baseline_dict[subject]['saccade_count'] = np.nan
            baseline_dict[subject]['saccade_rate_hz'] = np.nan
            baseline_dict[subject]['saccade_rate_kernel'] = np.nan
            baseline_dict[subject]['saccade_count_normalized'] = np.nan
            baseline_dict[subject]['saccade_count_normalized_spec'] = np.nan
            baseline_dict[subject]['p_omission'] = np.nan
            baseline_dict[subject]['p_wrong_choice'] = np.nan
            baseline_dict[subject]['p_wrong_response'] = np.nan
            baseline_dict[subject]['p_correct_response'] = np.nan
            baseline_dict[subject]['p_correct_choice'] = np.nan
            baseline_dict[subject]['p_response_given'] = np.nan
            baseline_dict[subject]['rt_median'] = np.nan
            baseline_dict[subject]['d_prime'] = np.nan
            baseline_dict[subject]['criterion_c'] = np.nan
            baseline_dict[subject]['p_response_right'] = np.nan
            baseline_dict[subject]['p_response_left'] = np.nan
            baseline_dict[subject]['p_response_high'] = np.nan
            baseline_dict[subject]['p_response_low'] = np.nan
            baseline_dict[subject]['amplitude_median'] = np.nan
            baseline_dict[subject]['duration_median'] = np.nan
            baseline_dict[subject]['latency_median'] = np.nan
            baseline_dict[subject]['velocity_median'] = np.nan

    return baseline_dict


# ============================================================================
# HELPER FUNCTIONS - DATA NORMALIZATION
# ============================================================================

def normalize_by_baseline(results_df, baseline_dict, metric_column='saccade_count',
                          baseline_range=None, window_size=None, suffix='_normalized'):
    """
    Normalize a metric by its baseline value for each subject.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe with window_center and metric columns
    baseline_dict : dict
        {subject: {metric: value}} baseline values from calculate_baseline_metrics_from_trials
    metric_column : str
        Column name to normalize
    baseline_range : tuple or None
        (min_time, max_time) for baseline period in ms. Required for saccade_count normalization.
    window_size : float or None
        Window size in ms. Required for saccade_count normalization.
    suffix : str
        Suffix to append to metric_column for output column name (default: '_normalized')

    Returns:
    --------
    pd.DataFrame : Copy of results_df with added normalized column
    """
    normalized_dfs = []

    # Calculate normalization factor for saccade_count to account for different time durations
    normalization_factor = 1.0
    if metric_column == 'saccade_count' and baseline_range is not None and window_size is not None:
        baseline_duration = baseline_range[1] - baseline_range[0]  # Duration in ms
        normalization_factor = window_size / baseline_duration

    for subject in results_df['id_subject'].unique():
        subj_data = results_df[results_df['id_subject'] == subject].copy()
        baseline_value = baseline_dict.get(subject, {}).get(metric_column, 0)

        # Scale baseline to match window duration
        scaled_baseline = baseline_value * normalization_factor

        if scaled_baseline > 0:
            # Calculate percent change from baseline: ((value - baseline) / baseline) * 100
            subj_data[f'{metric_column}{suffix}'] = ((subj_data[metric_column] - scaled_baseline) / scaled_baseline) * 100
        else:
            subj_data[f'{metric_column}{suffix}'] = np.nan

        normalized_dfs.append(subj_data)

    return pd.concat(normalized_dfs, ignore_index=True)


# ============================================================================
# HELPER FUNCTIONS - SIGNAL DETECTION THEORY
# ============================================================================

# Maps each SDT column to its conventional "positive" (signal) value.
# 'sound_direction': right sounds are the signal; 'sound_pitch': high-pitched sounds are the signal.
_SDT_POSITIVE_VALUES = {
    'sound_direction': 'right',
    'sound_pitch': 'high',
}


def _compute_sdt_metrics(responded_data, sdt_column, positive_value, min_correction_n=0):
    """
    Compute d' and criterion c from responded trials using signal detection theory.

    Parameters
    ----------
    responded_data : pd.DataFrame
        Trials where a response was given (response_given == 1).
    sdt_column : str
        Column that distinguishes signal ('positive_value') from noise trials.
    positive_value : str
        The value in sdt_column that counts as the "signal" / "positive" category
        (e.g. 'right' for direction tasks, 'high' for pitch tasks).
    min_correction_n : int
        When > 0, the Macmillan-Creelman extreme-proportion correction uses
        max(actual_n, min_correction_n) instead of actual_n.  This prevents
        small windows (few trials) from pulling d' down via an overly large
        correction term (e.g. 1 - 1/(2*2) = 0.75 instead of ~0.995).
        Set to the same value used in calculate_running_window_metrics so that
        the window and baseline d' estimates are on the same scale.
        Default 0 = use actual n (original behaviour).

    Returns
    -------
    dict with keys 'd_prime' and 'criterion_c'.
    """
    if sdt_column not in responded_data.columns or len(responded_data) == 0:
        return {'d_prime': np.nan, 'criterion_c': np.nan}

    valid = responded_data.dropna(subset=[sdt_column])
    if len(valid) == 0:
        return {'d_prime': np.nan, 'criterion_c': np.nan}

    signal_mask = valid[sdt_column] == positive_value
    signal_data = valid[signal_mask]
    noise_data = valid[~signal_mask]

    n_signal = len(signal_data)
    n_noise = len(noise_data)

    if n_signal == 0 or n_noise == 0:
        return {'d_prime': np.nan, 'criterion_c': np.nan}

    # Use the 'response' column when its values match the SDT positive value
    # (e.g. 'left'/'right' for direction tasks). For pitch experiments the response
    # column holds button-press directions ('left'/'right'), not pitch labels
    # ('high'/'low'), so positive_value would never appear — fall back to acc.
    response_col_usable = (
        'response' in valid.columns
        and positive_value in set(valid['response'].dropna().unique())
    )
    if response_col_usable:
        n_hits = (signal_data['response'] == positive_value).sum()
        n_false_alarms = (noise_data['response'] == positive_value).sum()
    else:
        # acc == 1 on signal trial  → correctly identified signal → hit
        # acc == 0 on noise trial   → incorrectly identified noise → false alarm
        n_hits = (signal_data['acc'] == 1).sum()
        n_false_alarms = (noise_data['acc'] == 0).sum()

    H = n_hits / n_signal
    F = n_false_alarms / n_noise

    # Correction for proportions of exactly 0 or 1 (Macmillan & Creelman).
    # Use max(actual_n, min_correction_n) so that small windows (few trials)
    # don't receive an overly large correction that pulls d' down artificially.
    eff_n_signal = max(n_signal, min_correction_n) if min_correction_n > 0 else n_signal
    eff_n_noise  = max(n_noise,  min_correction_n) if min_correction_n > 0 else n_noise

    if H == 0:
        H = 1 / (2 * eff_n_signal)
    elif H == 1:
        H = 1 - 1 / (2 * eff_n_signal)

    if F == 0:
        F = 1 / (2 * eff_n_noise)
    elif F == 1:
        F = 1 - 1 / (2 * eff_n_noise)

    z_H = stats.norm.ppf(H)
    z_F = stats.norm.ppf(F)

    d_prime = z_H - z_F
    criterion_c = -0.5 * (z_H + z_F)

    return {'d_prime': d_prime, 'criterion_c': criterion_c}


def _compute_response_direction_probs(responded_data):
    """
    Compute P(response=='right') and P(response=='left') among responded trials.

    Uses 'response' column directly if present; otherwise infers from acc + sound_direction.
    Returns NaN for both if neither is available (e.g., pitch-only tasks without response col).
    """
    if len(responded_data) == 0:
        return {'p_response_right': np.nan, 'p_response_left': np.nan}

    if 'response' in responded_data.columns:
        valid = responded_data.dropna(subset=['response'])
        if len(valid) == 0:
            return {'p_response_right': np.nan, 'p_response_left': np.nan}
        p_right = (valid['response'] == 'right').sum() / len(valid)
    elif 'sound_direction' in responded_data.columns and 'acc' in responded_data.columns:
        valid = responded_data.dropna(subset=['sound_direction', 'acc'])
        if len(valid) == 0:
            return {'p_response_right': np.nan, 'p_response_left': np.nan}
        n_right = (
            ((valid['sound_direction'] == 'right') & (valid['acc'] == 1)) |
            ((valid['sound_direction'] == 'left') & (valid['acc'] == 0))
        ).sum()
        p_right = n_right / len(valid)
    else:
        return {'p_response_right': np.nan, 'p_response_left': np.nan}

    return {'p_response_right': float(p_right), 'p_response_left': float(1 - p_right)}


def _compute_response_pitch_probs(responded_data):
    """
    Compute P(response=='high') and P(response=='low') among responded trials.

    The pitch tasks (EX_3, EX_8) collected the judgement on the up/down arrow keys;
    preprocessing maps 'ANSWER_UP' -> 'high' and 'ANSWER_DOWN' -> 'low', so the
    'response' column holds 'high'/'low' there. This is the pitch-task counterpart
    of _compute_response_direction_probs: it measures which way the response was
    biased, on that task's own (non-spatial) response axis.

    Returns NaN for both if 'response' is absent or holds no high/low values
    (e.g. a left/right localization task).
    """
    if len(responded_data) == 0 or 'response' not in responded_data.columns:
        return {'p_response_high': np.nan, 'p_response_low': np.nan}

    valid = responded_data[responded_data['response'].isin(['high', 'low'])]
    if len(valid) == 0:
        return {'p_response_high': np.nan, 'p_response_low': np.nan}

    p_high = (valid['response'] == 'high').sum() / len(valid)

    return {'p_response_high': float(p_high), 'p_response_low': float(1 - p_high)}


# ============================================================================
# METRIC CALCULATION
# ============================================================================

def calculate_running_window_metrics(data, analysis_filter=None, time_reference='time_to_sacc_start',
                                     window_size=100, step_size=25, time_range=(-400, 400),
                                     baseline_range=None, sdt_column=None, sdt_min_correction_n=0):
    """
    Calculate metrics in running/sliding windows with overlap.

    This function calculates all peri-saccadic metrics for each subject and window:
    - Saccade metrics: count, rate (Hz), kernel-smoothed rate
    - Behavioral metrics: accuracy, RT, omission rate
    - Normalized metrics: saccade_count_normalized (if baseline_range provided)

    Parameters:
    -----------
    data : pd.DataFrame
        Long format data
    analysis_filter : pd.Series (boolean) or None
        Boolean Series to filter rows
    time_reference : str
        Column name for time reference (e.g., 'time_from_sound_to_sacc_start')
    window_size : int
        Size of each window in ms (default: 100)
    step_size : int
        Step size between window centers in ms (default: 25)
    time_range : tuple
        (min_time, max_time) range to analyze in ms (default: (-400, 400))
    baseline_range : tuple or None
        If provided, also calculates saccade_count_normalized using this baseline period
        Format: (min_time, max_time) in ms (e.g., (-500, 0))
    sdt_column : str or None
        Column used to split trials into signal vs. noise for SDT metrics
        (e.g. 'sound_direction' for direction tasks, 'sound_pitch' for pitch tasks).
        Should be taken from EXPERIMENT_CONFIGS[exp]['sdt_column'].
        If None, d_prime and criterion_c are returned as NaN for all windows.

    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with columns:
        - id_subject: Subject identifier
        - window_center: Center of time window
        - sound_count: Number of sounds in window
        - saccade_count: Number of unique saccades
        - saccade_rate_hz: Saccade rate in Hz (per-trial average)
        - saccade_rate_kernel: Kernel-smoothed saccade rate (τ=50ms, Rolfs et al., 2008)
        - p_wrong_response: P(omission or wrong choice)
        - p_wrong_choice: P(wrong choice | response given)
        - p_omission: P(no response)
        - p_correct_response: P(correct response) = 1 - p_wrong_response
        - p_correct_choice: P(correct choice | response given) = 1 - p_wrong_choice
        - p_response_given: P(response) = 1 - p_omission
        - rt_median: Median RT for trials with responses
        - amplitude_median: Median saccade amplitude in degrees
        - d_prime: SDT sensitivity (responded trials only)
        - criterion_c: SDT response bias (responded trials only)
        - p_response_right: P(response=='right' | response given)
        - p_response_left: P(response=='left' | response given)
        - saccade_count_normalized: Normalized count (% change)
        - saccade_count_normalized_spec: Special normalized count using FIXED baseline
          (time_from_sound_to_sacc_start, response_given==1, baseline_range=(-500, 0))

        Each row represents one subject and one window
    """

    # Apply filter if provided
    if analysis_filter is not None:
        df = data[analysis_filter].copy()
    else:
        df = data.copy()

    # Resolve SDT positive value once (outside the loop).
    # sdt_column comes from the caller (e.g. config['sdt_column']); None → SDT metrics are NaN.
    if sdt_column is not None:
        sdt_positive = _SDT_POSITIVE_VALUES.get(sdt_column)
        if sdt_positive is None:
            raise ValueError(
                f"sdt_column='{sdt_column}' has no known positive value. "
                f"Add it to _SDT_POSITIVE_VALUES."
            )
    else:
        sdt_positive = None

    # Generate window centers
    min_time, max_time = time_range
    window_centers = np.arange(min_time, max_time + step_size, step_size)

    results = []

    for subject in df['id_subject'].unique():
        subj_data = df[df['id_subject'] == subject]

        for center in window_centers:
            # Define window bounds
            window_min = center - window_size / 2
            window_max = center + window_size / 2

            # Get sounds in this window
            window_data = subj_data[
                (subj_data[time_reference] >= window_min) &
                (subj_data[time_reference] < window_max)
            ]

            if len(window_data) == 0:
                continue

            # Calculate metrics
            n_sounds = len(window_data)
            n_responded = (window_data['response_given'] == 1).sum()
            n_omissions = (window_data['response_given'] == 0).sum()
            n_correct = ((window_data['response_given'] == 1) & (window_data['acc'] == 1)).sum()
            n_wrong_choice = ((window_data['response_given'] == 1) & (window_data['acc'] == 0)).sum()

            # Saccade metrics
            # For rate: calculate per-trial rate first, then average
            # For each trial (sound), count saccades and calculate rate, then average across trials
            saccades_per_trial = window_data.groupby('id_sound')['id_sacc'].nunique()
            trial_rates = saccades_per_trial / (window_size / 1000.0)
            saccade_rate_hz = trial_rates.mean()
            # For count: total unique saccades across all trials
            n_saccades = window_data['id_sacc'].nunique()

            # Kernel-based rate (exponential smoothing as in Rolfs et al., 2008)
            # Uses causal exponential kernel: k(t) = (1/τ)·exp(-t/τ) for t >= 0
            tau = 50.0  # ms, decay constant (α = 1/50 ms in the paper)
            peak_shift = 50.0  # ms, shift to account for kernel peak

            # Get unique saccades for this subject with their times (VECTORIZED)
            unique_saccades = subj_data[['id_sacc', time_reference]].drop_duplicates(subset=['id_sacc'])
            saccade_times = unique_saccades[time_reference].values

            # For this window center (adjusted for peak shift), calculate kernel contributions (VECTORIZED)
            adjusted_center = center - peak_shift
            dt = adjusted_center - saccade_times  # Time elapsed since each saccade
            causal_mask = dt >= 0  # Causal: only saccades before this time contribute
            kernel_contributions = (1/tau) * np.exp(-dt[causal_mask] / tau)
            saccade_rate_kernel = kernel_contributions.sum() if causal_mask.any() else 0

            # p(omission) - proportion of sounds with no response
            p_omission = n_omissions / n_sounds if n_sounds > 0 else np.nan

            # p(wrong choice) - proportion of wrong responses among those who responded
            p_wrong_choice = n_wrong_choice / n_responded if n_responded > 0 else np.nan

            # p(wrong response) - proportion with either no response OR wrong response
            n_wrong_response = n_omissions + n_wrong_choice
            p_wrong_response = n_wrong_response / n_sounds if n_sounds > 0 else np.nan

            # Inverse metrics (for "positive" framing of results)
            p_correct_response = 1 - p_wrong_response if not np.isnan(p_wrong_response) else np.nan
            p_correct_choice = 1 - p_wrong_choice if not np.isnan(p_wrong_choice) else np.nan
            p_response_given = 1 - p_omission if not np.isnan(p_omission) else np.nan

            # RT median (only for trials with responses)
            responded_data = window_data[window_data['response_given'] == 1]
            rt_median = responded_data['rt'].median() if len(responded_data) > 0 else np.nan

            # Amplitude median (median amplitude of saccades in this window)
            if 'amplitude_deg' in window_data.columns:
                amplitude_median = window_data['amplitude_deg'].median()
            else:
                amplitude_median = np.nan

            # Duration median (median saccade duration)
            if 'sacc_dur' in window_data.columns:
                duration_median = window_data['sacc_dur'].median()
            else:
                duration_median = np.nan

            # Latency median (median duration of fixation preceding saccade)
            if 'prev_fix_duration' in window_data.columns:
                latency_median = window_data['prev_fix_duration'].median()
            else:
                latency_median = np.nan

            # Velocity median (median peak saccade velocity in deg/s)
            if 'peak_velocity' in window_data.columns:
                velocity_median = window_data['peak_velocity'].median()
            else:
                velocity_median = np.nan

            # SDT metrics (only for responded trials)
            sdt = _compute_sdt_metrics(responded_data, sdt_column, sdt_positive,
                                       min_correction_n=sdt_min_correction_n) \
                if sdt_column is not None else {'d_prime': np.nan, 'criterion_c': np.nan}

            # Response direction probabilities (only for responded trials)
            resp_dir = _compute_response_direction_probs(responded_data)
            resp_pitch = _compute_response_pitch_probs(responded_data)

            results.append({
                'id_subject': subject,
                'window_center': center,
                'sound_count': n_sounds,
                'saccade_count': n_saccades,
                'saccade_rate_hz': saccade_rate_hz,
                'saccade_rate_kernel': saccade_rate_kernel,
                'p_wrong_response': p_wrong_response,
                'p_wrong_choice': p_wrong_choice,
                'p_omission': p_omission,
                'p_correct_response': p_correct_response,
                'p_correct_choice': p_correct_choice,
                'p_response_given': p_response_given,
                'rt_median': rt_median,
                'amplitude_median': amplitude_median,
                'duration_median': duration_median,
                'latency_median': latency_median,
                'velocity_median': velocity_median,
                'd_prime': sdt['d_prime'],
                'criterion_c': sdt['criterion_c'],
                'p_response_right': resp_dir['p_response_right'],
                'p_response_left': resp_dir['p_response_left'],
                'p_response_high': resp_pitch['p_response_high'],
                'p_response_low': resp_pitch['p_response_low'],
            })

    results_df = pd.DataFrame(results)

    # Automatically add normalized saccade count column if baseline_range is provided
    if baseline_range is not None and len(results_df) > 0:
        # Calculate baseline metrics from raw trial data
        baseline_dict = calculate_baseline_metrics_from_trials(
            df, baseline_range, time_reference
        )

        # Add saccade_count_normalized column (does NOT replace saccade_count)
        results_df = normalize_by_baseline(
            results_df, baseline_dict, 'saccade_count',
            baseline_range=baseline_range, window_size=window_size,
            suffix='_normalized'
        )

        # Calculate special normalized metric (saccade_count_normalized_spec)
        # This uses a FIXED baseline: time_from_sound_to_sacc_start, response_given == 1
        if 'response_given' in df.columns:
            # Filter data to only include trials with responses
            df_responded = df[df['response_given'] == 1].copy()

            # Calculate baseline using the FIXED time reference
            baseline_dict_spec = calculate_baseline_metrics_from_trials(
                df_responded, baseline_range=(-500,0), time_reference='time_from_sound_to_sacc_start'
            )

            # Add saccade_count_normalized_spec column using the special baseline
            results_df = normalize_by_baseline(
                results_df, baseline_dict_spec, 'saccade_count',
                baseline_range=(-500,0), window_size=window_size,
                suffix='_normalized_spec'
            )

    return results_df


# ============================================================================
# HELPER FUNCTIONS - STATISTICAL TESTING
# ============================================================================

def test_condition_vs_baseline(window_data, baseline_dict, metric, subject_col='id_subject',
                                metric_suffix='', alpha=0.05, baseline_range=None, window_size=None):
    """
    Test one condition against baseline using Wilcoxon signed-rank test.

    Parameters:
    -----------
    window_data : pd.DataFrame
        Data for one time window
    baseline_dict : dict
        {subject: {metric: value}} baseline values
    metric : str
        Metric name to test
    subject_col : str
        Column name for subject ID
    metric_suffix : str
        Suffix added to metric name in window_data (e.g., '_cond1')
    alpha : float
        Significance level
    baseline_range : tuple or None
        (min_time, max_time) for baseline period in ms. Used to normalize saccade_count.
    window_size : float or None
        Window size in ms. Used to normalize saccade_count.

    Returns:
    --------
    dict : {'p_value': float, 'significant': bool}
    """
    window_values = []
    baseline_values = []

    metric_col = metric + metric_suffix

    # Check if column exists
    if metric_col not in window_data.columns:
        return {'p_value': np.nan, 'significant': False}

    # Calculate normalization factor for saccade_count to account for different time durations
    normalization_factor = 1.0
    if metric == 'saccade_count' and baseline_range is not None and window_size is not None:
        baseline_duration = baseline_range[1] - baseline_range[0]  # Duration in ms
        normalization_factor = window_size / baseline_duration

    for subject in window_data[subject_col].unique():
        window_val = window_data[window_data[subject_col] == subject][metric_col].values
        baseline_val = baseline_dict.get(subject, {}).get(metric, np.nan)

        # Normalize baseline value for saccade_count to match window duration
        if not np.isnan(baseline_val):
            baseline_val = baseline_val * normalization_factor

        if len(window_val) > 0 and not np.isnan(window_val[0]) and not np.isnan(baseline_val):
            window_values.append(window_val[0])
            baseline_values.append(baseline_val)

    if len(window_values) > 2:
        try:
            stat, p_val = stats.wilcoxon(window_values, baseline_values, alternative='two-sided')
            return {'p_value': p_val, 'significant': p_val < alpha}
        except ValueError:
            return {'p_value': np.nan, 'significant': False}
    else:
        return {'p_value': np.nan, 'significant': False}


def test_pairwise_conditions(merged_data, metric, cond1_suffix, cond2_suffix, alpha=0.05):
    """
    Test pairwise difference between two conditions using Wilcoxon signed-rank test.

    Parameters:
    -----------
    merged_data : pd.DataFrame
        Merged data with columns for both conditions
    metric : str
        Base metric name
    cond1_suffix : str
        Suffix for condition 1 (e.g., '_cond1')
    cond2_suffix : str
        Suffix for condition 2 (e.g., '_cond2')
    alpha : float
        Significance level

    Returns:
    --------
    dict : {'p_value': float, 'significant': bool}
    """
    col1 = metric + cond1_suffix
    col2 = metric + cond2_suffix
    diff = merged_data[col2] - merged_data[col1]

    if len(diff) > 2 and not diff.isna().all():
        try:
            stat, p_val = stats.wilcoxon(diff.dropna(), alternative='two-sided')
            return {'p_value': p_val, 'significant': p_val < alpha}
        except ValueError:
            return {'p_value': np.nan, 'significant': False}
    else:
        return {'p_value': np.nan, 'significant': False}


def apply_fdr_correction(significance_dict, alpha=0.05):
    """
    Apply FDR Benjamini-Hochberg correction to p-values.

    Parameters:
    -----------
    significance_dict : dict
        Dictionary of significance results
        Format: {test_name: [{window_center: x, p_value: p, significant: bool}, ...]}
    alpha : float
        Significance level

    Returns:
    --------
    dict : Updated significance_dict with FDR-corrected p-values
    """
    # Collect all p-values
    all_p_values = []
    all_indices = []

    for test_name, sig_list in significance_dict.items():
        for i, sig_dict in enumerate(sig_list):
            all_p_values.append(sig_dict['p_value'])
            all_indices.append((test_name, i))

    # Convert to array and handle NaNs
    p_array = np.array(all_p_values)
    valid_mask = ~np.isnan(p_array)

    if valid_mask.sum() > 0:
        # Apply FDR correction only to valid p-values
        corrected_p_values = np.full_like(p_array, np.nan)
        reject, pvals_corrected, _, _ = multipletests(
            p_array[valid_mask], alpha=alpha, method='fdr_bh'
        )
        corrected_p_values[valid_mask] = pvals_corrected

        # Update significance dictionaries
        for idx, (test_name, list_idx) in enumerate(all_indices):
            significance_dict[test_name][list_idx]['p_value_corrected'] = corrected_p_values[idx]
            significance_dict[test_name][list_idx]['significant'] = (
                (corrected_p_values[idx] < alpha) if not np.isnan(corrected_p_values[idx]) else False
            )

    return significance_dict


# ============================================================================
# HELPER FUNCTIONS - PLOTTING
# ============================================================================

def plot_significance_bars(ax, significance_data, y_position_fraction, color, window_half_width=10, linewidth=4):
    """
    Plot horizontal bars for significant windows.

    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    significance_data : list of dict
        [{'window_center': x, 'significant': bool}, ...]
    y_position_fraction : float
        Position from top as fraction of y-range (e.g., 0.05 = 5% from top)
    color : str
        Color for the bars
    window_half_width : float
        Half-width of each bar in x-axis units
    linewidth : float
        Width of the significance bars (default: 4)
    """
    sig_df = pd.DataFrame(significance_data)
    if sig_df.empty or not sig_df['significant'].any():
        return

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    sig_y = y_max - y_position_fraction * y_range

    sig_windows = sig_df[sig_df['significant']]['window_center'].values
    for window in sig_windows:
        ax.plot([window - window_half_width, window + window_half_width],
                [sig_y, sig_y],
                color=color, linewidth=linewidth, solid_capstyle='round', solid_joinstyle='round')


def plot_single_condition(ax, results_df, metric, color, label, show_individual=False):
    """
    Plot a single condition (mean and SEM).

    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    results_df : pd.DataFrame
        Results dataframe with window_center and metric columns
    metric : str
        Metric column name
    color : str
        Line color
    label : str
        Legend label
    show_individual : bool
        If True, plot individual subject lines
    """
    # Calculate group statistics
    group_stats = results_df.groupby('window_center').agg({
        metric: ['mean', 'std', 'count']
    })
    group_stats[(metric, 'sem')] = group_stats[(metric, 'std')] / np.sqrt(group_stats[(metric, 'count')])

    window_centers = group_stats.index.values
    mean_values = group_stats[(metric, 'mean')].values
    sem_values = group_stats[(metric, 'sem')].values

    # Plot mean line
    ax.plot(window_centers, mean_values, color=color, linewidth=3, label=label)

    # Plot SEM shading (only if not showing individuals)
    if not show_individual:
        ax.fill_between(window_centers, mean_values - sem_values, mean_values + sem_values,
                        color=color, alpha=0.2)

    # Plot individual subjects
    if show_individual:
        for subject in results_df['id_subject'].unique():
            subj_data = results_df[results_df['id_subject'] == subject].sort_values('window_center')
            ax.plot(subj_data['window_center'], subj_data[metric],
                   color=color, alpha=0.5, linewidth=1.5)
