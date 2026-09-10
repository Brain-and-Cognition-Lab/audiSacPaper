"""
Shared configuration and data loading for the four experiments reported in the
manuscript.

Directory names are historical and do NOT match the paper's numbering:

    EX_1  ->  Experiment 1a  (binaural-direction, 10 ms pink noise)
    EX_9  ->  Experiment 1b  (binaural-direction, 25 ms pink noise)
    EX_7  ->  Experiment 2   (monaural-direction, 6 ms tone)
    EX_3  ->  Experiment 3   (monaural-pitch, 6 ms tone)

All amplitudes are in degrees of visual angle.
"""

import os

import numpy as np
import pandas as pd


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_CONFIGS = {
    'EX_1': {
        'name': 'EX_1',
        'paper_label': 'Exp 1a. binaural-direction',
        'description': 'Sound localization (left/right), 10 ms binaural pink noise with ITD',
        'target_field': 'sound_direction',
        'response_values': ('left', 'right'),
        'rt_ylim': (600, 750),
        'sdt_column': 'sound_direction',   # task-relevant column for SDT (d-prime, c)
    },
    'EX_9': {
        'name': 'EX_9',
        'paper_label': 'Exp 1b. binaural-direction',
        'description': 'Sound localization (left/right), 25 ms binaural pink noise with ITD',
        'target_field': 'sound_direction',
        'response_values': ('left', 'right'),
        'rt_ylim': (600, 750),
        'sdt_column': 'sound_direction',
    },
    'EX_7': {
        'name': 'EX_7',
        'paper_label': 'Exp 2. monaural-direction',
        'description': 'Sound localization (left/right), 6 ms monaural tone; pitch varies but is task-irrelevant',
        'target_field': 'sound_direction',
        'response_values': ('left', 'right'),
        'rt_ylim': (500, 650),
        'sdt_column': 'sound_direction',
    },
    'EX_3': {
        'name': 'EX_3',
        'paper_label': 'Exp 3. monaural-pitch',
        'description': 'Pitch discrimination (high/low), 6 ms monaural tone; ear of presentation is task-irrelevant',
        'target_field': 'sound_pitch',
        # NOTE: answered on the up/down keys, so `response` holds 'high'/'low'
        # here and 'left'/'right' in the three localization experiments.
        'response_values': ('low', 'high'),
        'rt_ylim': (750, 900),
        'sdt_column': 'sound_pitch',
    },
}

EXPERIMENTS = ['EX_1', 'EX_9', 'EX_7', 'EX_3']
EXPERIMENT_LABELS = [EXPERIMENT_CONFIGS[e]['paper_label'] for e in EXPERIMENTS]


# ============================================================================
# UNITS AND COLUMN MAPPING
# ============================================================================
# The paper analyses saccade amplitude in degrees of visual angle.

AMPLITUDE_UNIT = 'degrees'
AMPLITUDE_UNIT_LABEL = '\N{DEGREE SIGN}'

METRIC_COLUMN_MAP = {
    'amplitude': 'amplitude_deg',
    'x_start': 'x_start_deg',
    'y_start': 'y_start_deg',
    'x_end': 'x_end_deg',
    'y_end': 'y_end_deg',
    'duration': 'sacc_dur',
}


# ============================================================================
# FILTER PARAMETERS
# ============================================================================

MAX_DURATION = 100    # ms   - maximum saccade duration
MIN_AMPLITUDE = 0.05  # deg  - minimum saccade amplitude (~1 pixel)
MIN_RT = 300          # ms   - responses faster than this are treated as anticipations


# ============================================================================
# SCREEN DIMENSIONS
# ============================================================================

SCREEN_WIDTH_PX = 1920      # Screen width in pixels
SCREEN_HEIGHT_PX = 1080     # Screen height in pixels
SCREEN_WIDTH_DEG = 32.24    # Screen width in degrees (visual angle)
SCREEN_HEIGHT_DEG = 18.14   # Screen height in degrees (visual angle)

SCREEN_WIDTH = SCREEN_WIDTH_DEG
SCREEN_HEIGHT = SCREEN_HEIGHT_DEG


# ============================================================================
# HELPER FUNCTION TO BUILD FILTER
# ============================================================================

def build_filter(data, include_response=False):
    """
    Build the standard analysis filter used by every analysis in the paper.

    Keeps saccades that are at most MAX_DURATION long, at least MIN_AMPLITUDE
    wide, not part of a blink, and paired with a non-empty sound; trials with a
    response faster than MIN_RT are dropped as anticipations.

    Parameters
    ----------
    data : pd.DataFrame
        Data to filter
    include_response : bool
        If True, also require valid responses (for RT analyses)

    Returns
    -------
    pd.Series
        Boolean filter
    """
    analysis_filter = (
        (data['sacc_dur'] <= MAX_DURATION) &
        (data[METRIC_COLUMN_MAP['amplitude']] >= MIN_AMPLITUDE) &
        (data['is_blink'] == 0) &
        (data['sound_nonempty'] == 1) &
        ((data['response_given'] == 0) | (data['rt'] > MIN_RT))
    )

    if include_response:
        analysis_filter = analysis_filter & (data['acc'] != -1) & (data['rt'] > 0)

    return analysis_filter


# ============================================================================
# ALTERNATIVE PREPROCESSING DATA PREPARATION
# ============================================================================

def prepare_alternative_data(data_raw, sound_selection='both'):
    """
    Prepare alternative preprocessing data for analysis by creating derived columns.

    The alternative preprocessing structure has sound_1_* and sound_2_* columns for
    both sounds in each trial. This function creates derived columns that extract
    information from the closest sound, making the data compatible with existing
    analysis functions.

    Parameters
    ----------
    data_raw : pd.DataFrame
        Raw data from alternative preprocessing (long_format_alt_all.csv)
    sound_selection : str, optional
        Which sound(s) to include in the analysis:
        - 'both': Use closest sound (default, original behavior)
        - 'sound1': Only use sound 1 data
        - 'sound2': Only use sound 2 data

    Returns
    -------
    pd.DataFrame
        Data with derived columns added for compatibility with analysis functions

    Notes
    -----
    Derived columns created:
    - sound_direction: Direction of closest sound (or specified sound)
    - sound_pitch: Pitch of closest sound (if applicable)
    - sound_nonempty: Whether closest sound was non-empty
    - response: Response given to closest sound
    - response_given: Whether response was given to closest sound
    - acc: Accuracy for closest sound
    - rt: Reaction time for closest sound
    - sound_time: Timestamp of closest sound (or specified sound)
    - time_from_sound_to_sacc_start: Mapped from time_from_closest_sound_to_sacc_start (or sound-specific)
    - time_from_sound_to_sacc_end: Mapped from time_from_closest_sound_to_sacc_end (or sound-specific)
    - time_from_sacc_start_to_sound: Mapped from time_from_sacc_start_to_closest_sound (or sound-specific)
    - time_from_sacc_end_to_sound: Mapped from time_from_sacc_end_to_closest_sound (or sound-specific)
    - time_from_RT_to_sacc_start: Time from response to saccade start (for response-aligned analysis)
    - in_saccade: Mapped from in_saccade_closest (or sound-specific)
    - consistent: Mapped from consistent_with_closest (or sound-specific)
    - sound_number: Which sound this row represents (1 or 2, or from closest_sound)
    """

    data = data_raw.copy()

    # Determine which sound to use for each row
    if sound_selection == 'sound1':
        sound_idx = 1
    elif sound_selection == 'sound2':
        sound_idx = 2
    elif sound_selection == 'both':
        sound_idx = None  # Will use closest_sound column
    else:
        raise ValueError(f"Invalid sound_selection: {sound_selection}. Must be 'both', 'sound1', or 'sound2'")

    # Helper function to get the sound index for each row
    def get_sound_idx(row):
        if sound_idx is not None:
            return sound_idx
        elif not pd.isna(row['closest_sound']):
            return int(row['closest_sound'])
        else:
            return None

    # Derive columns for the selected sound
    data['sound_number'] = data.apply(get_sound_idx, axis=1)

    data['sound_direction'] = data.apply(
        lambda row: row[f'sound_{int(row["sound_number"])}_direction'] if row['sound_number'] is not None else None,
        axis=1
    )

    # Check if pitch columns exist (experiment-dependent)
    has_pitch = 'sound_1_pitch' in data.columns
    if has_pitch:
        data['sound_pitch'] = data.apply(
            lambda row: row[f'sound_{int(row["sound_number"])}_pitch'] if row['sound_number'] is not None else None,
            axis=1
        )

    data['sound_nonempty'] = data.apply(
        lambda row: row[f'sound_{int(row["sound_number"])}_nonempty'] if row['sound_number'] is not None else 0,
        axis=1
    )

    data['response'] = data.apply(
        lambda row: row[f'sound_{int(row["sound_number"])}_response'] if row['sound_number'] is not None else None,
        axis=1
    )

    data['response_given'] = data.apply(
        lambda row: row[f'sound_{int(row["sound_number"])}_response_given'] if row['sound_number'] is not None else 0,
        axis=1
    )

    data['acc'] = data.apply(
        lambda row: row[f'sound_{int(row["sound_number"])}_acc'] if row['sound_number'] is not None else 0,
        axis=1
    )

    data['rt'] = data.apply(
        lambda row: row[f'sound_{int(row["sound_number"])}_rt'] if row['sound_number'] is not None else np.nan,
        axis=1
    )

    # Add sound_time column
    data['sound_time'] = data.apply(
        lambda row: row[f'sound_{int(row["sound_number"])}_time'] if row['sound_number'] is not None else np.nan,
        axis=1
    )

    # Map the timing column names based on sound selection
    if sound_selection == 'both':
        # Use closest sound timing columns
        data['time_from_sound_to_sacc_start'] = data['time_from_closest_sound_to_sacc_start']
        data['time_from_sound_to_sacc_end'] = data['time_from_closest_sound_to_sacc_end']
        data['time_from_sacc_start_to_sound'] = data['time_from_sacc_start_to_closest_sound']
        data['time_from_sacc_end_to_sound'] = data['time_from_sacc_end_to_closest_sound']
        data['in_saccade'] = data['in_saccade_closest']
        data['consistent'] = data['consistent_with_closest']
    else:
        # Use specific sound timing columns
        sound_num = sound_idx
        data['time_from_sound_to_sacc_start'] = data[f'time_from_sound_{sound_num}_to_sacc_start']
        data['time_from_sound_to_sacc_end'] = data[f'time_from_sound_{sound_num}_to_sacc_end']
        data['time_from_sacc_start_to_sound'] = data[f'time_from_sacc_start_to_sound_{sound_num}']
        data['time_from_sacc_end_to_sound'] = data[f'time_from_sacc_end_to_sound_{sound_num}']
        data['in_saccade'] = data[f'in_saccade_sound_{sound_num}']
        data['consistent'] = data[f'consistent_with_sound_{sound_num}']

    # Calculate RT-based timing (for response-aligned analysis)
    # time_from_RT_to_sacc_start = sacc_start - RT_time
    # Note: RT is in ms from sound onset, so RT_time = sound_time + RT
    # Therefore: time_from_RT_to_sacc_start = sacc_start - (sound_time + RT)
    #           = (sacc_start - sound_time) - RT
    #           = time_from_sacc_start_to_sound - RT
    data['time_from_RT_to_sacc_start'] = data['time_from_sound_to_sacc_start'] - data['rt']

    return data


# ============================================================================
# UNIFIED DATA LOADING
# ============================================================================

def load_experiment_data(experiment, data_directory=None):
    """
    Load and prepare data for one experiment ('EX_1', 'EX_9', 'EX_7', 'EX_3').

    Reads long_format_alt_all.csv (one row per saccade, carrying the columns of
    both sounds in the trial) and runs it through prepare_alternative_data() to
    add the unified, closest-sound columns the analyses expect.

    Parameters
    ----------
    experiment : str
        Experiment directory name (see EXPERIMENT_CONFIGS).
    data_directory : str, optional
        Path to the data directory. Defaults to f'data_preprocessed/{experiment}'.

    Returns
    -------
    pd.DataFrame
        Unified format with, among the original columns:
        id_subject, sacc_direction, sound_direction, consistent, response,
        response_given, acc, rt, sound_nonempty, in_saccade,
        time_from_sound_to_sacc_start, time_from_sound_to_sacc_end,
        time_from_sacc_start_to_sound, time_from_sacc_end_to_sound
    """
    if experiment not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown experiment: {experiment}. "
            f"Must be one of {list(EXPERIMENT_CONFIGS)}."
        )

    if data_directory is None:
        data_directory = f'data_preprocessed/{experiment}'

    file_path = os.path.join(data_directory, 'long_format_alt_all.csv')

    print(f"Loading: {file_path}")
    data_raw = pd.read_csv(file_path, low_memory=False)
    print(f"  Raw data loaded: {len(data_raw):,} rows")
    print(f"  Unique subjects: {data_raw['id_subject'].nunique()}")

    data = prepare_alternative_data(data_raw, sound_selection='both')
    print("  Data prepared using closest sound approach")

    return data
