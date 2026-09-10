"""
Trial-based preprocessing: one row per saccade, both sounds per trial.

This module provides an alternative approach to preprocessing that:
1. Logs ALL saccades within each trial (no search window filtering)
2. Includes information about ALL sounds (2 per trial)
3. Creates rows for trials even when no saccades occur
4. Provides timing information relative to both sounds and closest sound

Key differences from standard preprocessing:
- Trial-based structure (uses IMAGE_ONSET to IMAGE_OFFSET boundaries)
- No saccade filtering by search window
- Each row contains info about both sounds in the trial
- Includes timing columns for sound 1, sound 2, and closest sound
"""

import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Dict, Optional


def extract_trial_boundaries(et_data: List[List]) -> List[Tuple[int, int, int]]:
    """
    Extract trial start (IMAGE_ONSET) and end (IMAGE_OFFSET) times.
    Used for EX_1, EX_3, EX_5.

    Parameters:
    -----------
    et_data : list
        Eye-tracking data (list of lists)

    Returns:
    --------
    list : List of tuples (trial_number, start_time, end_time)
           Note: trial_number corresponds to TRIALID from the ASC file
    """
    trials = []
    current_trial_number = None
    current_start_time = None

    for row in et_data:
        if len(row) >= 3 and row[0] == 'MSG':
            # Extract TRIALID
            if row[2] == 'TRIALID' and len(row) >= 4:
                try:
                    current_trial_number = int(row[3])
                except (ValueError, IndexError):
                    pass

            # Extract IMAGE_ONSET (trial start)
            elif row[2] == 'IMAGE_ONSET':
                current_start_time = int(row[1])

            # Extract IMAGE_OFFSET (trial end)
            elif row[2] == 'IMAGE_OFFSET':
                if current_trial_number is not None and current_start_time is not None:
                    end_time = int(row[1])
                    trials.append((current_trial_number, current_start_time, end_time))

    return trials


def extract_trial_boundaries_from_trial_markers(et_data: List[List]) -> List[Tuple[int, int, int]]:
    """
    Extract trial boundaries from TRIAL_N_START/END markers.
    Used for EX_2, EX_4.

    Parameters:
    -----------
    et_data : list
        Eye-tracking data (list of lists)

    Returns:
    --------
    list : List of tuples (trial_number, start_time, end_time)
           Note: trial_number is extracted from TRIAL_N markers
    """
    trials = []
    trial_starts = {}  # Map trial_number -> start_time

    for row in et_data:
        if len(row) >= 3 and row[0] == 'MSG':
            # Look for TRIAL_N_*_START and TRIAL_N_*_END markers
            if row[2].startswith('TRIAL_') and ('_START' in row[2] or '_END' in row[2]):
                try:
                    # Extract trial number from TRIAL_N_...
                    trial_parts = row[2].split('_')
                    trial_num = int(trial_parts[1])
                    timestamp = int(row[1])

                    if '_START' in row[2]:
                        # Store start time
                        trial_starts[trial_num] = timestamp
                    elif '_END' in row[2]:
                        # Match with start time and create trial boundary
                        if trial_num in trial_starts:
                            start_time = trial_starts[trial_num]
                            end_time = timestamp
                            trials.append((trial_num, start_time, end_time))
                except (ValueError, IndexError):
                    continue

    return trials


def organize_sounds_by_trial(matched: List) -> Dict[int, Dict]:
    """
    Organize matched sound-response data by trial number.

    Parameters:
    -----------
    matched : list
        List of matched sound-response events from find_rt
        Format: [sound_time, sound_direction, sound_pitch, response, rt, trial_number,
                 image, sound_1_or_2, trial_start_time, sound_nonempty]

    Returns:
    --------
    dict : Dictionary mapping trial_number to trial info:
           {
               'trial_number': int,
               'image': str,
               'trial_start_time': int,
               'sound_1': dict with sound info,
               'sound_2': dict with sound info
           }
    """
    trials_dict = {}

    for match in matched:
        sound_time, sound_direction, sound_pitch, response, rt, trial_number, image, sound_1_or_2, trial_start_time, sound_nonempty = match

        # Initialize trial if not seen before
        if trial_number not in trials_dict:
            trials_dict[trial_number] = {
                'trial_number': trial_number,
                'image': image,
                'trial_start_time': trial_start_time,
                'sound_1': None,
                'sound_2': None
            }

        # Create sound info dictionary
        sound_info = {
            'time': sound_time,
            'direction': sound_direction,
            'pitch': sound_pitch,
            'response': response,
            'rt': rt,
            'nonempty': sound_nonempty
        }

        # Assign to sound_1 or sound_2
        if sound_1_or_2 == 1:
            trials_dict[trial_number]['sound_1'] = sound_info
        elif sound_1_or_2 == 2:
            trials_dict[trial_number]['sound_2'] = sound_info

    return trials_dict


def find_saccades_in_trial(saccades: List, trial_start: int, trial_end: int) -> List[Tuple[int, List]]:
    """
    Find all saccades that occur within a trial's time boundaries.

    Parameters:
    -----------
    saccades : list
        List of saccades from parse_esacc
    trial_start : int
        Trial start timestamp (IMAGE_ONSET)
    trial_end : int
        Trial end timestamp (IMAGE_OFFSET)

    Returns:
    --------
    list : List of tuples (saccade_index, saccade_data)
           Saccades are included if their start time is within [trial_start, trial_end]
    """
    saccades_in_trial = []

    for idx, sacc in enumerate(saccades):
        sacc_start = sacc[0]  # First element is start time

        # Include saccade if it starts within trial boundaries
        if trial_start <= sacc_start <= trial_end:
            saccades_in_trial.append((idx, sacc))

    return saccades_in_trial


def calculate_timing_columns(sacc_start: int, sacc_end: int,
                            sound_1_time: Optional[int], sound_2_time: Optional[int]) -> Dict:
    """
    Calculate timing columns relative to sound 1, sound 2, and closest sound.

    Parameters:
    -----------
    sacc_start : int
        Saccade start timestamp
    sacc_end : int
        Saccade end timestamp
    sound_1_time : int or None
        First sound timestamp
    sound_2_time : int or None
        Second sound timestamp

    Returns:
    --------
    dict : Dictionary with timing columns
    """
    timing = {}

    # Timing relative to sound 1
    if sound_1_time is not None:
        timing['time_from_sound_1_to_sacc_start'] = sacc_start - sound_1_time
        timing['time_from_sound_1_to_sacc_end'] = sacc_end - sound_1_time
        timing['time_from_sacc_start_to_sound_1'] = sound_1_time - sacc_start
        timing['time_from_sacc_end_to_sound_1'] = sound_1_time - sacc_end
        timing['in_saccade_sound_1'] = (sound_1_time >= sacc_start) and (sound_1_time < sacc_end)
    else:
        timing['time_from_sound_1_to_sacc_start'] = np.nan
        timing['time_from_sound_1_to_sacc_end'] = np.nan
        timing['time_from_sacc_start_to_sound_1'] = np.nan
        timing['time_from_sacc_end_to_sound_1'] = np.nan
        timing['in_saccade_sound_1'] = np.nan

    # Timing relative to sound 2
    if sound_2_time is not None:
        timing['time_from_sound_2_to_sacc_start'] = sacc_start - sound_2_time
        timing['time_from_sound_2_to_sacc_end'] = sacc_end - sound_2_time
        timing['time_from_sacc_start_to_sound_2'] = sound_2_time - sacc_start
        timing['time_from_sacc_end_to_sound_2'] = sound_2_time - sacc_end
        timing['in_saccade_sound_2'] = (sound_2_time >= sacc_start) and (sound_2_time < sacc_end)
    else:
        timing['time_from_sound_2_to_sacc_start'] = np.nan
        timing['time_from_sound_2_to_sacc_end'] = np.nan
        timing['time_from_sacc_start_to_sound_2'] = np.nan
        timing['time_from_sacc_end_to_sound_2'] = np.nan
        timing['in_saccade_sound_2'] = np.nan

    # Determine closest sound and calculate timing
    if sound_1_time is not None and sound_2_time is not None:
        dist_to_sound_1 = abs(sacc_start - sound_1_time)
        dist_to_sound_2 = abs(sacc_start - sound_2_time)

        if dist_to_sound_1 <= dist_to_sound_2:
            timing['closest_sound'] = 1
            timing['time_from_closest_sound_to_sacc_start'] = timing['time_from_sound_1_to_sacc_start']
            timing['time_from_closest_sound_to_sacc_end'] = timing['time_from_sound_1_to_sacc_end']
            timing['time_from_sacc_start_to_closest_sound'] = timing['time_from_sacc_start_to_sound_1']
            timing['time_from_sacc_end_to_closest_sound'] = timing['time_from_sacc_end_to_sound_1']
            timing['in_saccade_closest'] = timing['in_saccade_sound_1']
        else:
            timing['closest_sound'] = 2
            timing['time_from_closest_sound_to_sacc_start'] = timing['time_from_sound_2_to_sacc_start']
            timing['time_from_closest_sound_to_sacc_end'] = timing['time_from_sound_2_to_sacc_end']
            timing['time_from_sacc_start_to_closest_sound'] = timing['time_from_sacc_start_to_sound_2']
            timing['time_from_sacc_end_to_closest_sound'] = timing['time_from_sacc_end_to_sound_2']
            timing['in_saccade_closest'] = timing['in_saccade_sound_2']
    elif sound_1_time is not None:
        # Only sound 1 exists
        timing['closest_sound'] = 1
        timing['time_from_closest_sound_to_sacc_start'] = timing['time_from_sound_1_to_sacc_start']
        timing['time_from_closest_sound_to_sacc_end'] = timing['time_from_sound_1_to_sacc_end']
        timing['time_from_sacc_start_to_closest_sound'] = timing['time_from_sacc_start_to_sound_1']
        timing['time_from_sacc_end_to_closest_sound'] = timing['time_from_sacc_end_to_sound_1']
        timing['in_saccade_closest'] = timing['in_saccade_sound_1']
    elif sound_2_time is not None:
        # Only sound 2 exists
        timing['closest_sound'] = 2
        timing['time_from_closest_sound_to_sacc_start'] = timing['time_from_sound_2_to_sacc_start']
        timing['time_from_closest_sound_to_sacc_end'] = timing['time_from_sound_2_to_sacc_end']
        timing['time_from_sacc_start_to_closest_sound'] = timing['time_from_sacc_start_to_sound_2']
        timing['time_from_sacc_end_to_closest_sound'] = timing['time_from_sacc_end_to_sound_2']
        timing['in_saccade_closest'] = timing['in_saccade_sound_2']
    else:
        # No sounds
        timing['closest_sound'] = np.nan
        timing['time_from_closest_sound_to_sacc_start'] = np.nan
        timing['time_from_closest_sound_to_sacc_end'] = np.nan
        timing['time_from_sacc_start_to_closest_sound'] = np.nan
        timing['time_from_sacc_end_to_closest_sound'] = np.nan
        timing['in_saccade_closest'] = np.nan

    return timing


def calculate_accuracy(target: Optional[str], response, config: Dict) -> Tuple[int, int]:
    """
    Calculate accuracy and response given.

    Parameters:
    -----------
    target : str or None
        Target value (sound direction or pitch depending on experiment)
    response : str or -1
        Participant response
    config : dict
        Experiment configuration

    Returns:
    --------
    tuple : (response_given, acc)
    """
    if response == -1 or target is None:
        return 0, 0
    else:
        acc = 1 if target == response else 0
        return 1, acc


def generate_long_format_alt(id: str, saccades: List, matched: List, config: Dict,
                            et_data: List, fixations: Optional[List] = None,
                            output_dir: Optional[str] = None):
    """
    Generate alternative long format CSV for a participant with trial-based structure.

    This function creates a dataset where:
    - Each row represents a saccade within a trial (or null saccade if trial has none)
    - All saccades within trial boundaries are included (no search window filtering)
    - Each row contains information about both sounds in the trial
    - Timing columns calculated relative to sound 1, sound 2, and closest sound

    Parameters:
    -----------
    id : str
        Participant ID
    saccades : list
        List of ALL saccades (NOT filtered by search window)
    matched : list
        List of matched sound-response events from find_rt
    config : dict
        Experiment configuration dictionary
    et_data : list
        Eye-tracking data (needed to extract trial boundaries)
    fixations : list or None
        List of fixations from parse_efix
    output_dir : str or None
        Output directory for long format files

    Returns:
    --------
    None : Saves CSV file to disk
    """
    # Conversion factors: pixels to degrees of visual angle
    PIX_TO_DEG_X = 32.24 / 1920
    PIX_TO_DEG_Y = 18.14 / 1080

    if output_dir is None:
        output_dir = f'data_preprocessed/{config["name"]}/long_format_alt'

    os.makedirs(output_dir, exist_ok=True)

    # Extract trial boundaries (method depends on experiment type)
    if config.get('training_removal') == 'trial_number':
        # EX_2, EX_4: Use TRIAL_N_START/END markers
        trial_boundaries = extract_trial_boundaries_from_trial_markers(et_data)
    else:
        # EX_1, EX_3, EX_5: Use IMAGE_ONSET/OFFSET markers
        trial_boundaries = extract_trial_boundaries(et_data)

    trial_boundaries_dict = {trial_num: (start, end) for trial_num, start, end in trial_boundaries}

    # Organize sounds by trial
    trials_dict = organize_sounds_by_trial(matched)

    results = []

    # Process each trial
    for trial_number in sorted(trials_dict.keys()):
        trial_info = trials_dict[trial_number]

        # Get trial boundaries
        if trial_number not in trial_boundaries_dict:
            print(f"Warning: Trial {trial_number} has no trial boundary markers. Skipping.")
            continue

        trial_start_time, trial_end_time = trial_boundaries_dict[trial_number]

        # Get sounds
        sound_1 = trial_info['sound_1']
        sound_2 = trial_info['sound_2']

        # Validate that trial has both sounds (as per experimental design)
        if sound_1 is None or sound_2 is None:
            print(f"Warning: Trial {trial_number} does not have both sounds. Sound 1: {sound_1 is not None}, Sound 2: {sound_2 is not None}")

        # Find all saccades in this trial
        saccades_in_trial = find_saccades_in_trial(saccades, trial_start_time, trial_end_time)

        # If no saccades, create TWO rows (one for each sound) with null saccade data
        if len(saccades_in_trial) == 0:
            row_sound_1 = create_row_null_saccade(trial_info, sound_1, sound_2, trial_start_time, trial_end_time, config, which_sound=1)
            row_sound_2 = create_row_null_saccade(trial_info, sound_1, sound_2, trial_start_time, trial_end_time, config, which_sound=2)
            results.append(row_sound_1)
            results.append(row_sound_2)
        else:
            # Create one row per saccade
            for sacc_idx_in_trial, (sacc_global_idx, sacc) in enumerate(saccades_in_trial, start=1):
                row = create_row_with_saccade(
                    trial_info, sound_1, sound_2, trial_start_time, trial_end_time,
                    sacc, sacc_global_idx, sacc_idx_in_trial, len(saccades_in_trial),
                    fixations, config, PIX_TO_DEG_X, PIX_TO_DEG_Y
                )
                results.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Round coordinate columns
    coordinate_cols = ['amplitude', 'amplitude_deg', 'x_start', 'x_end', 'y_start', 'y_end',
                      'x_start_deg', 'x_end_deg', 'y_start_deg', 'y_end_deg']
    for col in coordinate_cols:
        if col in df.columns:
            df[col] = df[col].round(2)

    # Reorder columns for better organization
    ordered_cols = get_ordered_columns(df, config)
    df = df[ordered_cols]

    # Save to CSV
    output_path = os.path.join(output_dir, f"{id}_long_alt.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved alternative long format for subject {id} ({len(df)} rows, {df['trial_number'].nunique()} trials)")


def create_row_null_saccade(trial_info: Dict, sound_1: Optional[Dict], sound_2: Optional[Dict],
                           trial_start_time: int, trial_end_time: int, config: Dict, which_sound: int) -> Dict:
    """
    Create a row for a trial with no saccades.

    Parameters
    ----------
    which_sound : int
        Which sound this row represents (1 or 2)
    """
    row = {}

    # Trial info
    row['trial_number'] = trial_info['trial_number']
    row['image'] = trial_info['image']
    row['trial_start_time'] = trial_start_time
    row['trial_end_time'] = trial_end_time
    row['trial_duration'] = trial_end_time - trial_start_time

    # Saccade info (all null)
    row['id_sacc'] = np.nan
    row['sacc_ordinal_in_trial'] = np.nan
    row['n_saccades_in_trial'] = 0
    row['sacc_start'] = np.nan
    row['sacc_end'] = np.nan
    row['sacc_direction'] = None
    row['sacc_dur'] = np.nan
    row['amplitude'] = np.nan
    row['amplitude_deg'] = np.nan
    row['x_start'] = np.nan
    row['x_end'] = np.nan
    row['y_start'] = np.nan
    row['y_end'] = np.nan
    row['x_start_deg'] = np.nan
    row['x_end_deg'] = np.nan
    row['y_start_deg'] = np.nan
    row['y_end_deg'] = np.nan
    row['is_blink'] = np.nan
    row['prev_fix_duration'] = np.nan
    row['next_fix_duration'] = np.nan

    # Timing columns (all null)
    for col in ['time_from_sound_1_to_sacc_start', 'time_from_sound_1_to_sacc_end',
                'time_from_sacc_start_to_sound_1', 'time_from_sacc_end_to_sound_1',
                'in_saccade_sound_1', 'time_from_sound_2_to_sacc_start',
                'time_from_sound_2_to_sacc_end', 'time_from_sacc_start_to_sound_2',
                'time_from_sacc_end_to_sound_2', 'in_saccade_sound_2',
                'time_from_closest_sound_to_sacc_start',
                'time_from_closest_sound_to_sacc_end', 'time_from_sacc_start_to_closest_sound',
                'time_from_sacc_end_to_closest_sound', 'in_saccade_closest',
                'time_from_trial_start_to_sacc_start', 'time_from_trial_end_to_sacc_start']:
        row[col] = np.nan

    # Set closest_sound and id_sound based on which_sound parameter
    row['closest_sound'] = which_sound
    row['id_sound'] = (trial_info['trial_number'] - 1) * 2 + which_sound

    # Consistency columns
    row['consistent_with_sound_1'] = np.nan
    row['consistent_with_sound_2'] = np.nan
    row['consistent_with_closest'] = np.nan

    # Sound 1 info
    if sound_1 is not None:
        row['sound_1_time'] = sound_1['time']
        row['sound_1_direction'] = sound_1['direction']
        row['sound_1_nonempty'] = sound_1['nonempty']
        row['sound_1_response'] = sound_1['response'] if sound_1['response'] != -1 else None
        row['sound_1_rt'] = sound_1['rt'] if sound_1['rt'] != -1 else np.nan

        target_1 = sound_1['direction'] if config['target_field'] == 'sound_direction' else sound_1['pitch']
        response_given_1, acc_1 = calculate_accuracy(target_1, sound_1['response'], config)
        row['sound_1_response_given'] = response_given_1
        row['sound_1_acc'] = acc_1

        if config['has_pitch']:
            row['sound_1_pitch'] = sound_1['pitch']
    else:
        row['sound_1_time'] = np.nan
        row['sound_1_direction'] = None
        row['sound_1_nonempty'] = np.nan
        row['sound_1_response'] = None
        row['sound_1_rt'] = np.nan
        row['sound_1_response_given'] = 0
        row['sound_1_acc'] = 0
        if config['has_pitch']:
            row['sound_1_pitch'] = None

    # Sound 2 info
    if sound_2 is not None:
        row['sound_2_time'] = sound_2['time']
        row['sound_2_direction'] = sound_2['direction']
        row['sound_2_nonempty'] = sound_2['nonempty']
        row['sound_2_response'] = sound_2['response'] if sound_2['response'] != -1 else None
        row['sound_2_rt'] = sound_2['rt'] if sound_2['rt'] != -1 else np.nan

        target_2 = sound_2['direction'] if config['target_field'] == 'sound_direction' else sound_2['pitch']
        response_given_2, acc_2 = calculate_accuracy(target_2, sound_2['response'], config)
        row['sound_2_response_given'] = response_given_2
        row['sound_2_acc'] = acc_2

        if config['has_pitch']:
            row['sound_2_pitch'] = sound_2['pitch']
    else:
        row['sound_2_time'] = np.nan
        row['sound_2_direction'] = None
        row['sound_2_nonempty'] = np.nan
        row['sound_2_response'] = None
        row['sound_2_rt'] = np.nan
        row['sound_2_response_given'] = 0
        row['sound_2_acc'] = 0
        if config['has_pitch']:
            row['sound_2_pitch'] = None

    return row


def create_row_with_saccade(trial_info: Dict, sound_1: Optional[Dict], sound_2: Optional[Dict],
                           trial_start_time: int, trial_end_time: int,
                           sacc: List, sacc_global_idx: int, sacc_idx_in_trial: int,
                           n_saccades_in_trial: int, fixations: Optional[List],
                           config: Dict, PIX_TO_DEG_X: float, PIX_TO_DEG_Y: float) -> Dict:
    """
    Create a row for a trial with a saccade.
    """
    sacc_start, sacc_end, sacc_dir, sacc_duration, sacc_dist, x_start, x_end, y_start, y_end, is_blink, *_sacc_extra = sacc
    peak_velocity = _sacc_extra[0] if _sacc_extra else np.nan

    row = {}

    # Trial info
    row['trial_number'] = trial_info['trial_number']
    row['image'] = trial_info['image']
    row['trial_start_time'] = trial_start_time
    row['trial_end_time'] = trial_end_time
    row['trial_duration'] = trial_end_time - trial_start_time

    # Saccade info
    row['id_sacc'] = sacc_global_idx
    row['sacc_ordinal_in_trial'] = sacc_idx_in_trial
    row['n_saccades_in_trial'] = n_saccades_in_trial
    row['sacc_start'] = sacc_start
    row['sacc_end'] = sacc_end
    row['sacc_direction'] = sacc_dir
    row['sacc_dur'] = sacc_duration
    row['peak_velocity'] = peak_velocity

    # Calculate amplitude
    amplitude = np.sqrt(sacc_dist**2 + (y_end - y_start)**2)
    x_dist_deg = sacc_dist * PIX_TO_DEG_X
    y_dist_deg = abs(y_end - y_start) * PIX_TO_DEG_Y
    amplitude_deg = np.sqrt(x_dist_deg**2 + y_dist_deg**2)

    row['amplitude'] = amplitude
    row['amplitude_deg'] = amplitude_deg
    row['x_start'] = x_start
    row['x_end'] = x_end
    row['y_start'] = y_start
    row['y_end'] = y_end
    row['x_start_deg'] = x_start * PIX_TO_DEG_X
    row['x_end_deg'] = x_end * PIX_TO_DEG_X
    row['y_start_deg'] = y_start * PIX_TO_DEG_Y
    row['y_end_deg'] = y_end * PIX_TO_DEG_Y
    row['is_blink'] = is_blink

    # Find preceding and following fixations
    prev_fix_duration = np.nan
    next_fix_duration = np.nan

    if fixations is not None:
        prev_fix = None
        for fix in fixations:
            fix_start, fix_end, fix_dur = fix
            if fix_end <= sacc_start:
                prev_fix = fix
            else:
                break

        if prev_fix is not None:
            prev_fix_duration = prev_fix[2]

        for fix in fixations:
            fix_start, fix_end, fix_dur = fix
            if fix_start >= sacc_end:
                next_fix_duration = fix_dur
                break

    row['prev_fix_duration'] = prev_fix_duration
    row['next_fix_duration'] = next_fix_duration

    # Timing relative to trial boundaries
    row['time_from_trial_start_to_sacc_start'] = sacc_start - trial_start_time
    row['time_from_trial_end_to_sacc_start'] = sacc_start - trial_end_time

    # Calculate timing columns
    sound_1_time = sound_1['time'] if sound_1 is not None else None
    sound_2_time = sound_2['time'] if sound_2 is not None else None

    timing = calculate_timing_columns(sacc_start, sacc_end, sound_1_time, sound_2_time)
    row.update(timing)

    # Create id_sound: unique identifier for the closest sound (1-400 for 200 trials)
    if not np.isnan(timing['closest_sound']):
        row['id_sound'] = (trial_info['trial_number'] - 1) * 2 + int(timing['closest_sound'])
    else:
        row['id_sound'] = np.nan

    # Consistency columns
    if sound_1 is not None and sound_1['direction'] is not None:
        row['consistent_with_sound_1'] = (sacc_dir == sound_1['direction'])
    else:
        row['consistent_with_sound_1'] = np.nan

    if sound_2 is not None and sound_2['direction'] is not None:
        row['consistent_with_sound_2'] = (sacc_dir == sound_2['direction'])
    else:
        row['consistent_with_sound_2'] = np.nan

    # Consistency with closest sound
    if not np.isnan(timing['closest_sound']):
        closest = int(timing['closest_sound'])
        closest_sound = sound_1 if closest == 1 else sound_2
        if closest_sound is not None and closest_sound['direction'] is not None:
            row['consistent_with_closest'] = (sacc_dir == closest_sound['direction'])
        else:
            row['consistent_with_closest'] = np.nan
    else:
        row['consistent_with_closest'] = np.nan

    # Sound 1 info
    if sound_1 is not None:
        row['sound_1_time'] = sound_1['time']
        row['sound_1_direction'] = sound_1['direction']
        row['sound_1_nonempty'] = sound_1['nonempty']
        row['sound_1_response'] = sound_1['response'] if sound_1['response'] != -1 else None
        row['sound_1_rt'] = sound_1['rt'] if sound_1['rt'] != -1 else np.nan

        target_1 = sound_1['direction'] if config['target_field'] == 'sound_direction' else sound_1['pitch']
        response_given_1, acc_1 = calculate_accuracy(target_1, sound_1['response'], config)
        row['sound_1_response_given'] = response_given_1
        row['sound_1_acc'] = acc_1

        if config['has_pitch']:
            row['sound_1_pitch'] = sound_1['pitch']
    else:
        row['sound_1_time'] = np.nan
        row['sound_1_direction'] = None
        row['sound_1_nonempty'] = np.nan
        row['sound_1_response'] = None
        row['sound_1_rt'] = np.nan
        row['sound_1_response_given'] = 0
        row['sound_1_acc'] = 0
        if config['has_pitch']:
            row['sound_1_pitch'] = None

    # Sound 2 info
    if sound_2 is not None:
        row['sound_2_time'] = sound_2['time']
        row['sound_2_direction'] = sound_2['direction']
        row['sound_2_nonempty'] = sound_2['nonempty']
        row['sound_2_response'] = sound_2['response'] if sound_2['response'] != -1 else None
        row['sound_2_rt'] = sound_2['rt'] if sound_2['rt'] != -1 else np.nan

        target_2 = sound_2['direction'] if config['target_field'] == 'sound_direction' else sound_2['pitch']
        response_given_2, acc_2 = calculate_accuracy(target_2, sound_2['response'], config)
        row['sound_2_response_given'] = response_given_2
        row['sound_2_acc'] = acc_2

        if config['has_pitch']:
            row['sound_2_pitch'] = sound_2['pitch']
    else:
        row['sound_2_time'] = np.nan
        row['sound_2_direction'] = None
        row['sound_2_nonempty'] = np.nan
        row['sound_2_response'] = None
        row['sound_2_rt'] = np.nan
        row['sound_2_response_given'] = 0
        row['sound_2_acc'] = 0
        if config['has_pitch']:
            row['sound_2_pitch'] = None

    return row


def get_ordered_columns(df: pd.DataFrame, config: Dict) -> List[str]:
    """
    Get ordered list of columns for better organization.
    """
    # Trial info columns
    trial_cols = ['trial_number', 'image', 'trial_start_time', 'trial_end_time', 'trial_duration']

    # Saccade ID columns
    sacc_id_cols = ['id_sacc', 'sacc_ordinal_in_trial', 'n_saccades_in_trial']

    # Saccade timing columns
    sacc_timing_cols = ['sacc_start', 'sacc_end', 'time_from_trial_start_to_sacc_start',
                        'time_from_trial_end_to_sacc_start']

    # Sound timing columns
    sound_timing_cols = [
        'time_from_sound_1_to_sacc_start', 'time_from_sound_1_to_sacc_end',
        'time_from_sacc_start_to_sound_1', 'time_from_sacc_end_to_sound_1', 'in_saccade_sound_1',
        'time_from_sound_2_to_sacc_start', 'time_from_sound_2_to_sacc_end',
        'time_from_sacc_start_to_sound_2', 'time_from_sacc_end_to_sound_2', 'in_saccade_sound_2',
        'closest_sound', 'time_from_closest_sound_to_sacc_start', 'time_from_closest_sound_to_sacc_end',
        'time_from_sacc_start_to_closest_sound', 'time_from_sacc_end_to_closest_sound', 'in_saccade_closest'
    ]

    # Saccade properties
    sacc_props = ['sacc_direction', 'sacc_dur', 'amplitude', 'amplitude_deg',
                  'x_start', 'x_end', 'y_start', 'y_end',
                  'x_start_deg', 'x_end_deg', 'y_start_deg', 'y_end_deg',
                  'is_blink', 'prev_fix_duration', 'next_fix_duration']

    # Consistency columns
    consistency_cols = ['consistent_with_sound_1', 'consistent_with_sound_2', 'consistent_with_closest']

    # Sound 1 columns
    sound_1_cols = ['sound_1_time', 'sound_1_direction', 'sound_1_nonempty',
                    'sound_1_response', 'sound_1_response_given', 'sound_1_acc', 'sound_1_rt']
    if config['has_pitch']:
        sound_1_cols.insert(2, 'sound_1_pitch')

    # Sound 2 columns
    sound_2_cols = ['sound_2_time', 'sound_2_direction', 'sound_2_nonempty',
                    'sound_2_response', 'sound_2_response_given', 'sound_2_acc', 'sound_2_rt']
    if config['has_pitch']:
        sound_2_cols.insert(2, 'sound_2_pitch')

    # Build ordered list
    ordered = []
    for col_list in [trial_cols, sacc_id_cols, sacc_timing_cols, sound_timing_cols,
                     sacc_props, consistency_cols, sound_1_cols, sound_2_cols]:
        for col in col_list:
            if col in df.columns:
                ordered.append(col)

    # Add any remaining columns
    for col in df.columns:
        if col not in ordered:
            ordered.append(col)

    return ordered


def collect_all_long_files_alt(config: Dict, input_dir: Optional[str] = None,
                               output_dir: Optional[str] = None,
                               output_file: str = 'long_format_alt_all.csv'):
    """
    Collect all individual alternative long format files into one aggregated file.

    Parameters:
    -----------
    config : dict
        Experiment configuration dictionary
    input_dir : str or None
        Directory containing individual long format files (if None, uses default)
    output_dir : str or None
        Output directory for aggregated file (if None, uses default)
    output_file : str
        Name of the output file (default: 'long_format_alt_all.csv')

    Returns:
    --------
    None : Saves aggregated CSV file to disk
    """
    if input_dir is None:
        input_dir = f'data_preprocessed/{config["name"]}/long_format_alt'
    if output_dir is None:
        output_dir = f'data_preprocessed/{config["name"]}'

    all_files = [f for f in os.listdir(input_dir) if f.endswith('_long_alt.csv')]
    df_all = pd.DataFrame()

    for file in all_files:
        participant_id = file.replace('_long_alt.csv', '')
        df = pd.read_csv(os.path.join(input_dir, file))
        df.insert(0, 'id_subject', participant_id)
        df_all = pd.concat([df_all, df], ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    df_all.to_csv(output_path, index=False)
    print(f"Individual alternative files were collected into one file: {output_path}")
    print(f"Total: {len(df_all)} rows, {df_all['id_subject'].nunique()} subjects, {df_all['trial_number'].nunique()} unique trial numbers")
