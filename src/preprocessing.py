import pandas as pd
import numpy as np
import os


# ====================================================================
# TRAINING DATA REMOVAL
# ====================================================================

def find_experiment_data(data, search_item='People_167_h.jpg'):
    """
    Removes training data by finding the last occurrence of training stimulus.
    Used for EX_1, EX_3, EX_5.

    Parameters:
    -----------
    data : list
        List of lists containing eye-tracking data rows
    search_item : str
        String to search for in the data to identify the last training trial
        (default: 'People_167_h.jpg')

    Returns:
    --------
    list : Data with training trials removed (only experimental trials)
    """
    last_occurrence_idx = -1
    ending_line = ['MSG', None, 'TRIAL_RESULT', '0']

    for i, sublist in enumerate(data):
        if search_item in sublist:
            last_occurrence_idx = i

    print(f'The last occurrence of {search_item} is line {last_occurrence_idx}. Training ended there')
    if last_occurrence_idx == -1:
        print(f'{search_item} not found.')

    for i in range(last_occurrence_idx + 1, len(data)):
        sublist = data[i]
        if (len(sublist) >= 4 and
            sublist[0] == ending_line[0] and
            sublist[2] == ending_line[2] and
            sublist[3] == ending_line[3]):
            return data[i + 1:]

    return data


def find_experiment_data_by_trial_number(data, min_trial_number=20):
    """
    Removes training data by finding the first occurrence of experimental trial.
    Used for EX_2, EX_4 which use TRIAL_N_*_START markers.

    Training trials are numbered 0-19, experimental trials start from 20+.
    This function finds the first TRIAL_20_*_START marker and returns data from there.

    Parameters:
    -----------
    data : list
        List of lists containing eye-tracking data rows
    min_trial_number : int
        Minimum trial number for experimental trials (default: 20)

    Returns:
    --------
    list : Data with training trials removed (only experimental trials)
    """
    for i, row in enumerate(data):
        if len(row) >= 3 and row[0] == 'MSG':
            # Look for TRIAL_N_*_START where N >= min_trial_number
            if row[2].startswith('TRIAL_') and '_START' in row[2]:
                try:
                    # Extract trial number from TRIAL_N_...
                    trial_parts = row[2].split('_')
                    trial_num = int(trial_parts[1])
                    if trial_num >= min_trial_number:
                        print(f'First experimental trial (TRIAL_{trial_num}) found at line {i}. Training ended there')
                        return data[i:]
                except (ValueError, IndexError):
                    continue

    print(f'Warning: No experimental trials (TRIAL_{min_trial_number}+) found. Returning all data.')
    return data


# ====================================================================
# SOUND EVENT EXTRACTION
# ====================================================================

def extract_sound_events_ex1(et_data, behavior_df, config):
    """
    Extract sound events for EX_1 (uses behavioral CSV for direction info).

    Parameters:
    -----------
    et_data : list
        Eye-tracking data (list of lists)
    behavior_df : pd.DataFrame
        Behavioral data containing trial information
    config : dict
        Experiment configuration dictionary

    Returns:
    --------
    list : List of sound events in format [time_sound, sound_direction, sound_pitch,
                                           trial_number, image, sound_1_or_2, trial_start_time, sound_nonempty]
    """
    events = []
    sound_markers = config['sound_markers']

    # Filter only experimental trials
    exp_trials = behavior_df[behavior_df['trial_type'] == 'experiment'].reset_index(drop=True)

    # Track trial start times from IMAGE_ONSET markers
    current_trial_start_time = None
    sound_counter = 0

    for row in et_data:
        if len(row) >= 3 and row[0] == 'MSG':
            # Extract IMAGE_ONSET as trial start time
            if row[2] == 'IMAGE_ONSET':
                current_trial_start_time = int(row[1])

            # Extract sound events
            elif row[2] in sound_markers:
                time_sound = int(row[1])
                marker = row[2]

                trial_idx = sound_counter // 2
                if trial_idx >= len(exp_trials):
                    break  # safeguard

                if marker == 'FIRST_SOUND_PLAYED':
                    direction = exp_trials.loc[trial_idx, 'first_sound_type']
                    sound_1_or_2 = 1
                else:
                    direction = exp_trials.loc[trial_idx, 'second_sound_type']
                    sound_1_or_2 = 2

                direction = direction.replace('sound_', '')  # 'sound_left' → 'left'

                # Extract trial_number and image information
                trial_number = exp_trials.loc[trial_idx, 'trial_number']
                image = exp_trials.loc[trial_idx, 'image']

                # For EX_1, all sounds are non-empty
                sound_nonempty = 1

                # For EX_1, pitch is not relevant (set to None)
                events.append([time_sound, direction, None, trial_number, image, sound_1_or_2, current_trial_start_time, sound_nonempty])
                sound_counter += 1

    return events


def extract_sound_events_ex3(et_data, behavior_df, config):
    """
    Extract sound events for EX_3/EX_5 (extracts direction and pitch from marker name).
    Also extracts trial number, image, sound order, and trial start time from ASC file MSG lines.

    Parameters:
    -----------
    et_data : list
        Eye-tracking data (list of lists)
    behavior_df : pd.DataFrame or None
        Behavioral data (not used for EX_3/EX_5, can be None)
    config : dict
        Experiment configuration dictionary

    Returns:
    --------
    list : List of sound events in format [time_sound, sound_direction, sound_pitch,
                                           trial_number, image, sound_1_or_2, trial_start_time, sound_nonempty]
    """
    events = []
    sound_markers = config['sound_markers']

    # Track current trial information
    current_trial_number = None
    current_image = None
    current_trial_start_time = None
    sounds_in_current_trial = 0

    for row in et_data:
        if len(row) >= 3 and row[0] == 'MSG':
            # Extract TRIALID
            # Format: MSG    7820753 TRIALID 35
            if row[2] == 'TRIALID' and len(row) >= 4:
                try:
                    current_trial_number = int(row[3])
                    sounds_in_current_trial = 0  # Reset sound counter for new trial
                except (ValueError, IndexError):
                    pass

            # Extract IMAGE_ONSET as trial start time
            # Format: MSG    2207493 IMAGE_ONSET
            elif row[2] == 'IMAGE_ONSET':
                current_trial_start_time = int(row[1])

            # Extract IMGLOAD
            # Format: MSG    6627865 !V IMGLOAD CENTER Faces_106_h.jpg 1920 1080 1920 1080
            elif row[2] == '!V' and len(row) >= 7 and row[3] == 'IMGLOAD' and row[4] == 'CENTER':
                current_image = row[5]

            # Extract sound events (including empty sounds)
            elif row[2] in sound_markers:
                time_sound = int(row[1])
                sound_marker = row[2].replace('PLAYED_SOUND_', '').lower().split('_')

                # Track sound order within trial (before checking if empty)
                sounds_in_current_trial += 1
                sound_1_or_2 = sounds_in_current_trial

                # Check if this is an 'empty' sound presentation
                if len(sound_marker) == 1 and sound_marker[0] == 'empty':
                    # Empty sound: no actual auditory stimulus
                    sound_direction = None
                    sound_pitch = None
                    sound_nonempty = 0
                elif len(sound_marker) == 1:
                    # Direction-only marker (e.g., PLAYED_SOUND_LEFT for EX_9)
                    sound_direction = sound_marker[0]
                    sound_pitch = None
                    sound_nonempty = 1
                else:
                    # Direction + pitch marker (e.g., PLAYED_SOUND_LEFT_HIGH for EX_3/EX_7/EX_8)
                    sound_direction, sound_pitch = sound_marker[0], sound_marker[1]
                    sound_nonempty = 1

                # Use current trial number, image, and trial start time
                events.append([time_sound, sound_direction, sound_pitch, current_trial_number,
                             current_image, sound_1_or_2, current_trial_start_time, sound_nonempty])

    return events


def extract_sound_events_ex2(et_data, behavior_df, config):
    """
    Extract sound events for EX_2 (pitch discrimination with TRIAL_N_START markers).

    Parameters:
    -----------
    et_data : list
        Eye-tracking data (list of lists)
    behavior_df : pd.DataFrame or None
        Behavioral data (not used for extraction, but needed for trial filtering)
    config : dict
        Experiment configuration dictionary

    Returns:
    --------
    list : List of sound events in format [time_sound, sound_direction, sound_pitch,
                                           trial_number, image, sound_1_or_2, trial_start_time, sound_nonempty]
    """
    events = []
    sound_markers = config['sound_markers']

    # Track current trial information
    current_trial_number = None
    current_trial_start_time = None
    sounds_in_current_trial = 0

    for row in et_data:
        if len(row) >= 3 and row[0] == 'MSG':
            # Extract trial number and start time from TRIAL_N_*_START marker
            # Format: MSG    3528862 TRIAL_0_[['SSACC', 30], 'left_low']_START
            if '_START' in row[2] and row[2].startswith('TRIAL_'):
                try:
                    # Extract trial number from TRIAL_N_...
                    trial_parts = row[2].split('_')
                    current_trial_number = int(trial_parts[1])
                    current_trial_start_time = int(row[1])
                    sounds_in_current_trial = 0  # Reset sound counter for new trial
                except (ValueError, IndexError):
                    pass

            # Extract sound events
            # Sound markers: LEFT_LOW_PLAYED, LEFT_HIGH_PLAYED, RIGHT_LOW_PLAYED, RIGHT_HIGH_PLAYED
            elif row[2] in sound_markers:
                time_sound = int(row[1])

                # Parse sound marker: "LEFT_LOW_PLAYED" → direction='left', pitch='low'
                sound_marker = row[2].replace('_PLAYED', '').lower().split('_')
                if len(sound_marker) == 2:
                    sound_direction, sound_pitch = sound_marker[0], sound_marker[1]
                    sound_nonempty = 1
                else:
                    # Shouldn't happen for EX_2, but handle gracefully
                    sound_direction = None
                    sound_pitch = None
                    sound_nonempty = 0

                # Track sound order within trial
                sounds_in_current_trial += 1
                sound_1_or_2 = sounds_in_current_trial

                # EX_2 has no images
                image = None

                events.append([time_sound, sound_direction, sound_pitch, current_trial_number,
                             image, sound_1_or_2, current_trial_start_time, sound_nonempty])

    return events


def extract_sound_events_ex4(et_data, behavior_df, config):
    """
    Extract sound events for EX_4 (noise localization with TRIAL_N_START markers).

    Parameters:
    -----------
    et_data : list
        Eye-tracking data (list of lists)
    behavior_df : pd.DataFrame or None
        Behavioral data (not used for extraction, but needed for trial filtering)
    config : dict
        Experiment configuration dictionary

    Returns:
    --------
    list : List of sound events in format [time_sound, sound_direction, sound_pitch,
                                           trial_number, image, sound_1_or_2, trial_start_time, sound_nonempty]
    """
    events = []
    sound_markers = config['sound_markers']

    # Track current trial information
    current_trial_number = None
    current_trial_start_time = None
    sounds_in_current_trial = 0

    for row in et_data:
        if len(row) >= 3 and row[0] == 'MSG':
            # Extract trial number and start time from TRIAL_N_*_START marker
            # Format: MSG    2513389 TRIAL_0_[['SSACC', 0], 'right_noise']_START
            if '_START' in row[2] and row[2].startswith('TRIAL_'):
                try:
                    # Extract trial number from TRIAL_N_...
                    trial_parts = row[2].split('_')
                    current_trial_number = int(trial_parts[1])
                    current_trial_start_time = int(row[1])
                    sounds_in_current_trial = 0  # Reset sound counter for new trial
                except (ValueError, IndexError):
                    pass

            # Extract sound events
            # Sound markers: LEFT_NOISE_PLAYED, RIGHT_NOISE_PLAYED
            elif row[2] in sound_markers:
                time_sound = int(row[1])

                # Parse sound marker: "LEFT_NOISE_PLAYED" → direction='left'
                sound_marker = row[2].replace('_NOISE_PLAYED', '').lower()
                sound_direction = sound_marker  # 'left' or 'right'

                # EX_4 has no pitch
                sound_pitch = None
                sound_nonempty = 1

                # Track sound order within trial
                sounds_in_current_trial += 1
                sound_1_or_2 = sounds_in_current_trial

                # EX_4 has no images
                image = None

                events.append([time_sound, sound_direction, sound_pitch, current_trial_number,
                             image, sound_1_or_2, current_trial_start_time, sound_nonempty])

    return events


def extract_sound_events(et_data, behavior_df, config):
    """
    Unified interface for sound event extraction.
    Dispatches to experiment-specific function based on config.

    Parameters:
    -----------
    et_data : list
        Eye-tracking data (list of lists)
    behavior_df : pd.DataFrame or None
        Behavioral data containing trial information (required for EX_1, not for EX_3/EX_5/EX_7/EX_8/EX_9)
    config : dict
        Experiment configuration dictionary with 'name' key

    Returns:
    --------
    list : List of sound events in format [time_sound, sound_direction, sound_pitch,
                                           trial_number, image, sound_1_or_2, trial_start_time, sound_nonempty]

    Raises:
    -------
    ValueError : If experiment name is not recognized
    """
    if config['name'] == 'EX_1':
        return extract_sound_events_ex1(et_data, behavior_df, config)
    elif config['name'] in ('EX_3', 'EX_5', 'EX_7', 'EX_8', 'EX_9'):
        return extract_sound_events_ex3(et_data, behavior_df, config)
    elif config['name'] == 'EX_2':
        return extract_sound_events_ex2(et_data, behavior_df, config)
    elif config['name'] == 'EX_4':
        return extract_sound_events_ex4(et_data, behavior_df, config)
    else:
        raise ValueError(f"Unknown experiment: {config['name']}")


# ====================================================================
# ANSWER EVENT EXTRACTION
# ====================================================================

def extract_answer_events(et_data, config):
    """
    Extract answer events using experiment-specific markers and mapping.

    Parameters:
    -----------
    et_data : list
        Eye-tracking data (list of lists)
    config : dict
        Experiment configuration dictionary with 'answer_markers' and 'answer_mapping' keys

    Returns:
    --------
    list : List of answer events in format [time_answer, answer]
    """
    events = []
    answer_markers = config['answer_markers']
    answer_mapping = config['answer_mapping']

    for row in et_data:
        if len(row) >= 3 and row[0] == 'MSG' and row[2] in answer_markers:
            time_answer = int(row[1])
            answer = answer_mapping(row[2])
            events.append([time_answer, answer])

    return events


# ====================================================================
# REACTION TIME CALCULATION
# ====================================================================

def find_rt(sound_events, answer_events):
    """
    Matches responses to sounds and calculates reaction times.

    Parameters:
    -----------
    sound_events : list
        List of sound events in format [time_sound, sound_direction, sound_pitch,
                                        trial_number, image, sound_1_or_2, trial_start_time, sound_nonempty]
    answer_events : list
        List of answer events in format [time_answer, answer]

    Returns:
    --------
    list : List of matched events in format
           [sound_time, sound_direction, sound_pitch, response, rt, trial_number, image,
            sound_1_or_2, trial_start_time, sound_nonempty]
           where response=-1 and rt=-1 if no response was found
    """
    matched = []
    answer_idx = 0
    num_answers = len(answer_events)

    for i in range(len(sound_events)):
        sound_time = sound_events[i][0]
        sound_direction = sound_events[i][1]
        sound_pitch = sound_events[i][2]
        trial_number = sound_events[i][3]
        image = sound_events[i][4]
        sound_1_or_2 = sound_events[i][5]
        trial_start_time = sound_events[i][6]
        sound_nonempty = sound_events[i][7]
        next_sound_time = sound_events[i + 1][0] if i + 1 < len(sound_events) else float('inf')

        matched_answer = -1
        reaction_time = -1

        # Skip answers that came before this sound
        while answer_idx < num_answers and answer_events[answer_idx][0] <= sound_time:
            answer_idx += 1

        # If answer is between current and next sound, assign it
        if answer_idx < num_answers and sound_time < answer_events[answer_idx][0] < next_sound_time:
            matched_answer = answer_events[answer_idx][1]
            reaction_time = answer_events[answer_idx][0] - sound_time
            answer_idx += 1

        matched.append([sound_time, sound_direction, sound_pitch, matched_answer, reaction_time,
                       trial_number, image, sound_1_or_2, trial_start_time, sound_nonempty])

    return matched


# ====================================================================
# EYE-TRACKING EVENT PARSING
# ====================================================================

def parse_eblink(lines, eye=None):
    """
    Parses EBLINK lines and returns list of blinks: [start_time, end_time]

    Parameters:
    -----------
    lines : list
        Eye-tracking data (list of lists)
    eye : str or None
        "L" for left eye, "R" for right eye, None for both eyes

    Returns:
    --------
    list : List of blinks in format [start_time, end_time]
    """
    results = []

    for line in lines:
        if not isinstance(line, list):
            continue
        if len(line) == 0 or line[0] != "EBLINK":
            continue
        if len(line) < 4:
            continue

        # Check eye if specified
        if eye is not None and len(line) > 1 and line[1] != eye:
            continue

        try:
            start_time = int(line[2])
            end_time = int(line[3])
            results.append([start_time, end_time])
        except (ValueError, IndexError):
            continue

    return results


def parse_efix(lines, eye=None):
    """
    Parses EFIX lines and returns list of fixations: [start_time, end_time, duration]

    Parameters:
    -----------
    lines : list
        Eye-tracking data (list of lists)
    eye : str or None
        "L" for left eye, "R" for right eye, None for both eyes

    Returns:
    --------
    list : List of fixations in format [start_time, end_time, duration]
    """
    results = []

    for line in lines:
        if not isinstance(line, list):
            continue
        if len(line) == 0 or line[0] != "EFIX":
            continue
        if len(line) < 5:
            continue

        # Check eye if specified
        if eye is not None and len(line) > 1 and line[1] != eye:
            continue

        try:
            start_time = int(line[2])
            end_time = int(line[3])
            duration = int(line[4])
            results.append([start_time, end_time, duration])
        except (ValueError, IndexError):
            continue

    return results


def parse_esacc(lines, blinks=None, eye=None, max_duration=None, min_x_distance=None):
    """
    Parses ESACC lines and returns saccade data with rejection statistics.

    Parameters:
    -----------
    lines : list
        Eye-tracking data (list of lists)
    blinks : list or None
        List of blinks in format [[start_time, end_time], ...]
    eye : str or None
        "L" for left eye, "R" for right eye, None for both eyes
    max_duration : int or None
        Maximum saccade duration in ms (None = no filtering)
    min_x_distance : float or None
        Minimum horizontal distance in pixels (None = no filtering)

    Returns:
    --------
    tuple : (results, stats)
        results : list of saccades in format
                  [start_time, end_time, direction, duration, x_distance,
                   x_start, x_end, y_start, y_end, is_blink]
        stats : dict with rejection counters
    """
    results = []
    stats = {
        "rejected_duration": 0,
        "rejected_x_distance": 0,
        "rejected_bad_data": 0
    }

    if blinks is None:
        blinks = []

    for line in lines:
        if not isinstance(line, list):
            continue
        if len(line) == 0 or line[0] != "ESACC":
            continue
        if len(line) < 9:
            continue

        # Check eye if specified
        if eye is not None and len(line) > 1 and line[1] != eye:
            continue

        try:
            start_time = int(line[2])
            end_time = int(line[3])
            duration = int(line[4])
            x_start = float(line[5])
            y_start = float(line[6])
            x_end = float(line[7])
            y_end = float(line[8])
            peak_velocity = float(line[10]) if len(line) > 10 else np.nan
        except (ValueError, IndexError):
            stats["rejected_bad_data"] += 1
            continue

        direction = "right" if x_end > x_start else "left"
        x_distance = abs(x_end - x_start)

        # Filtering conditions
        if max_duration is not None and duration > max_duration:
            stats["rejected_duration"] += 1
            continue
        if min_x_distance is not None and x_distance < min_x_distance:
            stats["rejected_x_distance"] += 1
            continue

        # Check if blink overlaps with saccade (ANY overlap, not just fully contained)
        is_blink = 0
        for blink_start, blink_end in blinks:
            # Overlap if: blink_start < sacc_end AND blink_end > sacc_start
            if blink_start < end_time and blink_end > start_time:
                is_blink = 1
                break

        results.append([start_time, end_time, direction, duration, x_distance, x_start, x_end, y_start, y_end, is_blink, peak_velocity])

    return results, stats


# ====================================================================
# LONG FORMAT DATA GENERATION
# ====================================================================

def generate_long_format(id, saccades, matched, config, fixations=None, search_window=(-300, 300), output_dir=None):
    """
    Generate long format CSV for a participant with saccade-sound paired data.

    Parameters:
    -----------
    id : str
        Participant ID
    saccades : list
        List of saccades from parse_esacc
    matched : list
        List of matched sound-response events from find_rt
        Format: [sound_time, sound_direction, sound_pitch, response, rt, trial_number, image,
                 sound_1_or_2, trial_start_time]
    config : dict
        Experiment configuration dictionary
    fixations : list or None
        List of fixations from parse_efix
    search_window : tuple
        Time window around saccade start to look for sounds (min_time, max_time) in ms
    output_dir : str or None
        Output directory for long format files (if None, uses default based on config)

    Output Columns:
    ---------------
    Timing columns:
        - time_from_sacc_start_to_sound: Time from saccade start to sound (ms)
        - time_from_sacc_end_to_sound: Time from saccade end to sound (ms)
        - time_from_sound_to_sacc_start: Time from sound to saccade start (ms, negative if before)
        - time_from_sound_to_sacc_end: Time from sound to saccade end (ms)
        - time_from_RT_to_sacc_start: Time from response time to saccade start (ms)
        - time_from_trial_start_to_sacc_start: Time from trial start to saccade start (ms)

    Ordinal columns:
        - sacc_ordinal_after_sound: Ordinal number of saccade after sound (1, 2, 3, ...)
        - sacc_ordinal_after_sound_plus50: Ordinal number of saccade after sound + 50 ms (1, 2, 3, ...)
        - sacc_ordinal_before_RT: Ordinal number of saccade before RT (-1, -2, -3, ...)

    Trial/Sound info:
        - sound_1_or_2: Whether this is the first (1) or second (2) sound in the trial
        - trial_start_time: Timestamp of trial start (IMAGE_ONSET marker)

    Returns:
    --------
    None : Saves CSV file to disk
    """
    # Conversion factors: pixels to degrees of visual angle
    # Screen size: 1920 x 1080 pixels = 32.24 x 18.14 degrees
    PIX_TO_DEG_X = 32.24 / 1920  # degrees per pixel (horizontal)
    PIX_TO_DEG_Y = 18.14 / 1080  # degrees per pixel (vertical)

    if output_dir is None:
        output_dir = f'data_preprocessed/{config["name"]}/long_format'

    os.makedirs(output_dir, exist_ok=True)

    results = []

    for sacc_idx, sacc in enumerate(saccades):
        sacc_start, sacc_end, sacc_dir, sacc_duration, sacc_dist, x_start, x_end, y_start, y_end, is_blink, *_sacc_extra = sacc
        peak_velocity = _sacc_extra[0] if _sacc_extra else np.nan

        # Find preceding and following fixations
        prev_fix_duration = np.nan
        next_fix_duration = np.nan

        if fixations is not None:
            # Find preceding fixation (fixation that ends closest to but before saccade start)
            prev_fix = None
            for fix in fixations:
                fix_start, fix_end, fix_dur = fix
                if fix_end <= sacc_start:
                    prev_fix = fix
                else:
                    break  # fixations are sorted, no need to continue

            if prev_fix is not None:
                prev_fix_duration = prev_fix[2]

            # Find following fixation (fixation that starts closest to but after saccade end)
            for fix in fixations:
                fix_start, fix_end, fix_dur = fix
                if fix_start >= sacc_end:
                    next_fix_duration = fix_dur
                    break

        for sound_idx, match in enumerate(matched):
            sound_time, sound_direction, sound_pitch, response, rt, trial_number, image, sound_1_or_2, trial_start_time, sound_nonempty = match

            # Calculate time relative to saccade (RENAMED COLUMNS)
            time_from_sacc_start_to_sound = sound_time - sacc_start
            time_from_sacc_end_to_sound = sound_time - sacc_end

            # Calculate time relative to sound (inverted for sound-aligned analysis)
            time_from_sound_to_sacc_start = -time_from_sacc_start_to_sound  # Positive = saccade AFTER sound
            time_from_sound_to_sacc_end = -time_from_sacc_end_to_sound

            # NEW: Calculate time from RT to saccade start
            # RT time = sound_time + rt (if response exists)
            if response != -1 and rt != -1:
                rt_time = sound_time + rt
                time_from_RT_to_sacc_start = sacc_start - rt_time
            else:
                time_from_RT_to_sacc_start = np.nan

            # NEW: Calculate time from trial start to saccade start
            if trial_start_time is not None:
                time_from_trial_start_to_sacc_start = sacc_start - trial_start_time
            else:
                time_from_trial_start_to_sacc_start = np.nan

            if search_window[0] <= time_from_sacc_start_to_sound <= search_window[1]:
                in_saccade = sacc_start <= sound_time < sacc_end

                # Determine target based on experiment type
                target = sound_direction if config['target_field'] == 'sound_direction' else sound_pitch

                # Calculate accuracy and clean RT
                if response == -1:
                    response_given = 0
                    acc = 0  # No response = task failed
                    rt_clean = np.nan  # Use NaN for missing numerical data
                else:
                    response_given = 1
                    acc = 1 if target == response else 0
                    rt_clean = rt

                # Calculate amplitude (Euclidean distance) in pixels
                amplitude = np.sqrt(sacc_dist**2 + (y_end - y_start)**2)

                # Calculate amplitude in degrees
                x_dist_deg = sacc_dist * PIX_TO_DEG_X
                y_dist_deg = abs(y_end - y_start) * PIX_TO_DEG_Y
                amplitude_deg = np.sqrt(x_dist_deg**2 + y_dist_deg**2)

                # Build row dictionary
                row = {
                    'id_sacc': sacc_idx,
                    'time_from_sacc_start_to_sound': time_from_sacc_start_to_sound,
                    'time_from_sacc_end_to_sound': time_from_sacc_end_to_sound,
                    'time_from_sound_to_sacc_start': time_from_sound_to_sacc_start,
                    'time_from_sound_to_sacc_end': time_from_sound_to_sacc_end,
                    'time_from_RT_to_sacc_start': time_from_RT_to_sacc_start,
                    'time_from_trial_start_to_sacc_start': time_from_trial_start_to_sacc_start,
                    'in_saccade': in_saccade,
                    'sacc_direction': sacc_dir,
                    'sacc_dur': sacc_duration,
                    'peak_velocity': peak_velocity,
                    'amplitude': amplitude,
                    'amplitude_deg': amplitude_deg,
                    'x_start': x_start,
                    'x_end': x_end,
                    'y_start': y_start,
                    'y_end': y_end,
                    'x_start_deg': x_start * PIX_TO_DEG_X,
                    'x_end_deg': x_end * PIX_TO_DEG_X,
                    'y_start_deg': y_start * PIX_TO_DEG_Y,
                    'y_end_deg': y_end * PIX_TO_DEG_Y,
                    'is_blink': is_blink,
                    'prev_fix_duration': prev_fix_duration,
                    'next_fix_duration': next_fix_duration,
                    'sound_time': sound_time,
                    'sacc_start': sacc_start,
                    'sacc_end': sacc_end,
                    'id_sound': sound_idx,
                    'sound_1_or_2': sound_1_or_2,
                    'sound_nonempty': sound_nonempty,
                    'response_given': response_given,
                    'acc': acc,
                    'rt': rt_clean
                }

                # Add experiment-specific columns
                if trial_number is not None:
                    row['trial_number'] = trial_number
                if trial_start_time is not None:
                    row['trial_start_time'] = trial_start_time
                if image is not None:
                    row['image'] = image
                if sound_direction is not None:
                    row['sound_direction'] = sound_direction
                if config['has_pitch'] and sound_pitch is not None:
                    row['sound_pitch'] = sound_pitch

                results.append(row)

    df = pd.DataFrame(results)

    # POST-PROCESSING: Add ordinal numbers for saccades
    # Initialize ordinal columns
    df['sacc_ordinal_after_sound'] = np.nan
    df['sacc_ordinal_after_sound_plus50'] = np.nan
    df['sacc_ordinal_before_RT'] = np.nan

    # Group by id_sound (each sound event)
    for sound_id, group in df.groupby('id_sound'):
        # 1. Saccades AFTER sound (time_from_sound_to_sacc_start > 0)
        # These are saccades that started after the sound was played
        after_sound_mask = (df['id_sound'] == sound_id) & (df['time_from_sound_to_sacc_start'] > 0)
        after_sound_mask_plus50 = (df['id_sound'] == sound_id) & (df['time_from_sound_to_sacc_start'] > 50)
        after_sound_indices = df[after_sound_mask].sort_values('sacc_start').index
        after_sound_indices_plus50 = df[after_sound_mask_plus50].sort_values('sacc_start').index
        df.loc[after_sound_indices, 'sacc_ordinal_after_sound'] = range(1, len(after_sound_indices) + 1)
        df.loc[after_sound_indices_plus50, 'sacc_ordinal_after_sound_plus50'] = range(1, len(after_sound_indices_plus50) + 1)

        # 2. Saccades BEFORE RT (time_from_RT_to_sacc_start < 0)
        # These are saccades that started before the response time
        before_rt_mask = (df['id_sound'] == sound_id) & (df['time_from_RT_to_sacc_start'] < 0)
        before_rt_indices = df[before_rt_mask].sort_values('sacc_start').index
        # Number in reverse: last saccade before RT gets -1, second-to-last gets -2, etc.
        if len(before_rt_indices) > 0:
            df.loc[before_rt_indices, 'sacc_ordinal_before_RT'] = range(-len(before_rt_indices), 0)

    # POST-PROCESSING: Round amplitude and coordinate columns to 2 decimal places
    coordinate_cols = ['amplitude', 'amplitude_deg', 'x_start', 'x_end', 'y_start', 'y_end',
                      'x_start_deg', 'x_end_deg', 'y_start_deg', 'y_end_deg']
    for col in coordinate_cols:
        if col in df.columns:
            df[col] = df[col].round(2)

    # POST-PROCESSING: Reorder columns for better organization
    # ID columns first
    id_cols = ['trial_number', 'id_sound', 'sound_1_or_2', 'sound_nonempty', 'id_sacc']

    # Timing columns
    timing_cols = ['time_from_sacc_start_to_sound', 'time_from_sacc_end_to_sound',
                   'time_from_sound_to_sacc_start', 'time_from_sound_to_sacc_end',
                   'time_from_RT_to_sacc_start', 'time_from_trial_start_to_sacc_start',
                   'sacc_ordinal_after_sound', 'sacc_ordinal_after_sound_plus50', 'sacc_ordinal_before_RT',
                   'sound_time', 'sacc_start', 'sacc_end', 'trial_start_time', 'rt']

    # Build ordered column list
    ordered_cols = []

    # Add ID columns (only if they exist)
    for col in id_cols:
        if col in df.columns:
            ordered_cols.append(col)

    # Add timing columns (only if they exist)
    for col in timing_cols:
        if col in df.columns:
            ordered_cols.append(col)

    # Add all remaining columns
    remaining_cols = [col for col in df.columns if col not in ordered_cols]
    ordered_cols.extend(remaining_cols)

    # Reorder DataFrame
    df = df[ordered_cols]

    df.to_csv(os.path.join(output_dir, f"{id}_long.csv"), index=False)
    print(f"Saved long format for subject {id}")


def collect_all_long_files(config, input_dir=None, output_dir=None, output_file='long_format_all.csv'):
    """
    Collect all individual long format files into one aggregated file.

    Parameters:
    -----------
    config : dict
        Experiment configuration dictionary
    input_dir : str or None
        Directory containing individual long format files (if None, uses default)
    output_dir : str or None
        Output directory for aggregated file (if None, uses default)
    output_file : str
        Name of the output file (default: 'long_format_all.csv')

    Returns:
    --------
    None : Saves aggregated CSV file to disk
    """
    if input_dir is None:
        input_dir = f'data_preprocessed/{config["name"]}/long_format'
    if output_dir is None:
        output_dir = f'data_preprocessed/{config["name"]}'

    all_files = [f for f in os.listdir(input_dir) if f.endswith('_long.csv')]
    df_all = pd.DataFrame()

    for file in all_files:
        participant_id = file.replace('_long.csv', '')
        df = pd.read_csv(os.path.join(input_dir, file))
        df.insert(0, 'id_subject', participant_id)
        df_all = pd.concat([df_all, df], ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    df_all.to_csv(output_path, index=False)
    print(f"Individual files were collected into one file: {output_path}")
