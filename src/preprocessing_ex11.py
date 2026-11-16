"""
Preprocessing for EX_11 (paper Experiment 5) -- the up/down-key variant of EX_9 (Experiment 1b).

EX_9 recorded one .EDF per subject and collected the sound-direction judgement with the LEFT /
RIGHT arrows. EX_11 runs the same free-viewing paradigm TWICE per subject, once under each
UP/DOWN key mapping, on a different half of the image pool each time. This module turns those two
recordings into ONE long-format table per subject.

Everything that did not have to change is IMPORTED from `preprocessing.py` /
`preprocessing_alt.py` -- saccade and fixation parsing, trial boundaries, the row builders --
so EX_11 is assembled exactly like the other experiments.

What is new here, and nowhere else in this repository:

  1. SESSIONS. A subject is a pair of .asc files, not a file. `discover_sessions` pairs them by
     the subject ID in the filename and orders them by `session_half`, which the procedure's
     dialog fixes before the recording starts. Because the two files are separate recordings,
     THEIR TIMESTAMPS SHARE NO CLOCK: `trial_start_time`, `sound_1_time`, `sacc_start` and
     friends are only comparable within a session. Everything the analysis compares across
     sessions is a latency (a difference computed inside one trial), never a raw timestamp.

  2. ANSWER DECODING. The marker says which KEY was pressed ('up' / 'down'); the side it meant
     depends on the session's mapping. Both the RESPONSE_MAPPING marker and the filename carry
     that mapping, and `resolve_session_markers` refuses to continue when they disagree -- a
     wrong mapping silently inverts every answer in a session, which would look like a real
     (and dramatic) effect rather than like an error.

  3. TRAINING REMOVAL WITHOUT A NAMED IMAGE. `preprocessing.find_experiment_data` cuts the
     training off at the last occurrence of one hard-coded training photo. That cannot work
     here: the two mappings use DIFFERENT training-2 image sets (`training2/` for upR,
     `training2_additional/` -- new1..new5.jpg -- for upL), and the experimenter can skip either
     training with ENTER, in which case the photo never appears at all. `find_experiment_data`
     would then silently keep the training trials. `trim_to_experiment_stage` instead uses the
     structure the procedure guarantees: TRIALID restarts at 1 in every stage and the experiment
     is always the last stage, so everything from the LAST `TRIALID 1` onwards is the
     experiment. It then checks the trial count against what the config expects.

  4. NUMBERING ACROSS THE PAIR. `trial_number`, `id_sacc` and `id_sound` run over the subject's
     whole session pair, so they stay unique and `groupby(['id_subject', 'trial_number'])` --
     which every analysis does -- cannot collide two trials from different sessions.
     `trial_number_in_session` keeps the within-file value, `block_number` the 1..5 block within
     a session and `block_number_global` the 1..10 block over the visit.

One deliberate difference from EX_9, and it is not cosmetic
-----------------------------------------------------------
EX_9's notebook config lists only `PLAYED_SOUND_LEFT` / `PLAYED_SOUND_RIGHT` as sound markers,
so its empty (catch) sounds are invisible to preprocessing. EX_11 follows EX_3/EX_5/EX_7
instead and includes `PLAYED_SOUND_EMPTY`, because leaving it out has two consequences that only
show up downstream: a trial whose FIRST sound was empty has its second sound recorded as
`sound_1`, and a keypress that answered the empty sound is handed by `find_rt` to whichever real
sound preceded it. With the marker included, the empty sound occupies its own slot, carries
`sound_nonempty = 0`, and is dropped by `config.build_filter` like every other empty sound in
the project. Set `EX11_PREPROCESSING_CONFIG['sound_markers']` back to the two directional
markers for an EX_9-identical run.

As in the rest of the project: NOTHING is filtered here. Every saccade inside a trial is kept
and flagged; screening is `config.build_filter`'s job.
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from preprocessing import (
    extract_sound_events_ex3,
    find_rt,
    parse_eblink,
    parse_efix,
    parse_esacc,
)
from preprocessing_alt import (
    create_row_null_saccade,
    create_row_with_saccade,
    extract_trial_boundaries,
    find_saccades_in_trial,
    organize_sounds_by_trial,
)


# ====================================================================
# EXPERIMENT CONFIGURATION (preprocessing side)
# ====================================================================
#
# The other experiments keep their PREPROCESSING config in `01_preprocessing.ipynb` and their
# ANALYSIS config in src/config.py. EX_11's preprocessing config lives here instead, because two things need it (this module and the
# validation helpers below) and because several of its entries -- the mapping table, the block
# size, the expected trial count -- are facts about the procedure that should not be re-typed
# per notebook.

EX11_PREPROCESSING_CONFIG = {
    'name': 'EX_11',
    'description': 'Sound localization (left/right) answered with UP/DOWN, two counterbalanced '
                   'key mappings per subject',
    # EX_9's markers plus the empty (catch) sound -- see the module docstring.
    'sound_markers': ['PLAYED_SOUND_LEFT', 'PLAYED_SOUND_RIGHT', 'PLAYED_SOUND_EMPTY'],
    'answer_markers': ['ANSWER_UP', 'ANSWER_DOWN'],
    'uses_behavioral_csv': False,       # the .csv is a cross-check, not an input (as in EX_9)
    'has_pitch': False,
    'target_field': 'sound_direction',
    # The marker names a KEY, not a side. Decoding to a side needs the session's mapping and
    # happens in `decode_answer_key`, so this stops at the key.
    'answer_mapping': lambda marker: marker.replace('ANSWER_', '').lower(),
    'training_removal': 'last_stage',   # see `trim_to_experiment_stage`
    'response_mappings': ('upL', 'upR'),
    'session_halves': (1, 2),
    'trials_per_session': 100,          # half of the 200-image pool
    'trials_per_block': 20,             # break + drift check every 20 experiment trials
    'asc_directory': 'data_raw/EX_11/asc',
    'edf_directory': 'data_raw/EX_11/edf',
    'csv_directory': 'data_raw/EX_11/csv',
    'long_format_dir': 'long_format_ex11',
    'long_format_suffix': '_long_ex11.csv',
    'long_format_file': 'long_format_ex11_all.csv',
}

# UP means this side, under this mapping. The single source of truth for the decoding; both
# `decode_answer_key` and `encode_answer_key` read it, so they cannot drift apart.
UP_MEANS = {'upL': 'left', 'upR': 'right'}
DOWN_MEANS = {'upL': 'right', 'upR': 'left'}


def _is_missing(value) -> bool:
    """True for None / NaN, false for everything else. Keeps the guards below to one line."""
    return value is None or (isinstance(value, float) and np.isnan(value))


def decode_answer_key(key, mapping: str) -> Optional[str]:
    """
    Turn a pressed key into the side it stood for, under one session's mapping.

    Parameters:
    -----------
    key : str or None
        'up' or 'down'; None / NaN passes through
    mapping : str
        'upL' (UP = left) or 'upR' (UP = right)

    Returns:
    --------
    str or None : 'left' / 'right'
    """
    if _is_missing(key):
        return None
    if mapping not in UP_MEANS:
        raise ValueError(f"Unknown response mapping {mapping!r}; expected one of {sorted(UP_MEANS)}.")
    key = str(key).lower()
    if key == 'up':
        return UP_MEANS[mapping]
    if key == 'down':
        return DOWN_MEANS[mapping]
    raise ValueError(f"Unknown response key {key!r}; expected 'up' or 'down'.")


def encode_answer_key(side, mapping: str) -> Optional[str]:
    """
    The inverse of `decode_answer_key`: which key produced this side under this mapping.

    Used to put `answer_key` back on the finished rows. The key is what a keypress bias lives
    on, and it is NOT recoverable from a side alone -- 'left' is the UP key in one of a
    subject's two sessions and the DOWN key in the other.

    Parameters:
    -----------
    side : str or None
        'left' / 'right'; None / NaN passes through
    mapping : str
        'upL' or 'upR'

    Returns:
    --------
    str or None : 'up' / 'down'
    """
    if _is_missing(side):
        return None
    if mapping not in UP_MEANS:
        raise ValueError(f"Unknown response mapping {mapping!r}; expected one of {sorted(UP_MEANS)}.")
    side = str(side).lower()
    if side == UP_MEANS[mapping]:
        return 'up'
    if side == DOWN_MEANS[mapping]:
        return 'down'
    raise ValueError(f"Unknown response side {side!r}; expected 'left' or 'right'.")


# ====================================================================
# SESSIONS: PAIRING THE TWO RECORDINGS OF ONE SUBJECT
# ====================================================================
#
# The experiment script names the recording it pulls off the Host as
#     <participant ID>_<mapping>_half<n>.EDF
# e.g. E11001_upL_half1.EDF, and the .asc keeps that stem. The name carries no
# timestamp -- the EDF is downloaded under a name the script composes, not under the dialog's
# date string -- so the ORDER comes from `session_half` instead (see `discover_sessions`).

SESSION_FILENAME = re.compile(
    r'^(?P<subject>[A-Za-z]+\d+)'
    r'_(?P<mapping>upL|upR)'
    r'_half(?P<half>\d+)$'
)

# As written by the experiment script:
#     beh_<ID>_<age>_<gender>_<YYYY_MM_DD_HH_MM_SS>_<mapping>_half<n>.csv
# As published in the data deposit, without demographics or the test date (as for the other
# experiments' behavioural logs):
#     beh_<ID>_<mapping>_half<n>.csv
# Both are accepted. Without the date, `discover_sessions` cannot check the session order against
# the recording times and relies on `session_half` alone; the decoding check is unaffected.
BEHAVIOURAL_FILENAME = re.compile(
    r'^beh_(?P<subject>[A-Za-z]+\d+)'
    r'(?:_(?P<age>[^_]+)_(?P<gender>[^_]+)_'
    r'(?P<date>\d{4}_\d{2}_\d{2})_(?P<time>\d{2}_\d{2}_\d{2}))?'
    r'_(?P<mapping>upL|upR)_half(?P<half>\d+)$'
)


def parse_session_filename(path: str) -> Dict:
    """
    Pull subject, key mapping and image half out of an EX_11 filename.

    Parameters:
    -----------
    path : str
        Path or bare filename of an .asc / .EDF

    Returns:
    --------
    dict : file, path, id_subject, mapping, session_half

    Raises:
    -------
    ValueError : on any name that does not fit the pattern. Deliberate: an unrecognised name is
        a file we cannot assign a subject OR a mapping to, and both are load-bearing.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    match = SESSION_FILENAME.match(stem)
    if match is None:
        raise ValueError(
            f"{stem!r} does not look like an EX_11 recording. Expected "
            f"<ID>_upL|upR_half<n>, e.g. E11001_upL_half1."
        )
    return {
        'file': stem,
        'path': path,
        'id_subject': match['subject'],
        'mapping': match['mapping'],
        'session_half': int(match['half']),
    }


def behavioural_session_times(csv_directory: Optional[str] = None,
                              cfg: Optional[Dict] = None) -> Dict[Tuple[str, str, int], datetime]:
    """
    Wall-clock start time of each session, read from the behavioural .csv filename.

    The .csv is NOT a preprocessing input for EX_11 (as in EX_9), but its name is the only
    record of WHEN a session ran, and that is what `discover_sessions` checks its ordering
    against. Missing files are not an error here -- the caller decides what to do with a
    session that has no time.

    Returns:
    --------
    dict : (id_subject, mapping, session_half) -> datetime
    """
    cfg = cfg or EX11_PREPROCESSING_CONFIG
    csv_directory = csv_directory or cfg['csv_directory']

    times: Dict[Tuple[str, str, int], datetime] = {}
    if not os.path.isdir(csv_directory):
        return times

    for name in sorted(os.listdir(csv_directory)):
        if not name.lower().endswith('.csv'):
            continue
        match = BEHAVIOURAL_FILENAME.match(os.path.splitext(name)[0])
        if match is None or match['date'] is None:      # no date in a deposited file name
            continue
        key = (match['subject'], match['mapping'], int(match['half']))
        times[key] = datetime.strptime(f"{match['date']}_{match['time']}",
                                       '%Y_%m_%d_%H_%M_%S')
    return times


def discover_sessions(asc_directory: Optional[str] = None,
                      exclude_subjects: Tuple[str, ...] = (),
                      cfg: Optional[Dict] = None,
                      csv_directory: Optional[str] = None) -> Dict[str, List[Dict]]:
    """
    Group the .asc files in a directory into one ordered session list per subject.

    Order is by `session_half`, not by filename sort order: the half is chosen in the dialog
    before the recording starts and a subject always runs half 1 first, so it IS the position in
    the visit. Because that is an assumption about how the sessions were run rather than
    something the .asc states, it is CHECKED against the behavioural .csv timestamps whenever
    those exist -- a subject whose half 2 was recorded before their half 1 is reported loudly
    rather than silently mis-ordered, since `mapping_order` is the whole of the order analysis.

    Parameters:
    -----------
    asc_directory : str or None
        Defaults to the config's `asc_directory`
    exclude_subjects : tuple of str
        Subject IDs to leave out entirely (test recordings, withdrawals)
    cfg : dict or None
        Defaults to EX11_PREPROCESSING_CONFIG
    csv_directory : str or None
        Where to look for the behavioural .csv timestamps; None uses the config's

    Returns:
    --------
    dict : id_subject -> list of session dicts, ordered, each carrying mapping_order (1-based),
           mapping_first, is_first_mapping, n_sessions, recorded_at (or None)
    """
    cfg = cfg or EX11_PREPROCESSING_CONFIG
    asc_directory = asc_directory or cfg['asc_directory']
    times = behavioural_session_times(csv_directory, cfg)

    sessions: Dict[str, List[Dict]] = {}
    for name in sorted(os.listdir(asc_directory)):
        if not name.endswith('.asc'):
            continue
        session = parse_session_filename(os.path.join(asc_directory, name))
        if session['id_subject'] in exclude_subjects:
            continue
        session['recorded_at'] = times.get(
            (session['id_subject'], session['mapping'], session['session_half']))
        sessions.setdefault(session['id_subject'], []).append(session)

    for subject, subject_sessions in sessions.items():
        subject_sessions.sort(key=lambda s: s['session_half'])
        first_mapping = subject_sessions[0]['mapping']
        for order, session in enumerate(subject_sessions, start=1):
            session['mapping_order'] = order
            session['mapping_first'] = first_mapping
            session['is_first_mapping'] = int(order == 1)
            session['n_sessions'] = len(subject_sessions)

        stamps = [s['recorded_at'] for s in subject_sessions]
        if all(t is not None for t in stamps) and stamps != sorted(stamps):
            raise ValueError(
                f"{subject}: the session halves are not in recording order -- "
                f"{[(s['session_half'], s['recorded_at'].isoformat()) for s in subject_sessions]}. "
                f"`mapping_order` is derived from `session_half`, so this subject would be "
                f"scored as having done the wrong mapping first. Check the .csv timestamps "
                f"before continuing."
            )

    return dict(sorted(sessions.items()))


def describe_sessions(sessions: Dict[str, List[Dict]],
                      cfg: Optional[Dict] = None) -> pd.DataFrame:
    """
    One row per recording: who, which mapping, which image half, in which order, when.

    Also the place the counterbalancing is checked. A subject with one session, or with the same
    mapping twice, cannot contribute to any mapping or order comparison and has to be seen
    before anything is computed rather than after.

    Returns:
    --------
    pd.DataFrame : id_subject, mapping_order, response_mapping, session_half, mapping_first,
                   recorded_at, file, problem
    """
    cfg = cfg or EX11_PREPROCESSING_CONFIG
    expected_mappings = set(cfg['response_mappings'])
    expected_halves = set(cfg['session_halves'])

    rows = []
    for subject, subject_sessions in sessions.items():
        found_mappings = [s['mapping'] for s in subject_sessions]
        found_halves = [s['session_half'] for s in subject_sessions]
        problem = ''
        if len(subject_sessions) != len(expected_mappings):
            problem = f'*** {len(subject_sessions)} session(s), expected {len(expected_mappings)} ***'
        elif set(found_mappings) != expected_mappings:
            problem = f'*** mappings {found_mappings}, expected one of each ***'
        elif set(found_halves) != expected_halves:
            problem = f'*** image halves {found_halves}, expected one of each ***'
        for session in subject_sessions:
            rows.append({
                'id_subject': subject,
                'mapping_order': session['mapping_order'],
                'response_mapping': session['mapping'],
                'session_half': session['session_half'],
                'mapping_first': session['mapping_first'],
                'recorded_at': session['recorded_at'],
                'file': session['file'],
                'problem': problem,
            })
    return (pd.DataFrame(rows)
            .sort_values(['id_subject', 'mapping_order'])
            .reset_index(drop=True))


# ====================================================================
# READING ONE .asc
# ====================================================================

def parse_asc(path: str) -> List[List]:
    """
    Read an .asc into the list-of-token-lists the parsers expect.

    Same split as `01_preprocessing.ipynb`. EX_11 has no bracketed trial markers, so there is
    nothing to re-join.

    Returns:
    --------
    list : one list of whitespace-separated tokens per line
    """
    et_data = []
    with open(path, 'r') as handle:
        for line in handle:
            et_data.append(line.strip().split())
    return et_data


def trim_to_experiment_stage(et_data: List[List], cfg: Optional[Dict] = None,
                             verbose: bool = True) -> List[List]:
    """
    Drop training 1 and training 2, keeping only the experiment stage.

    The rule is structural, not image-based: the procedure sends `TRIALID 1` at the start of
    every stage -- training 1, each training 2 attempt, then the experiment -- and the
    experiment is always last, so the LAST `TRIALID 1` opens it. That survives the two things
    `preprocessing.find_experiment_data` does not (see the module docstring): the per-mapping
    training-2 image sets, and either training being skipped with ENTER.

    The trial count is then checked against `trials_per_session`. A mismatch means the stage
    boundary was found in the wrong place or the session was aborted, and both have to be seen
    before the numbers are used -- so it warns rather than passing silently.

    Parameters:
    -----------
    et_data : list
        Output of `parse_asc`
    cfg : dict or None
        Defaults to EX11_PREPROCESSING_CONFIG
    verbose : bool
        Print where the cut was made and how many trials survived

    Returns:
    --------
    list : et_data from the experiment's first TRIALID onwards

    Raises:
    -------
    ValueError : if no TRIALID marker exists at all
    """
    cfg = cfg or EX11_PREPROCESSING_CONFIG

    starts = [i for i, row in enumerate(et_data)
              if len(row) >= 4 and row[0] == 'MSG' and row[2] == 'TRIALID' and row[3] == '1']
    if not starts:
        raise ValueError(
            "No 'TRIALID 1' marker in this recording. Every stage of the procedure opens with "
            "one, so a file without any is not an EX_11 recording (or the conversion dropped "
            "its messages)."
        )

    trimmed = et_data[starts[-1]:]
    n_trials = len({row[3] for row in trimmed
                    if len(row) >= 4 and row[0] == 'MSG' and row[2] == 'TRIALID'})

    if verbose:
        print(f"  Training removed: experiment starts at the last of {len(starts)} "
              f"'TRIALID 1' markers (line {starts[-1]}), {n_trials} trials follow")
    expected = cfg.get('trials_per_session')
    if expected is not None and n_trials != expected:
        print(f"  WARNING: {n_trials} experiment trials, expected {expected}. Either the "
              f"session was aborted or the stage boundary is wrong -- check before using.")

    return trimmed


def resolve_session_markers(et_data: List[List], session: Dict) -> Tuple[str, int]:
    """
    Agree on one response mapping and one image half for a session, from markers and filename.

    Two independent records of the same two facts, and they must agree. If they do not, the
    session is refused rather than resolved by preference: the mapping decides what every answer
    in the session MEANS, so guessing wrong inverts 100 trials of data into a large, clean,
    entirely artefactual effect. Missing markers are equally fatal -- 'upL' is not a safe
    default.

    Parameters:
    -----------
    et_data : list
        Output of `parse_asc` (before or after trimming; the markers are sent per trial)
    session : dict
        From `discover_sessions`

    Returns:
    --------
    tuple : (mapping, session_half)
    """
    mappings, halves = set(), set()
    for row in et_data:
        if len(row) >= 4 and row[0] == 'MSG':
            if row[2] == 'RESPONSE_MAPPING':
                mappings.add(row[3])
            elif row[2] == 'SESSION_HALF':
                halves.add(row[3])

    if not mappings:
        raise ValueError(
            f"{session['file']}: no RESPONSE_MAPPING marker anywhere in the recording. Without "
            f"it the .asc does not say whether UP meant left or right, and every answer in the "
            f"session is unreadable."
        )
    if len(mappings) > 1:
        raise ValueError(
            f"{session['file']}: the RESPONSE_MAPPING marker changes within the session "
            f"({sorted(mappings)}). One recording is one mapping by construction "
            f"(research_et_main_upL.py / _upR.py set it once), so this file is not what it claims."
        )
    marker_mapping = mappings.pop()
    if marker_mapping != session['mapping']:
        raise ValueError(
            f"{session['file']}: the filename says mapping {session['mapping']!r} but the "
            f"RESPONSE_MAPPING marker says {marker_mapping!r}. The file was probably renamed or "
            f"copied over another; do not guess which is right."
        )

    if not halves:
        raise ValueError(
            f"{session['file']}: no SESSION_HALF marker. The half is what says which images this "
            f"session drew from, and `mapping_order` is derived from it."
        )
    if len(halves) > 1:
        raise ValueError(
            f"{session['file']}: the SESSION_HALF marker changes within the session "
            f"({sorted(halves)}). One recording is one half by construction."
        )
    marker_half = int(halves.pop())
    if marker_half != session['session_half']:
        raise ValueError(
            f"{session['file']}: the filename says half {session['session_half']} but the "
            f"SESSION_HALF marker says {marker_half}."
        )

    return marker_mapping, marker_half


def process_session(session: Dict, cfg: Optional[Dict] = None, eye: Optional[str] = 'R',
                    drop_training: bool = True) -> Dict:
    """
    Parse one .asc into matched sound/response events plus its eye events.

    Stops short of the long format on purpose: rows cannot be written until both of a subject's
    sessions are known, because `trial_number`, `id_sacc` and `id_sound` run across the pair.

    The chain is EX_9's, with exactly one step inserted -- the keys are decoded to sides with
    this session's mapping BEFORE `find_rt` sees them, so `sound_N_response`, `sound_N_acc` and
    `consistent_with_*` mean in EX_11 precisely what they mean in EX_9 and every downstream
    analysis carries over unchanged.

    Parameters:
    -----------
    session : dict
        From `discover_sessions`
    cfg : dict or None
        Defaults to EX11_PREPROCESSING_CONFIG
    eye : str or None
        'L', 'R', or None for both
    drop_training : bool
        Drop the training stages (see `trim_to_experiment_stage`)

    Returns:
    --------
    dict : the session dict plus matched, saccades, fixations, blinks, et_data_trimmed
    """
    cfg = cfg or EX11_PREPROCESSING_CONFIG
    print(f"  {session['file']}  ({session['mapping']}, half {session['session_half']}, "
          f"session {session['mapping_order']} of {session['n_sessions']})")

    et_data = parse_asc(session['path'])
    mapping, _half = resolve_session_markers(et_data, session)
    if drop_training:
        et_data = trim_to_experiment_stage(et_data, cfg)

    sound_events = extract_sound_events_ex3(et_data, None, cfg)

    # The one EX_11-specific step: ANSWER_UP/ANSWER_DOWN name a key, so decode it to a side
    # with this session's mapping before anything compares it to a sound.
    to_key = cfg['answer_mapping']
    answer_markers = set(cfg['answer_markers'])
    answer_events = [[int(row[1]), decode_answer_key(to_key(row[2]), mapping)]
                     for row in et_data
                     if len(row) >= 3 and row[0] == 'MSG' and row[2] in answer_markers]

    matched = find_rt(sound_events, answer_events)
    n_nonempty = sum(1 for event in sound_events if event[7] == 1)
    print(f"    {len(sound_events)} sounds ({n_nonempty} non-empty), "
          f"{len(answer_events)} keypresses")

    blinks = parse_eblink(et_data, eye=eye)
    fixations = parse_efix(et_data, eye=eye)
    # ALL saccades: max_duration=None / min_x_distance=None disable parse_esacc's own filtering.
    saccades, _stats = parse_esacc(et_data, blinks=blinks, eye=eye,
                                   max_duration=None, min_x_distance=None)
    print(f"    {len(blinks)} blinks, {len(fixations)} fixations, "
          f"{len(saccades)} saccades (unfiltered)")

    out = dict(session)
    out.update({'mapping': mapping, 'matched': matched, 'saccades': saccades,
                'fixations': fixations, 'blinks': blinks, 'et_data': et_data})
    return out


# ====================================================================
# LONG FORMAT, ACROSS THE SESSION PAIR
# ====================================================================

# Conversion factors: pixels to degrees of visual angle. Copied from
# `preprocessing_alt.generate_long_format_alt` rather than imported, because they are local
# constants there; EX_11 was recorded on the same rig as EX_9, so they are the same numbers.
PIX_TO_DEG_X = 32.24 / 1920
PIX_TO_DEG_Y = 18.14 / 1080


def _session_fields(session: Dict, trial_number_in_session: int, cfg: Dict) -> Dict:
    """The session-, mapping- and block-level fields every row of EX_11 carries."""
    trials_per_block = cfg['trials_per_block']
    return {
        'response_mapping': session['mapping'],
        'session_half': session['session_half'],
        'mapping_order': session['mapping_order'],
        'mapping_first': session['mapping_first'],
        'is_first_mapping': session['is_first_mapping'],
        'session_file': session['file'],
        'session_recorded_at': session['recorded_at'],
        'trial_number_in_session': trial_number_in_session,
        'block_number': (trial_number_in_session - 1) // trials_per_block + 1,
    }


def generate_long_format_ex11(id_subject: str, sessions: List[Dict],
                              cfg: Optional[Dict] = None,
                              output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Build one subject's long format from BOTH of their sessions: one row per saccade.

    The EX_9 row builders are reused exactly as they are (`create_row_with_saccade` /
    `create_row_null_saccade`), and the session columns are added on top, so an EX_11 row is an
    EX_9 row plus the mapping/half/block information and nothing is quietly redefined.

    Numbering across the pair:
      trial_number   1..2N over both sessions in recording order, so it is unique per subject
                     and `groupby(['id_subject', 'trial_number'])` cannot collide two trials
                     from different sessions. It is set BEFORE the row builders run, so
                     `id_sound = (trial_number - 1) * 2 + closest_sound` comes out global too.
      id_sacc        offset by the earlier session's saccade count, for the same reason.
      block_number_global  the 10 blocks of the visit, 1..10 in the order they were run.

    What is deliberately NOT made global: the raw timestamps. The two recordings have separate
    tracker clocks, so `trial_start_time`, `sound_1_time`, `sacc_start` etc. are within-session
    values. Every analysis quantity is a latency computed inside a trial, so this never bites,
    but do not diff two timestamps from different `session_file`s.

    Parameters:
    -----------
    id_subject : str
        Participant ID
    sessions : list
        `process_session` results for that subject, in any order (sorted here by mapping_order)
    cfg : dict or None
        Defaults to EX11_PREPROCESSING_CONFIG
    output_dir : str or None
        Defaults to data_preprocessed/<name>/<long_format_dir>

    Returns:
    --------
    pd.DataFrame : the frame that was written to disk
    """
    cfg = cfg or EX11_PREPROCESSING_CONFIG
    if output_dir is None:
        output_dir = f'data_preprocessed/{cfg["name"]}/{cfg["long_format_dir"]}'
    os.makedirs(output_dir, exist_ok=True)

    results = []
    trial_offset, sacc_offset = 0, 0

    for session in sorted(sessions, key=lambda s: s['mapping_order']):
        boundaries = {number: (start, end)
                      for number, start, end in extract_trial_boundaries(session['et_data'])}
        trials_dict = organize_sounds_by_trial(session['matched'])
        n_trials_in_session = 0

        for trial_number_in_session in sorted(trials_dict.keys()):
            trial_info = dict(trials_dict[trial_number_in_session])
            n_trials_in_session = max(n_trials_in_session, trial_number_in_session)

            if trial_number_in_session not in boundaries:
                print(f"  Warning: trial {trial_number_in_session} of {session['file']} has no "
                      f"IMAGE_ONSET/OFFSET pair. Skipping.")
                continue
            trial_start_time, trial_end_time = boundaries[trial_number_in_session]

            # Renumber BEFORE building rows: the builders derive id_sound from trial_number.
            trial_info['trial_number'] = trial_offset + trial_number_in_session

            sound_1 = trial_info['sound_1']
            sound_2 = trial_info['sound_2']
            if sound_1 is None or sound_2 is None:
                print(f"  Warning: trial {trial_number_in_session} of {session['file']} does not "
                      f"have both sounds (sound 1: {sound_1 is not None}, "
                      f"sound 2: {sound_2 is not None})")

            session_fields = _session_fields(session, trial_number_in_session, cfg)
            saccades_in_trial = find_saccades_in_trial(session['saccades'],
                                                       trial_start_time, trial_end_time)

            if len(saccades_in_trial) == 0:
                # Two rows, one per sound -- the EX_9 convention for a trial with no saccade.
                for which_sound in (1, 2):
                    row = create_row_null_saccade(trial_info, sound_1, sound_2, trial_start_time,
                                                  trial_end_time, cfg, which_sound=which_sound)
                    row.update(session_fields)
                    results.append(row)
            else:
                n_saccades = len(saccades_in_trial)
                for ordinal, (sacc_idx, sacc) in enumerate(saccades_in_trial, start=1):
                    row = create_row_with_saccade(
                        trial_info, sound_1, sound_2, trial_start_time, trial_end_time,
                        sacc, sacc_offset + sacc_idx, ordinal, n_saccades,
                        session['fixations'], cfg, PIX_TO_DEG_X, PIX_TO_DEG_Y)
                    row.update(session_fields)
                    results.append(row)

        trial_offset += n_trials_in_session
        sacc_offset += len(session['saccades'])

    df = pd.DataFrame(results)
    df.insert(0, 'id_subject', id_subject)

    # Which key produced each answer. Not recoverable from the side alone -- 'left' is the UP
    # key in one of the subject's two sessions and the DOWN key in the other -- and it is the
    # axis a keypress bias lives on, so it is stored rather than re-derived per analysis.
    for which_sound in (1, 2):
        response_col = f'sound_{which_sound}_response'
        if response_col in df.columns:
            df[f'sound_{which_sound}_answer_key'] = [
                encode_answer_key(response, mapping)
                for response, mapping in zip(df[response_col], df['response_mapping'])
            ]

    # Blocks ranked over the whole visit: (which session, which block in it) -> 1, 2, ... 10
    block_order = (df[['mapping_order', 'block_number']]
                   .drop_duplicates()
                   .sort_values(['mapping_order', 'block_number'])
                   .reset_index(drop=True))
    block_order['block_number_global'] = block_order.index + 1
    df = df.merge(block_order, on=['mapping_order', 'block_number'], how='left')

    coordinate_cols = ['amplitude', 'amplitude_deg', 'x_start', 'x_end', 'y_start', 'y_end',
                       'x_start_deg', 'x_end_deg', 'y_start_deg', 'y_end_deg']
    for col in coordinate_cols:
        if col in df.columns:
            df[col] = df[col].round(2)

    df = df[get_ordered_columns_ex11(df)]

    output_path = os.path.join(output_dir, f"{id_subject}{cfg['long_format_suffix']}")
    df.to_csv(output_path, index=False)
    print(f"  Saved {output_path} ({len(df)} rows, {df['trial_number'].nunique()} trials, "
          f"{df['session_file'].nunique()} sessions)")
    return df


def get_ordered_columns_ex11(df: pd.DataFrame) -> List[str]:
    """
    Order columns: identity -> session/mapping/block -> trial -> saccade -> timing -> sounds.

    The session block sits second, right after the identity, because in EX_11 it is what a row
    has to be read against -- an answer of 'left' means nothing without its mapping. Columns not
    listed keep their existing order at the end, so a new field from the shared row builders
    still appears.
    """
    base_cols = [
        # identity
        'id_subject', 'trial_number',
        # session / mapping / block -- EX_11's own
        'response_mapping', 'session_half', 'mapping_order', 'mapping_first',
        'is_first_mapping', 'block_number', 'block_number_global',
        'trial_number_in_session', 'session_file', 'session_recorded_at',
        # trial
        'image', 'trial_start_time', 'trial_end_time', 'trial_duration',
        # saccade identity
        'id_sacc', 'sacc_ordinal_in_trial', 'n_saccades_in_trial', 'sacc_start', 'sacc_end',
        # saccade properties
        'sacc_direction', 'sacc_dur', 'amplitude', 'amplitude_deg', 'peak_velocity',
        'x_start', 'x_end', 'y_start', 'y_end',
        'x_start_deg', 'x_end_deg', 'y_start_deg', 'y_end_deg',
        'is_blink', 'prev_fix_duration', 'next_fix_duration',
        # timing
        'time_from_trial_start_to_sacc_start', 'time_from_trial_end_to_sacc_start',
        'time_from_sound_1_to_sacc_start', 'time_from_sound_1_to_sacc_end',
        'time_from_sacc_start_to_sound_1', 'time_from_sacc_end_to_sound_1',
        'time_from_sound_2_to_sacc_start', 'time_from_sound_2_to_sacc_end',
        'time_from_sacc_start_to_sound_2', 'time_from_sacc_end_to_sound_2',
        'closest_sound', 'id_sound',
        'time_from_closest_sound_to_sacc_start', 'time_from_closest_sound_to_sacc_end',
        'time_from_sacc_start_to_closest_sound', 'time_from_sacc_end_to_closest_sound',
        'in_saccade_sound_1', 'in_saccade_sound_2', 'in_saccade_closest',
        'consistent_with_sound_1', 'consistent_with_sound_2', 'consistent_with_closest',
        # sounds and responses
        'sound_1_time', 'sound_1_direction', 'sound_1_nonempty', 'sound_1_answer_key',
        'sound_1_response', 'sound_1_response_given', 'sound_1_acc', 'sound_1_rt',
        'sound_2_time', 'sound_2_direction', 'sound_2_nonempty', 'sound_2_answer_key',
        'sound_2_response', 'sound_2_response_given', 'sound_2_acc', 'sound_2_rt',
    ]
    ordered = [c for c in base_cols if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    return ordered


def collect_all_long_files_ex11(cfg: Optional[Dict] = None, input_dir: Optional[str] = None,
                                output_dir: Optional[str] = None,
                                output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Aggregate the per-subject long formats into one file.

    Carries over the warning from `preprocessing_alt.collect_all_long_files_alt`: this globs
    whatever is on disk, whether or not the subject's .asc still exists. A stale per-subject
    file once inflated EX_5 from 40 subjects to 41. If you drop a subject, delete their
    preprocessed file too, and check the counts printed below against the .asc count.

    Unlike the alt collector, `id_subject` is NOT taken from the filename here -- it is already a
    column, written by `generate_long_format_ex11`, because an EX_11 file is built from two
    recordings and its name is a subject rather than a recording.

    Returns:
    --------
    pd.DataFrame
    """
    cfg = cfg or EX11_PREPROCESSING_CONFIG
    if input_dir is None:
        input_dir = f'data_preprocessed/{cfg["name"]}/{cfg["long_format_dir"]}'
    if output_dir is None:
        output_dir = f'data_preprocessed/{cfg["name"]}'
    if output_file is None:
        output_file = cfg['long_format_file']

    suffix = cfg['long_format_suffix']
    all_files = sorted(f for f in os.listdir(input_dir) if f.endswith(suffix))
    df_all = pd.concat([pd.read_csv(os.path.join(input_dir, f)) for f in all_files],
                       ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    df_all.to_csv(output_path, index=False)

    print(f"Collected {len(all_files)} files into {output_path}")
    print(f"Total: {len(df_all):,} rows, {df_all['id_subject'].nunique()} subjects, "
          f"{df_all['session_file'].nunique()} sessions, "
          f"{df_all.groupby('id_subject')['trial_number'].nunique().sum():,} trials")
    return df_all


# ====================================================================
# EDF -> ASC
# ====================================================================

def convert_edf_files(edf_directory: Optional[str] = None, asc_directory: Optional[str] = None,
                      cfg: Optional[Dict] = None, edf2asc: Optional[str] = None,
                      overwrite: bool = False) -> List[str]:
    """
    Run SR Research's edf2asc over the recordings that do not have an .asc yet.

    Kept in the module rather than left as a shell step so that the conversion settings are
    recorded with the pipeline: DEFAULT FLAGS, i.e. events AND samples, matching how every other
    .asc in this study was produced. Changing the flags changes what `parse_esacc` and `parse_efix`
    can see, so they are not exposed as options. The output keeps the .EDF's exact base name.

    Parameters:
    -----------
    edf_directory, asc_directory : str or None
        Default to the config's directories
    cfg : dict or None
        Defaults to EX11_PREPROCESSING_CONFIG
    edf2asc : str or None
        Path to edf2asc.exe. None searches PATH and the standard SR Research install.
    overwrite : bool
        Re-convert files that already have an .asc

    Returns:
    --------
    list : paths of the .asc files that were written
    """
    import glob as _glob
    import shutil
    import subprocess

    cfg = cfg or EX11_PREPROCESSING_CONFIG
    edf_directory = edf_directory or cfg['edf_directory']
    asc_directory = asc_directory or cfg['asc_directory']

    if edf2asc is None:
        edf2asc = shutil.which('edf2asc') or \
            r'C:\Program Files (x86)\SR_Research\EyeLink\bin\edf2asc.exe'
    if not os.path.exists(edf2asc):
        raise FileNotFoundError(
            f"edf2asc not found at {edf2asc!r}. Install SR Research's EyeLink Developers Kit, "
            f"or pass edf2asc=<path>."
        )

    os.makedirs(asc_directory, exist_ok=True)
    written = []
    edf_files = sorted(_glob.glob(os.path.join(edf_directory, '*.EDF')))
    print(f"{len(edf_files)} .EDF in {edf_directory}")

    for edf_path in edf_files:
        stem = os.path.splitext(os.path.basename(edf_path))[0]
        asc_path = os.path.join(asc_directory, stem + '.asc')
        if os.path.exists(asc_path) and not overwrite:
            print(f"  skip (already converted): {stem}")
            continue
        # edf2asc returns a non-zero code even on success, so the written file is the test.
        subprocess.run([edf2asc, edf_path, '-p', asc_directory], capture_output=True, text=True)
        if not os.path.exists(asc_path):
            raise RuntimeError(f"edf2asc produced no .asc for {stem}")
        print(f"  converted {stem}  ({os.path.getsize(asc_path) / 1e6:.1f} MB)")
        written.append(asc_path)

    return written


# ====================================================================
# CROSS-CHECK AGAINST THE BEHAVIOURAL .csv
# ====================================================================

def load_behavioural_trials(cfg: Optional[Dict] = None,
                            csv_directory: Optional[str] = None) -> pd.DataFrame:
    """
    Read every behavioural .csv into one experiment-trials-only table.

    The procedure writes the same facts twice -- once into the EDF as markers, once into this
    .csv -- and EX_11's decoding is exactly the kind of thing that benefits from a second
    witness. It is NOT a preprocessing input (`uses_behavioral_csv` is False, as in EX_9); it
    exists so `check_against_behavioural` can compare.

    Returns:
    --------
    pd.DataFrame : the 'experiment' rows of every .csv, plus id_subject / response_mapping /
                   session_half / session_file columns taken from the filename
    """
    cfg = cfg or EX11_PREPROCESSING_CONFIG
    csv_directory = csv_directory or cfg['csv_directory']

    frames = []
    for name in sorted(os.listdir(csv_directory)):
        if not name.lower().endswith('.csv'):
            continue
        match = BEHAVIOURAL_FILENAME.match(os.path.splitext(name)[0])
        if match is None:
            print(f"  skipping unrecognised behavioural file: {name}")
            continue
        # The questions are Polish text written by PsychoPy on a Windows codepage.
        frame = pd.read_csv(os.path.join(csv_directory, name), encoding='latin1')
        frame['id_subject'] = match['subject']
        frame['response_mapping'] = match['mapping']
        frame['session_half'] = int(match['half'])
        frame['session_file'] = f"{match['subject']}_{match['mapping']}_half{match['half']}"
        frames.append(frame[frame['trial_type'] == 'experiment'])

    return pd.concat(frames, ignore_index=True)


def check_against_behavioural(df_all: pd.DataFrame, cfg: Optional[Dict] = None,
                              csv_directory: Optional[str] = None) -> pd.DataFrame:
    """
    Compare the decoded long format against the procedure's own .csv, session by session.

    The check that has to pass before any EX_11 number is worth reading. Two things are compared,
    per session and per sound:

      sounds   the .csv's `first_sound_type` / `second_sound_type` against `sound_N_direction`
      keys     the .csv's `first_answer` / `second_answer` -- raw 'up'/'down' -- against the key
               this pipeline says produced its decoded side (`sound_N_answer_key`)

    The second is the real test: it goes side -> key through `encode_answer_key`, so it only
    agrees if the mapping was applied the right way round. A session where the decoding was
    inverted shows 0% key agreement, not 50%.

    Parameters:
    -----------
    df_all : pd.DataFrame
        The aggregated long format from `collect_all_long_files_ex11`
    cfg, csv_directory
        Passed to `load_behavioural_trials`

    Returns:
    --------
    pd.DataFrame : one row per session, with n_trials, sound and key agreement, and a `problem`
                   flag for anything below 100%
    """
    cfg = cfg or EX11_PREPROCESSING_CONFIG
    behavioural = load_behavioural_trials(cfg, csv_directory)

    # One row per trial from the long format; the sound/response columns are trial-level, so any
    # saccade row of the trial carries them.
    trials = (df_all
              .sort_values(['id_subject', 'trial_number'])
              .drop_duplicates(subset=['id_subject', 'trial_number'])
              .set_index(['session_file', 'trial_number_in_session']))

    rows = []
    for session_file, beh in behavioural.groupby('session_file'):
        if session_file not in trials.index.get_level_values(0):
            rows.append({'session_file': session_file, 'n_trials_csv': len(beh),
                         'n_trials_matched': 0, 'sound_agreement': np.nan,
                         'key_agreement': np.nan, 'n_keys_compared': 0,
                         'problem': '*** not in the long format ***'})
            continue

        sound_hits = sound_total = key_hits = key_total = matched = 0
        for _, beh_row in beh.iterrows():
            try:
                trial = trials.loc[(session_file, int(beh_row['trial_number']))]
            except KeyError:
                continue
            matched += 1
            for which, prefix in ((1, 'first'), (2, 'second')):
                # 'sound_left' / 'sound_right' / 'sound_empty' -> 'left' / 'right' / 'empty'
                csv_sound = str(beh_row[f'{prefix}_sound_type']).replace('sound_', '')
                our_sound = trial[f'sound_{which}_direction']
                if csv_sound in ('left', 'right'):
                    sound_total += 1
                    sound_hits += int(our_sound == csv_sound)

                csv_key = beh_row[f'{prefix}_answer']
                if csv_key in ('up', 'down'):
                    key_total += 1
                    key_hits += int(trial[f'sound_{which}_answer_key'] == csv_key)

        sound_agreement = sound_hits / sound_total if sound_total else np.nan
        key_agreement = key_hits / key_total if key_total else np.nan
        problems = []
        if matched != len(beh):
            problems.append(f'{len(beh) - matched} trial(s) missing')
        if sound_total and sound_agreement < 1:
            problems.append('sound mismatch')
        if key_total and key_agreement < 1:
            problems.append('KEY MISMATCH -- decoding may be inverted')
        rows.append({
            'session_file': session_file,
            'n_trials_csv': len(beh),
            'n_trials_matched': matched,
            'sound_agreement': sound_agreement,
            'n_sounds_compared': sound_total,
            'key_agreement': key_agreement,
            'n_keys_compared': key_total,
            'problem': '*** ' + '; '.join(problems) + ' ***' if problems else '',
        })

    return pd.DataFrame(rows).sort_values('session_file').reset_index(drop=True)
