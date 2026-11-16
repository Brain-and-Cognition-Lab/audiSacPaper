"""
Exclusions for Experiment 5 (EX_11), applied identically by every notebook that analyses it.

Two kinds, in this order:

  1. REVERSED BLOCKS. A participant can lose track of the key mapping for part of a session and
     then recover. Such a block measures the reversed mapping, not the task, so it is dropped
     while the participant's other blocks are kept. `find_reversed_blocks` tests each block's
     choice accuracy against chance (one-sided binomial over responded sounds) and corrects the
     p-values with Benjamini-Hochberg across every block of the table; `drop_blocks` removes the
     flagged ones. Run it on the full preprocessed table, before any participant is excluded, so
     the correction family is always the same.

  2. PARTICIPANTS (`EXCLUDE_SUBJECTS`):
       E11004 -- responded to only 28 % of the sounds under the upL mapping
       E11010 -- whole upR session below chance, i.e. answered with the mapping reversed
                 throughout; a session-wide reversal is a participant exclusion, not a block one

Participants without a usable pair of sessions never reach the preprocessed table at all; that
step is in `02_preprocessing_exp5.ipynb`.

`apply_exp5_exclusions` does both and is what the notebooks call.
"""

import numpy as np
import pandas as pd
from scipy import stats


EXCLUDE_SUBJECTS = ('E11004', 'E11010')
BLOCK_FDR_Q = 0.05

SESSION_COLUMNS = ['response_mapping', 'session_half', 'mapping_order', 'mapping_first',
                   'is_first_mapping', 'block_number', 'block_number_global', 'session_file']


def make_sound_table(data, require_nonempty=True, min_rt=300):
    """
    One row per sound: the behavioural unit of EX_11.

    The long format repeats a trial's two sounds across every saccade row of that trial, so the
    trial is de-duplicated first and then the two sounds are stacked. Accuracy is a property of the
    sound and the keypress, and computing it off saccade rows would weight each trial by how many
    times the eyes moved during it.

    The screening mirrors the sound/response half of `config.build_filter`:
      - empty (catch) sounds dropped, when `require_nonempty`
      - a response kept only if it was not implausibly fast (`rt > min_rt`); a too-fast response
        is turned back into an omission rather than dropping the sound.

    Parameters:
    -----------
    data : pd.DataFrame
        The aggregated EX_11 long format (`long_format_ex11_all.csv`), raw or prepared
    require_nonempty : bool
        Drop sounds marked `sound_N_nonempty == 0`
    min_rt : float or None
        Responses at or below this (ms) are treated as no response. None keeps them all.

    Returns:
    --------
    pd.DataFrame : id_subject, trial_number, sound_number, id_sound, the session columns, and
                   sound_direction, sound_nonempty, answer_key, response, response_given, acc, rt
    """
    session_cols = [c for c in SESSION_COLUMNS if c in data.columns]
    trials = (data
              .sort_values(['id_subject', 'trial_number'])
              .drop_duplicates(subset=['id_subject', 'trial_number']))

    frames = []
    for which in (1, 2):
        block = trials[['id_subject', 'trial_number'] + session_cols].copy()
        block['sound_number'] = which
        block['id_sound'] = (trials['trial_number'] - 1) * 2 + which
        for field in ('direction', 'nonempty', 'answer_key', 'response',
                      'response_given', 'acc', 'rt'):
            source = f'sound_{which}_{field}'
            target = field if field.startswith(('response', 'acc', 'rt', 'answer')) \
                else f'sound_{field}'
            block[target] = trials[source].values if source in trials.columns else np.nan
        frames.append(block)

    sounds = pd.concat(frames, ignore_index=True)

    if require_nonempty:
        sounds = sounds[sounds['sound_nonempty'] == 1]
    if min_rt is not None:
        too_fast = (sounds['response_given'] == 1) & (sounds['rt'] <= min_rt)
        sounds.loc[too_fast, ['response', 'answer_key', 'rt']] = np.nan
        sounds.loc[too_fast, ['response_given', 'acc']] = 0

    return (sounds
            .sort_values(['id_subject', 'trial_number', 'sound_number'])
            .reset_index(drop=True))


def find_reversed_blocks(data, q=BLOCK_FDR_Q, chance=0.5, min_rt=300):
    """
    Blocks in which a participant answered with the key mapping reversed.

    Each block's choice accuracy over its responded, non-empty sounds is tested against `chance`
    with a one-sided binomial test (H1: below chance), and the p-values are Benjamini-Hochberg
    corrected across ALL blocks in `data`. The correction matters: at ~500 blocks an uncorrected
    p < .05 flags about one block by chance alone.

    Parameters:
    -----------
    data : pd.DataFrame
        EX_11 long format, raw or passed through `prepare_alternative_data`
    q : float
        FDR level
    chance : float
        Accuracy under guessing
    min_rt : float
        As in `make_sound_table`

    Returns:
    --------
    pd.DataFrame : one row per block -- id_subject, session_file, response_mapping, block_number,
                   block_number_global, n_responded, n_correct, p_correct_choice, p_below_chance,
                   p_fdr, reversed
    """
    sounds = make_sound_table(data, require_nonempty=True, min_rt=min_rt)
    responded = sounds[sounds['response_given'] == 1]
    keys = ['id_subject', 'session_file', 'response_mapping', 'block_number',
            'block_number_global']
    blocks = (responded.groupby(keys)['acc']
              .agg(n_responded='size', n_correct='sum')
              .reset_index())
    blocks['n_correct'] = blocks['n_correct'].astype(int)
    blocks['p_correct_choice'] = blocks['n_correct'] / blocks['n_responded']
    blocks['p_below_chance'] = [
        stats.binomtest(k, n, chance, alternative='less').pvalue
        for k, n in zip(blocks['n_correct'], blocks['n_responded'])]

    # Benjamini-Hochberg, step-up
    p = blocks['p_below_chance'].to_numpy()
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    blocks['p_fdr'] = np.empty_like(p)
    blocks.loc[blocks.index[order], 'p_fdr'] = np.minimum(adjusted, 1.0)
    blocks['reversed'] = blocks['p_fdr'] < q

    return blocks.sort_values(['id_subject', 'block_number_global']).reset_index(drop=True)


def drop_blocks(data, blocks):
    """
    Remove every row of `data` that belongs to a block flagged in `blocks['reversed']`.

    Matches on (id_subject, block_number_global), which is unique within a participant across both
    sessions. Works on the raw long format and on the prepared one alike.
    """
    flagged = blocks.loc[blocks['reversed'], ['id_subject', 'block_number_global']]
    if flagged.empty:
        return data
    key = pd.MultiIndex.from_frame(data[['id_subject', 'block_number_global']])
    drop = key.isin(pd.MultiIndex.from_frame(flagged))
    return data[~drop]


def apply_exp5_exclusions(data, verbose=True):
    """
    Drop reversed blocks (on the full table), then the excluded participants.

    Parameters:
    -----------
    data : pd.DataFrame
        The full EX_11 table, raw or prepared, before any exclusion
    verbose : bool
        Print what was dropped

    Returns:
    --------
    (pd.DataFrame, pd.DataFrame) : the remaining rows, and the per-block screen
    """
    screen = find_reversed_blocks(data, q=BLOCK_FDR_Q)
    data = drop_blocks(data, screen)
    data = data[~data['id_subject'].isin(EXCLUDE_SUBJECTS)]
    if verbose:
        flagged = screen[screen['reversed']]
        print(f"Exp 5: reversed blocks dropped: {len(flagged)} of {len(screen)}"
              + (' (' + ', '.join(f"{r.id_subject} {r.response_mapping} block {r.block_number}"
                                  for r in flagged.itertuples()) + ')' if len(flagged) else ''))
        print(f"Exp 5: participants excluded: {', '.join(EXCLUDE_SUBJECTS)} "
              f"-> {data['id_subject'].nunique()} analysed")
    return data, screen
