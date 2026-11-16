# audiSacPaper

Analysis code for **"Auditory localization improves with aligned saccadic orienting"**
(Jurewicz, Basiński, Basoń & Leszczyński).

Participants freely viewed natural scenes (8 s per scene) while brief sounds were presented.
In five experiments they performed a near-threshold auditory task — three localization tasks, one
pitch discrimination, and a localization task answered on keys orthogonal to the sound axis — and
in one they ignored the sounds. This repository contains everything needed to go from the raw
EyeLink recordings to every figure in the manuscript.

The core question: do spontaneous saccades relate to auditory performance in general, or
selectively to auditory computations involved in spatial orienting?

## Experiment ↔ directory mapping

The directory names (`EX_*`) are historical and **do not** match the experiment numbers used in the
paper. This is the authoritative mapping:

| Paper | Directory | Task | Stimulus | N |
|-------|-----------|------|----------|---|
| Exp 1a | `EX_1` | binaural-direction (left/right keys) | 10 ms pink noise, ITD 200 µs | 52 |
| Exp 1b | `EX_9` | binaural-direction (left/right keys) | 25 ms pink noise, ITD 200 µs | 40 |
| Exp 2 | `EX_7` | monaural-direction (left/right keys) | 6 ms tone, 2 / 2.5 kHz, one ear | 40 |
| Exp 3 | `EX_3` | monaural-pitch (higher/lower, up/down keys) | 6 ms tone, 2 / 2.5 kHz, one ear | 46 |
| Exp 4 | `EX_5` | none — sounds to be ignored, no responses | 6 ms tone, 2 / 2.5 kHz, one ear | 40 |
| Exp 5 | `EX_11` | binaural-direction (left/right judgement on up/down keys, mapping reversed halfway) | 10 ms pink noise, ITD 200 µs | 53 |

For Exps 1a–4 each N equals the number of `.asc` files in `data_raw/EX_*/asc/`. That equality is
the check worth repeating after any re-preprocessing — see the aggregation warning below. Exp 5 is
different: each participant was recorded twice (see below), 55 participants are preprocessed and
53 analysed after the exclusions in `src/exclusions.py`.

**Exp 3 answered on the up/down keys.** The pitch task collected its judgement with the up and down
arrows (`high` → up, `low` → down), so the `response` column holds `'high'`/`'low'` there and
`'left'`/`'right'` in the localization experiments. There is no left/right button in Exp 3, so no
spatial response bias can be computed for it.

**Exp 5 is two recordings per participant.** The localization judgement was given on the UP/DOWN
keys, once under each key mapping (`upL`: UP = left; `upR`: UP = right), each session on a different
half of the image pool. `02_preprocessing_exp5.ipynb` pairs the two sessions, decodes each key
press to the side it meant, and writes one table per participant in which `response` already
holds `'left'`/`'right'` — so every analysis treats Exp 5 like the other localization
experiments. Its data-quality exclusions (blocks answered with the mapping reversed; participants
E11004 and E11010) are applied by the analysis notebooks through `src/exclusions.py`.

## Figures

| Figure | Content | Notebook |
|--------|---------|----------|
| 1 | Study design (drawing) | — |
| 2 | Saccade count, Exps 1a, 1b, 2, 3 | `03_figures_2_4.ipynb` |
| 3 | Saccade count, Exp 4 | `04_figure_3.ipynb` |
| 4 | P(correct response), Exps 1a, 1b, 2, 3 | `03_figures_2_4.ipynb` |
| 5 | Exp 5: saccade count and P(correct), sound- and response-locked (panel A, the task schematic, is added in the manuscript) | `05_figure_5.ipynb` |
| S1 | RM-ANOVA, saccade count, Exps 1a, 1b, 2, 3, 5 | `07_stats_anova.ipynb` |
| S2 | Median RT and d′, Exps 1a, 1b, 2, 3, 5 | `06_figures_S2_S3.ipynb` |
| S3 | Response-locked saccade count and P(correct response), Exps 1a, 1b, 2, 3, 5 | `06_figures_S2_S3.ipynb` |
| S4 | RM-ANOVA, P(correct response), Exps 1a, 1b, 2, 3, 5 | `07_stats_anova.ipynb` |

Each notebook writes its figures to `summaries/plots/figure<N>.{png,pdf}` (300 dpi PNG and PDF).
`04_figure_3`, `05_figure_5` and `06_figures_S2_S3` end with a **Reported values** section that
prints the statistics as the manuscript quotes them (`src/stats_report.py`).

## Data

Data is **not** in the repository. It is deposited on Zenodo: **DOI TBD** — the DOI will be
supplemented upon publication.

Reproducing every figure needs only `preprocessed__long_format.zip` from that deposit:

```bash
unzip preprocessed__long_format.zip     # -> data_preprocessed/EX_*/long_format_*_all.csv
```

The archives' internal paths mirror this repository, so unzipping them here puts each file where
the notebooks look for it:

```
data_raw/EX_*/asc/<id>.asc                          # only needed to re-run preprocessing
data_raw/EX_11/asc/<id>_<mapping>_half<n>.asc       #   (Exp 5: two recordings per participant)
data_raw/EX_*/csv/beh_<id>.csv                      # behavioural logs
data_raw/EX_11/csv/beh_<id>_<mapping>_half<n>.csv   #   (Exp 5: one per session)
data_preprocessed/EX_*/long_format_alt_all.csv      # what the figure notebooks read
data_preprocessed/EX_11/long_format_ex11_all.csv    #   (Exp 5)
```

The `raw__EX_*__asc.zip` archives are needed only to regenerate the preprocessed tables from
scratch. The deposit also carries `participants.csv` (id, age, gender, whether the participant
enters the analysis) and a `MANIFEST.csv` of SHA-256 sums.

## Setup

```bash
uv sync        # Python: preprocessing, figures, running-window metrics
pixi install   # adds R, used by 07_stats_anova.ipynb for the repeated-measures ANOVAs
```

`07_stats_anova.ipynb` calls `Rscript` through `subprocess` and uses base R only — no extra R
packages. It looks for `Rscript` on `PATH`, under `C:/Program Files/R/*/bin/`, and in this
repository's pixi environment (whose DLL folders it adds to `PATH` for that one call).

## Pipeline

```
01_preprocessing.ipynb          →  src/preprocessing.py, src/preprocessing_alt.py
        ↓  data_preprocessed/EX_{1,9,7,3,5}/long_format_alt_all.csv
02_preprocessing_exp5.ipynb     →  src/preprocessing_ex11.py   (on top of the two above)
        ↓  data_preprocessed/EX_11/long_format_ex11_all.csv
03_figures_2_4.ipynb            →  Figures 2, 4      ┐ src/config.py, src/peri_sac.py,
04_figure_3.ipynb               →  Figure 3          │ src/analysis_heatmap.py, src/binning.py,
05_figure_5.ipynb               →  Figure 5          │ src/exclusions.py (Exp 5),
06_figures_S2_S3.ipynb          →  Figures S2, S3    │ src/stats_report.py
07_stats_anova.ipynb            →  Figures S1, S4    ┘ stats_anova_running.R (generated)
```

### 1. Preprocessing — `01_preprocessing.ipynb` (Exps 1a–4)

`src/preprocessing.py` parses the EyeLink `.asc` (`ESACC`, `EBLINK`, `EFIX`, `MSG` markers);
`src/preprocessing_alt.py` assembles trials for the two-sounds-per-scene design.

```
data_raw/EX_*/asc/<id>.asc  +  data_raw/EX_*/csv/beh_<id>.csv
        ↓
data_preprocessed/EX_*/long_format_alt/<id>_long_alt.csv
        ↓
data_preprocessed/EX_*/long_format_alt_all.csv    ← the file every analysis reads
```

Each row is one saccade paired with the sounds in its trial (`sound_1_*`, `sound_2_*` columns).
Saccades are kept if they fall within −500 to +1000 ms of a sound onset, and a saccade can be
assigned to only one sound per trial.

Set `EXPERIMENT` in the configuration cell and run once per experiment.

> **Aggregation globs the directory.** `collect_all_long_files_alt()` builds
> `long_format_alt_all.csv` from *every* `*_long_alt.csv` present in `long_format_alt/`. It has no
> notion of which participants are current, so a per-participant file left behind by an earlier run
> is silently included even if its raw `.asc` is long gone. Excluding a participant therefore means
> deleting **both** the raw file and the preprocessed one. After re-preprocessing, confirm that
> `data['id_subject'].nunique()` equals the `.asc` count before trusting any figure.

### 2. Preprocessing — `02_preprocessing_exp5.ipynb` (Exp 5)

`src/preprocessing_ex11.py` pairs each participant's two recordings, decodes the up/down key
presses to the sides they meant under each session's mapping, removes the training by the trial
structure rather than by a named image, and numbers trials, saccades and sounds across the pair
(`trial_number` 1–200, `block_number_global` 1–10). Everything else is `01`'s machinery, imported.

The notebook's decoding check compares every session against the experiment script's own
behavioural `.csv` — sound by sound and key by key — and must read 1.0 for every session before any
Exp 5 number is read: a mapping applied the wrong way round would invert every answer in a session.
It also reports the evidence for Exp 5's exclusions.

### 3. Figures — `03` to `06`

All four share the same machinery — `src/peri_sac.py` (running-window metrics, baseline
normalization, significance testing, line/bar primitives) and `src/analysis_heatmap.py`
(amplitude × time heatmaps) — and differ in their configuration and layout cells.

- **Sound-locked** panels (Figures 2–5 B/C, S2) use all sounds: `p_correct_response` counts an
  omitted response as an error.
- **Response-locked** panels (Figure 5D, S3) use responded sounds only — a sound with no response
  has no t = 0 — so there `p_correct_response` equals `p_correct_choice`. Their saccade count is
  `saccade_count_normalized_spec`: % change from the *sound-locked* pre-sound baseline of the same
  sounds, because no response-locked window is pre-sound for every trial.
- The heatmaps use a doubled window (200 ms / 50 ms step) so that each of the 10 amplitude bins
  holds enough saccades per cell.

`03` and `04` cache significance tests and heatmap matrices to `summaries/cache/`; set
`LOAD_CACHE = False` after changing a parameter. `05` and `06` cache their per-experiment curves
keyed on the parameters, the data file and the module source, so those rebuild themselves.

### 4. Statistics — `07_stats_anova.ipynb`

**Figures S1 and S4.** Repeated-measures ANOVA at each step of the sliding window:

```
outcome ~ consistency * amplitude + Error(subject / (consistency * amplitude))
```

Fitted with base R `aov()`. Amplitude is split into 3 per-participant quantile bins whose
boundaries are computed **only on pre-sound saccades** (`src/binning.py`). Figure S1 is the
`saccade_count_normalized` outcome, Figure S4 the `p_correct_response` outcome; each experiment is
one block of three effect strips, stacked vertically.

`stats_anova_running.R` is **generated**, not hand-written: the notebook writes it and then runs it
through `Rscript`. Edit the notebook, never the `.R`.

## Analysis parameters

Set in `src/config.py` and in the configuration cell of each notebook. The values used for the
paper:

| Parameter | Value |
|-----------|-------|
| `TIME_REFERENCE` | `time_from_sound_to_sacc_start` (saccade onset relative to sound onset); response-locked panels: `time_from_RT_to_sacc_start` |
| `TIME_RANGE` | −500 to +1000 ms (61 windows); response-locked −1000 to +500 ms |
| `WINDOW_SIZE` / `STEP_SIZE` | 100 ms / 25 ms |
| `BASELINE_RANGE` | −500 to 0 ms before sound onset |
| `MAX_DURATION` | 100 ms (saccades longer than this are dropped) |
| `MIN_AMPLITUDE` | 0.05° (≈1 px) |
| `MIN_RT` | 300 ms (faster responses are treated as anticipations) |
| Amplitude bins | 3 quantile bins (ANOVA, per participant) / 10 quantile bins (heatmaps, global), both from pre-sound saccades |
| Correction | Benjamini–Hochberg FDR, per experiment × dependent variable |

`config.build_filter()` is the single source of truth for inclusion: duration ≤ 100 ms,
amplitude ≥ 0.05°, no blink, non-empty sound, and — if a response was given — RT > 300 ms.

## Key columns in the preprocessed tables

| Column | Meaning |
|--------|---------|
| `id_subject` | Participant |
| `sacc_dur` | Saccade duration (ms) |
| `amplitude_deg`, `x_start_deg`, `x_end_deg` | Saccade geometry in degrees of visual angle |
| `sacc_direction` | `left` / `right`, from horizontal start vs end position |
| `time_from_sound_to_sacc_start` | Saccade onset re. sound onset (negative = saccade first) |
| `consistent` | Saccade direction matches sound side ("congruent" in the paper) |
| `acc` | 1 correct, 0 incorrect, −1 no response |
| `rt` | Reaction time in ms (−1 if no response) |
| `is_blink`, `sound_nonempty`, `response_given` | Filter flags |
| `response_mapping`, `session_half`, `block_number_global` | Exp 5 only: the session's key mapping, which half of the image pool, block 1–10 over the visit |

`config.load_experiment_data()` reads an experiment's table and derives the unified
`consistent` / `acc` / `rt` columns from whichever sound is closest to the saccade.
`config.prepare_alternative_data()` does the same for a frame already in memory.

## Modules

| Module | Role |
|--------|------|
| `src/config.py` | Experiment table, column mapping, `build_filter()`, data loading |
| `src/preprocessing.py` | EyeLink `.asc` parsing |
| `src/preprocessing_alt.py` | Trial assembly and per-experiment aggregation |
| `src/preprocessing_ex11.py` | Exp 5: session pairing, key decoding, decoding check against the behavioural logs |
| `src/exclusions.py` | Exp 5: reversed-block screen and participant exclusions, applied by every analysis |
| `src/peri_sac.py` | Running-window metrics, baseline normalization, significance testing, plotting primitives |
| `src/binning.py` | Quantile binning of amplitude with pre-sound edges |
| `src/analysis_heatmap.py` | Amplitude × time heatmap matrices and plotting |
| `src/stats_report.py` | The "Reported values" printouts: significant intervals, direction and size |

Notebooks add `src` to `sys.path` and import modules by bare name (`import peri_sac`).
`%load_ext autoreload` / `%autoreload 2` is on in the figure notebooks, so editing `src/` takes
effect without restarting the kernel.

Outputs go to `summaries/plots` and `summaries/stats` — both gitignored and safe to regenerate.

## Reproducing the figures

1. Download `preprocessed__long_format.zip` from the Zenodo deposit and unzip it here.
2. `uv sync` (and `pixi install` for R, needed only by `07_stats_anova.ipynb`).
3. Run `03` to `07`. Each writes its figures to `summaries/plots/`.

The notebooks are committed with their outputs, so the published figures are visible without
running anything. `01` and `02` are only needed to rebuild the preprocessed tables from the raw
recordings.

## Questions

Please open an issue, or contact the corresponding author (Marcin Leszczyński,
marcin.leszczynski@uj.edu.pl) or the first author (jurewicz.ka@gmail.com).
