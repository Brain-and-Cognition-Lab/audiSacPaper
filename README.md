# audiSacPaper

Analysis code for **"Auditory localization improves with aligned saccadic orienting"**
(Jurewicz, Basiński, Basoń & Leszczyński).

Participants freely viewed natural scenes (8 s per scene) while performing a near-threshold
auditory discrimination task. Four experiments: three required localization of a brief sound,
a fourth required discrimination of its pitch under closely matched stimulation. This repository
contains everything needed to go from the raw EyeLink recordings to the six figures in the
manuscript.

The core question: do spontaneous saccades relate to auditory performance in general, or
selectively to auditory computations involved in spatial orienting?

## Experiment ↔ directory mapping

The directory names (`EX_*`) are historical and **do not** match the experiment numbers used in the
paper. This is the authoritative mapping:

| Paper | Directory | Task | Stimulus | N |
|-------|-----------|------|----------|---|
| Exp 1a | `EX_1` | binaural-direction (left/right) | 10 ms pink noise, ITD 200 µs | 52 |
| Exp 1b | `EX_9` | binaural-direction (left/right) | 25 ms pink noise, ITD 200 µs | 40 |
| Exp 2 | `EX_7` | monaural-direction (left/right) | 6 ms tone, 2 / 2.5 kHz, one ear | 40 |
| Exp 3 | `EX_3` | monaural-pitch (higher/lower) | 6 ms tone, 2 / 2.5 kHz, one ear | 46 |

Each N equals the number of `.asc` files in that experiment's `data_raw/EX_*/asc/`. That equality
is the check worth repeating after any re-preprocessing — see the aggregation warning below.

**Exp 3 answered on the up/down keys.** The pitch task collected its judgement with the up and down
arrows (`high` → up, `low` → down), so the `response` column holds `'high'`/`'low'` there and
`'left'`/`'right'` in the three localization experiments. There is no left/right button in Exp 3,
so no spatial response bias can be computed for it.

## Data

Data is **not** in the repository. It is deposited on Zenodo: **DOI TBD** — the DOI will be 
supplemented upon publication.

Reproducing every figure needs only `preprocessed__long_format.zip` from that deposit:

```bash
unzip preprocessed__long_format.zip     # -> data_preprocessed/EX_*/long_format_alt_all.csv
```

The archives' internal paths mirror this repository, so unzipping them here puts each file where
the notebooks look for it:

```
data_raw/EX_*/asc/<id>.asc          # only needed to re-run 01_preprocessing
data_raw/EX_*/csv/beh_<id>.csv      # behavioural logs (read for EX_1 and EX_9)
data_preprocessed/EX_*/long_format_alt_all.csv    # what notebooks 02-04 read
```

`raw__EX_*__asc.zip` are needed only to regenerate the preprocessed tables from scratch. The
deposit also carries `participants.csv` (id, age, gender) and a `MANIFEST.csv` of SHA-256 sums.

## Setup

```bash
uv sync        # Python: preprocessing, figures, running-window metrics
pixi install   # adds R, used by 04_stats_anova.ipynb for the repeated-measures ANOVAs
```

`04_stats_anova.ipynb` calls `Rscript` through `subprocess` and uses base R only — no extra R
packages. A system R installation works just as well as the pixi one, as long as `Rscript` is on
`PATH` (the notebook also looks under `C:/Program Files/R/*/bin/`).

## Pipeline

```
01_preprocessing.ipynb        →  src/preprocessing.py, src/preprocessing_alt.py
        ↓  data_preprocessed/EX_*/long_format_alt_all.csv
02_figures_main.ipynb         →  Figures 2, 3      ┐ src/config.py, src/peri_sac.py,
03_figures_supplementary.ipynb→  Figure S2         ┤ src/analysis_heatmap.py,
04_stats_anova.ipynb          →  Figures S1, S3    ┘ src/binning.py, stats_anova_running.R
```

### 1. Preprocessing — `01_preprocessing.ipynb`

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
> notion of which subjects are current, so a per-subject file left behind by an earlier run is
> silently included even if its raw `.asc` is long gone. Excluding a subject therefore means
> deleting **both** the raw file and the preprocessed one. After re-preprocessing, confirm that
> `data['id_subject'].nunique()` equals the `.asc` count before trusting any figure.

### 2. Figures — `02` and `03`

Both notebooks share the same machinery — `src/peri_sac.py` (running-window metrics, baseline
normalization, significance testing, line/bar primitives) and `src/analysis_heatmap.py`
(amplitude × time heatmaps) — and differ only in their configuration cell.

| Notebook | Produces |
|----------|----------|
| `02_figures_main.ipynb` | **Figures 2 and 3** — 2×2 grid, one panel per experiment: peri-saccadic line plot (congruent vs incongruent) plus two amplitude × time heatmaps. Metrics `saccade_count_normalized` and `p_correct_response`. |
| `03_figures_supplementary.ipynb` | **Figure S2** — d′ and median reaction time, rows = metrics, columns = experiments. |

`02` caches its significance tests and heatmap matrices to `summaries/cache/`. Set
`LOAD_CACHE = False` in the cache cells after changing `METRIC`, `EXPERIMENTS` or any `HM_*`
parameter.

### 3. Statistics — `04_stats_anova.ipynb`

**Figures S1 and S3.** Repeated-measures ANOVA at each step of the sliding window:

```
outcome ~ consistency * amplitude + Error(subject / (consistency * amplitude))
```

Fitted with base R `aov()`. Amplitude is split into 3 per-subject quantile bins whose boundaries
are computed **only on pre-sound saccades** (`src/binning.py`). Figure S1 is the
`saccade_count_normalized` outcome, Figure S3 the `p_correct_response` outcome.

`stats_anova_running.R` is **generated**, not hand-written: the notebook writes it and then runs it
through `Rscript`. Edit the notebook, never the `.R`.

## Analysis parameters

Set in `src/config.py` and in the configuration cell of each notebook. The values used for the
paper:

| Parameter | Value |
|-----------|-------|
| `TIME_REFERENCE` | `time_from_sound_to_sacc_start` (saccade onset relative to sound onset) |
| `TIME_RANGE` | −500 to +1000 ms (61 windows) |
| `WINDOW_SIZE` / `STEP_SIZE` | 100 ms / 25 ms |
| `BASELINE_RANGE` | −500 to 0 ms |
| `MAX_DURATION` | 100 ms (saccades longer than this are dropped) |
| `MIN_AMPLITUDE` | 0.05° (≈1 px) |
| `MIN_RT` | 300 ms (faster responses are treated as anticipations) |
| Amplitude bins | 3 quantile bins (ANOVA, per subject) / 10 quantile bins (heatmaps, global), both from pre-sound saccades |
| Correction | Benjamini–Hochberg FDR, per experiment × dependent variable |

`config.build_filter()` is the single source of truth for inclusion: duration ≤ 100 ms,
amplitude ≥ 0.05°, no blink, non-empty sound, and — if a response was given — RT > 300 ms.

The heatmaps in `02` use a doubled window (200 ms / 50 ms step) so that each of the 10 amplitude
bins holds enough saccades per cell.

## Key columns in `long_format_alt_all.csv`

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

`config.load_experiment_data()` reads this file and derives the unified `consistent` / `acc` / `rt`
columns from whichever sound is closest to the saccade. `config.prepare_alternative_data()` does
the same for a frame already in memory.

## Modules

| Module | Role |
|--------|------|
| `src/config.py` | Experiment table, column mapping, `build_filter()`, data loading |
| `src/preprocessing.py` | EyeLink `.asc` parsing |
| `src/preprocessing_alt.py` | Trial assembly and per-experiment aggregation |
| `src/peri_sac.py` | Running-window metrics, baseline normalization, significance testing, plotting primitives |
| `src/binning.py` | Quantile binning of amplitude with pre-sound edges |
| `src/analysis_heatmap.py` | Amplitude × time heatmap matrices and plotting |

Notebooks add `src` to `sys.path` and import modules by bare name (`import peri_sac`).
`%load_ext autoreload` / `%autoreload 2` is on in the figure notebooks, so editing `src/` takes
effect without restarting the kernel.

Outputs go to `summaries/plots` and `summaries/stats` — both gitignored and safe to regenerate.

## Reproducing the figures

1. Download `preprocessed__long_format.zip` from the Zenodo deposit and unzip it here.
2. `uv sync` (and `pixi install` for R, needed only by `04_stats_anova.ipynb`).
3. Run `02_figures_main.ipynb`, `03_figures_supplementary.ipynb` and `04_stats_anova.ipynb`.
   Each writes its figures to `summaries/plots/` as both `.png` (300 dpi) and `.pdf`.

The notebooks are committed with their outputs, so the published figures are visible without
running anything. `01_preprocessing.ipynb` is only needed to rebuild the preprocessed tables from
the raw recordings.

## Questions

Please open an issue, or contact the corresponding author (Marcin Leszczyński,
marcin.leszczynski@uj.edu.pl) or the first author (jurewicz.ka@gmail.com).
