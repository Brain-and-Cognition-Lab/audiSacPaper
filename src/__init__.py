"""
audiSacPaper — analysis code for the saccades & audition manuscript.

Modules
-------
config            Experiment table, unit/column mapping, the standard saccade
                  filter, and unified data loading.
preprocessing     EyeLink .asc parsing: sound/answer markers, RT matching,
                  blinks, saccades, fixations.
preprocessing_alt Trial-based long-format generation (one row per saccade,
                  both sounds per trial) and per-experiment aggregation.
peri_sac          Running-window metrics relative to sound onset, baseline
                  normalization, significance testing, plotting primitives.
binning           Quantile binning of saccade amplitude, with edges taken from
                  pre-sound saccades only.
analysis_heatmap  Amplitude x time heatmap matrices and their plotting.

Notebooks add `src` to sys.path and import these by bare name
(`import peri_sac`), so this package is not normally imported as `src.*`.
"""

__all__ = []
