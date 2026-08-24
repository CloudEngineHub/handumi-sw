# Analyze and Curate Datasets

HandUMI separates dataset analysis from dataset curation:

1. `dataset analyze` reads metadata and quality results, computes statistics,
   and writes an auditable report with automatic review candidates.
2. A person reviews the candidates, histogram, and duration extremes.
3. `dataset curate` requires that person to name the source episode indices to
   remove, then creates a new validated dataset.

Neither command edits the source dataset or uploads anything. Statistical
outliers are not automatically defective, so analysis never makes the deletion
decision and curation never infers it from the report.

## Analyze a dataset

Run the episode-level sensor and motion checks first when they are relevant:

```bash
handumi validate outputs/hanoi \
  --quality-config configs/quality.yaml
```

Then compute dataset-level statistics and automatic IQR review candidates:

```bash
handumi dataset analyze outputs/hanoi
```

The default report is written to:

```text
outputs/hanoi/meta/handumi_analysis.json
outputs/hanoi/meta/handumi_analysis.md
```

It contains:

- frame and duration statistics;
- histogram bins and counts;
- automatic Tukey IQR fences and candidates;
- the five shortest and five longest episodes for human review;
- per-episode findings and task labels;
- task distribution and storage size by modality;
- global `observation.state` and `action` statistics;
- exact state/action alignment when their shapes are comparable;
- a payload fingerprint that binds the report to the analyzed dataset.

Use `--dry-run` to compute and print the summary without writing the report:

```bash
handumi dataset analyze outputs/hanoi --dry-run
```

## Review the report

Inspect `candidates_for_review`, the histogram, duration extremes, quality
findings, and relevant video. A candidate is evidence for review, not an
instruction to delete. The episode indices in the report always refer to the
source dataset.

The report records source episode and frame totals plus a payload fingerprint.
Curation refuses a stale report if the underlying metadata, Parquet, video, or
audio changed after analysis.

## Curate a new local dataset

After reviewing the report:

```bash
handumi dataset curate outputs/hanoi \
  --analysis outputs/hanoi/meta/handumi_analysis.json \
  --output outputs/hanoi_clean \
  --exclude 6,75 \
  --dry-run
```

`--exclude` is mandatory and contains the source indices confirmed by the
reviewer. The dry run prints the exact plan. Remove `--dry-run` to build the
derivative:

```bash
handumi dataset curate outputs/hanoi \
  --analysis outputs/hanoi/meta/handumi_analysis.json \
  --output outputs/hanoi_clean \
  --exclude 6,75
```

## What curation updates

The output is built in a temporary sibling directory and moved into place only
after validation. The command:

- filters and reindexes every Parquet row;
- rebuilds episode indices, frame indices, global indices, and task indices;
- reconstructs shared MP4 files and preserves their declared codec, pixel
  format, frame rate, and resolution;
- reindexes episode-aligned HandUMI audio when enabled;
- rebuilds episode metadata, video offsets, splits, and aggregate statistics;
- preserves HandUMI capture metadata and calibration snapshots;
- regenerates the dataset card;
- writes `meta/handumi_curation.json` with removed episodes and the complete
  source-to-output episode mapping;
- verifies Parquet counts, statistics, video frame counts, and loading through
  `LeRobotDataset`;
- checks that the source payload did not change during the operation.

The command has no Hub publishing option. Publication remains a separate,
explicit step after reviewing the curated dataset.
