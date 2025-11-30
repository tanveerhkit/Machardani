# Data Acquisition Guide

This project expects labeled mosquito wingbeat recordings sourced from open datasets plus optional synthetic augmentations. Follow the checklist below before running any preprocessing or training scripts.

## 1. Download Public Datasets

| Dataset | Source | Notes |
| --- | --- | --- |
| **HumBugDB** | https://humbug.ac.uk/humbugdb | Register for access; download the WAV bundles plus metadata CSVs. |
| **MosquitoSound** | https://zenodo.org/record/1322805 | Includes recordings for *Aedes*, *Anopheles*, *Culex*. |
| **WHO/CDC field clips** | Country-specific open portals | Optional supplemental clips to boost geographic diversity. |

Store all downloaded archives under `data/raw/` without renaming to keep provenance clear:

```
data/
  raw/
    humbugdb/
    mosquitosound/
    custom/
```

Each dataset subdirectory should contain the original metadata files (CSV/JSON) supplied by the provider.

## 2. Normalize Directory Structure

The training scripts expect this canonical organization after preparation:

```
data/
  processed/
    <dataset_name>/
      audio/<species>/<clip>.wav
      metadata.csv
```

Run `python scripts/prepare_dataset.py --source humbugdb --output data/processed/humbugdb` to:

1. Convert files to mono 16 kHz WAV.
2. Copy metadata columns (`species`, `location`, `label_source`, etc.) into a unified schema.
3. Validate clip durations (default 1-5 s). Invalid clips are listed in `data/processed/<dataset>/discarded.csv`.

## 3. Synthetic Augmentation (Optional)

If a species has fewer than 30 minutes of audio, generate synthetic wingbeat tones:

- Start from clean sine sweeps centered on expected frequencies (300-700 Hz).
- Modulate amplitude to mimic flight jitter.
- Mix with recorded background noise to approximate field conditions.

Save generated clips under `data/processed/<dataset>/audio_synth/<species>/`.

## 4. Dataset Manifest

`scripts/prepare_dataset.py` writes `metadata.csv` per dataset automatically. To train on multiple datasets, concatenate these metadata files (add a `dataset` column if missing) into `data/manifest.csv`, ensuring it contains:

- `clip_path`
- `species`
- `dataset`
- `duration_seconds`
- `split` (train/val/test)

This consolidated manifest feeds the preprocessing pipeline (`scripts/build_features.py`) and ensures reproducibility.

## 6. MosquitoSound Dataset

`data/raw/mosquitosound` already contains the UCR `MosquitoSound` dump (X.npy/X.csv, y labels, index splits). Run `scripts/convert_mosquitosound.py` to materialize WAV files and `metadata.csv` for the pipeline:

```
python scripts/convert_mosquitosound.py \
  --dataset-dir data/raw/mosquitosound \
  --output data/processed/mosquitosound \
  --limit 5000
```

`--limit` controls how many clips are written (default 5,000) so the initial conversion remains manageable; rerun with higher limits as needed. The script preserves the Stratified train/test indices so you can mirror the original evaluation splits.

## 5. Version Control

Large audio files are not tracked in git. Use Git LFS or object storage (S3/MinIO). Commit only manifests and scripts. Keep raw data backed up separately because some providers restrict redistribution.
