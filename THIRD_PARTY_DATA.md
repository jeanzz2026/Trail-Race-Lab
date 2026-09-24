# Third-party data and reproducibility

This repository contains the data needed to reproduce the checked-in model:

- `data/modern_race_score_labels.csv`: manually curated public score/time labels,
  including source URLs.
- `model_artifacts/trap_race_features.csv`: a race-level aggregate derived from
  the public TRAP corpus. It contains course and aggregate timing features, not
  the raw participant rows.
- `model_artifacts/*.csv` and `race_score_model_v1.json`: validation outputs and
  the trained model artifact.

## TRAP raw data

The raw TRAP files are not copied into this repository. The upstream TRAP-data
repository does not currently declare an explicit software/data license and the
two result files contain historical participant-level records. Keeping the raw
files at their original source avoids creating a second, apparently licensed
distribution.

To rebuild the race-level aggregate from the upstream raw files, run:

```powershell
.\scripts\fetch_trap_data.ps1
.\.venv\Scripts\python.exe scripts\train_offline_model.py `
  --trap-dir vendor\TRAP-data\ITRA
```

or on macOS/Linux:

```bash
./scripts/fetch_trap_data.sh
.venv/bin/python scripts/train_offline_model.py --trap-dir vendor/TRAP-data/ITRA
```

TRAP asks research users to cite:

> Fogliato, Riccardo, Natalia L. Oliveira, and Ronald Yurko. “TRAP: a
> predictive framework for the Assessment of Performance in Trail Running.”
> Journal of Quantitative Analysis in Sports (2020).

Sources:

- <https://github.com/ricfog/TRAP-data>
- <https://github.com/ricfog/ScrapITRA>

The application and model are independent, unofficial research prototypes and
are not affiliated with or endorsed by ITRA.
