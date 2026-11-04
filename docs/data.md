# Data and input formats

The uploaded code archive did **not** contain the underlying Manhattan CityFlow network files, traffic demand JSON, preprocessed OD matrices, trained model weights, or training results. These materials are therefore not claimed as downloadable from this repository.

The scripts expect the original CityFlow-style road network and demand scenarios described in the accompanying technical manuscript. The study's data are attributed there to Chen et al.; dataset reuse is subject to its source terms. No substitute data are silently provided as though they were the evaluated dataset.

## File layout used by the existing scripts

The original Python scripts resolve dataset paths relative to the folder containing the scripts or relative to their working directory. With the curated structure, the relevant local input/output folders live inside `pipeline/` and are excluded by `.gitignore`:

```text
pipeline/
├── Manhattan Network/
│   └── MPLight/data/manhattan/
│       ├── roadnet_manhattan.json
│       ├── manhattan_7846.json
│       ├── manhattan_11756.json
│       ├── manhattan_21430.json
│       └── manhattan_31217.json
├── Refined Data/
│   ├── Zonage/
│   ├── Traffic counts/
│   ├── OD matrice/
│   └── Traffic counts_zones_deletion/
└── Results/
```

The code's concrete filename patterns matter. `train_od_mlp.py` pairs files like `manhattan_<level>_traffic-counts-<start>_<end>_zone.csv` with `manhattan_<level>_zone-od-<start>_<end>_K20.csv` using level and time-window identifiers. Review the relevant script arguments if working with differently named data.
