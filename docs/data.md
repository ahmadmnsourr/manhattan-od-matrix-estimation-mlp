# Data and input formats

The repository does **not** include the Manhattan CityFlow network files, traffic demand JSON, preprocessed OD matrices, trained model weights, or saved training runs. The code and documentation are available, but running the pipeline requires obtaining the corresponding input data.

The scripts expect CityFlow-style Manhattan network geometry and synthetic traffic-demand scenarios. The scenario data were derived from external transportation simulation resources; their redistribution permissions must be assessed separately. No substitute files are presented as the original evaluation data.

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
