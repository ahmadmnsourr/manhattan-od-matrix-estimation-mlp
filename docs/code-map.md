# Script map

The original Python implementation has been preserved, with files renamed for readability. Original and repository filenames:

| Original script | Descriptive name |
|---|---|
| `Model.py` | [`train_od_mlp.py`](../pipeline/train_od_mlp.py) |
| `Model_tuned.py` | [`train_od_mlp_variant.py`](../pipeline/train_od_mlp_variant.py) |
| `tune_hyperparams_optuna.py` | [`tune_hyperparameters.py`](../pipeline/tune_hyperparameters.py) |
| `Zonage Creation.py` | [`create_zones.py`](../pipeline/create_zones.py) |
| `OD matrices link.py` | [`generate_link_od.py`](../pipeline/generate_link_od.py) |
| `OD_matrices_conversion.py` | [`convert_zonal_od.py`](../pipeline/convert_zonal_od.py) |
| `Traffic counts.py` | [`extract_link_counts.py`](../pipeline/extract_link_counts.py) |
| `Traffic counts_zones.py` | [`aggregate_zonal_counts.py`](../pipeline/aggregate_zonal_counts.py) |
| `Traffic counts_zones_deletion.py` | [`aggregate_zonal_counts_masked.py`](../pipeline/aggregate_zonal_counts_masked.py) |

The Optuna script also changes the default training-script reference to match the new filename. All other Python script bytes are unchanged.
