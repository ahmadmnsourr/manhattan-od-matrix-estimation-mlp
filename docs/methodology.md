# Methodology

## Problem formulation

For each simulated five-minute interval, an observed zonal traffic-count matrix is constructed from the Manhattan network. The underlying 20-zone OD matrix is the prediction target. The features include zonal link-flow counts, row sums representing observed departures, column sums representing observed arrivals, and diagonal intra-zonal flow values. Targets are transformed with `log1p` during training and reconstructed for evaluation.

## Network zoning and data preparation

A K-means partition groups Manhattan intersections into **K = 20 zones**. Link flows are then aggregated into within-zone and between-adjacent-zone counts, while OD records are converted from edge-level origin/destination pairs into zonal OD matrices. The provided scripts support creating these intermediary files from network geometry and simulated traffic JSON.

## Learning architecture

The reported baseline uses an MLP with three hidden layers of **512, 256 and 128 units**. The code supports alternate widths, dropout, different normalization and output-head options, and an optional low-rank target representation. Training uses PyTorch, log-transformed targets and standardized features.

The composite training objective includes data reconstruction terms and penalties relating to marginal totals, diagonal OD components and total traffic demand. Inference also supports optional global rescaling and iterative proportional fitting (IPF), depending on configuration.

## Evaluation design

The code supports a fixed-slot split for each of the four demand levels: the first twelve five-minute windows are partitioned into **eight training**, **two validation**, and **two testing** windows. The scaler is fit only on training data. See the training script's arguments for chunking and alternative configurations.

## Hyperparameter optimization

The Optuna search uses a **TPE sampler** and **Hyperband pruning**. Each trial launches the training script, streams intermediate validation sMAPE readings, and minimizes the best validation score. The search explores architecture, dropout, learning rate, training budget, regularization, loss weights and selected inference-time corrections.

## Reduced-observation experiments

Link-flow records are randomly masked before zonal aggregation. The model is then retrained for the reduced-availability scenario, and OD reconstruction metrics are evaluated on its corresponding held-out data. The two sets of reported results (tuned-model performance and masking sensitivity) represent distinct experiments.
