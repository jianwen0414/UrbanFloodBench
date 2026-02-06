"""
Standardized RMSE loss function.

Metric definition (competition scoring):
    For each node i, the error is normalised by the standard deviation
    of the ground-truth water level at that node across time.  The final
    score is the mean of these per-node standardised RMSEs.

        SRMSE = (1/N) Σ_i  sqrt( mean_t( (y_it - ŷ_it)² ) ) / σ_i
"""
