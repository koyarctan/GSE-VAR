# Favorita category-level sales

Input: official Store Sales `train.csv` and `stores.csv` in `data/favorita/raw/`. Select the smallest eligible store ID using training nonzero fractions >=0.8 and positive variance for all12 categories. The publication selected store1 in Quito.

Train2016-07-01 through2017-02-28; validate2017-03-01 through2017-04-30 (61 points); test2017-05-01 through2017-08-15 (107 points). Missing2016-12-25 is not interpolated; lag windows crossing it are excluded. Log1p sales are adjusted by training weekday means and divided by training residual standard deviations. K=7.

`config.json` records all30 GSE-VAR candidates (ten expected-L0 scales and three learning-rate pairs), ten linear-SGL candidates and the maximum5000 epochs. Within each GSE candidate save its best validation checkpoint, then select the global minimum validation MSE, breaking ties by epoch and candidate. Reuse the selected checkpoint without a train+validation refit. The publication contains seed42 only; `seeds` can be edited for additional repetitions, which are reported separately from publication results.

Run all three methods with python -m experiments.favorita.run --device cuda:0. Select one with --method GSE-VAR, --method cMLP or --method Linear-SGL. The linear comparison retains sparse-group lasso on fixed VAR coefficients; it is distinct from GSE-VAR's Bernoulli structure learner. The canonical protocol uses no external predictors. VARX functionality is tested separately in the library tests.

## Additional cMLP comparison

The additional NGC/cMLP experiment is part of this same data protocol:

python -m experiments.favorita.cmlp --device cuda:0

It prepares the identical calendar splits from official raw data on first use, searches ten regularization strengths and two learning rates, saves candidate-best validation states and evaluates the minimum-validation-MSE model on the test period. Use --check-data to verify data only or --smoke --device cpu for short pipeline verification. Its raw-scale RMSE/MAE and standardized MSE are labeled separately. Appendix A compares all three models on standardized MSE/RMSE, target-level errors and selected structures. The saved cMLP model has test MSE 0.47266290 and 39 nonself edges (seed42).
