# Lorenz96

Full protocol: p=20, F=40, K=5, T in {250,500}; three independent pilot series and five evaluation series. The source generator uses a 1000-point burn-in, approximately 0.05 sampling interval and Gaussian observation noise with standard deviation 0.1. Each condition uses the corresponding prefix of the same generated series.

GSE-VAR: 10000 epochs, width in {8,32} selected on pilot data, coefficient/logit learning rates 0.003/0.03, smoothness 0.01, coefficient L2 weight 0.001, initial gate probability 0.5. The ten L0 scales are `[0,0.3,1,2,3,4,5,7,10,30]`, converted using lambda_l0=s/(K*p*p). The exact NGC grids are in `run.py`; the model uses width100, K5 and hierarchical input-weight regularization. F1 is evaluated at the pilot-selected setting; curves use all ten settings at the selected width.

Publication selection: scale4 at T250 and scale5 at T500. New runs perform pilot selection rather than hard-code a favorable result. Reference generation and optimization code, software versions and random seeds determine numerical reproduction. Compare saved arrays with tolerances when versions differ.
