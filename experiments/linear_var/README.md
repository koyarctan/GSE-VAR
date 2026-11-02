# Linear VAR signed recovery

The fixed reference generator produces four-dimensional stable VAR(1) series with four nonself true edges, coefficients drawn from [-0.8,-0.2] or [0.2,0.8], and Gaussian noise SD0.4. Full lengths are250 and500. Ten generated series use data seeds42 through51 and a fixed fitting seed42.

Three separate pilot series select width in {8,32} and one L0 scale from `[0,0.01,0.03,0.1,0.3,1,3,10,30]`. GSE-VAR is trained for10000 epochs with coefficient/logit learning rates0.003/0.03, smoothness0.01 and coefficient L2 weight0.001. Publication selections were width8 and scales0.3/0.1.

The graph uses fixed binary gates. Signs are the signs of the time-median coefficients (K=1), with unselected entries set to0. Positive and negative effects are evaluated separately against all other pairs; the primary metric averages their two balanced accuracies. Self-edges are excluded. Missing sign classes produce NaN and are retained as missing values, never replaced with perfect scores.
