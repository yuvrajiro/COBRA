## COBRA

This is implementation of the [COBRA](https://doi.org/10.1016/j.jmva.2015.04.007)
algorithm, This is a straight forward implementation of the algorithm
that is based on the paper.

It is based upon Scikit-Survival Estimators.


### TODO :
1. Write the Documentation more clearly
2. Add the test cases
3. Add the W&B support
4. Add the support for the other optimizers
5. Add JAX Support







### References

[1] Gérard Biau, Aurélie Fischer, Benjamin Guedj, James D. Malley,
COBRA: A combined regression strategy,
Journal of Multivariate Analysis,
Volume 146,
2016,
Pages 18-28,
ISSN 0047-259X,
https://doi.org/10.1016/j.jmva.2015.04.007.
(https://www.sciencedirect.com/science/article/pii/S0047259X15000950)
Abstract: A new method for combining several initial estimators of the regression function is introduced. Instead of building a linear or convex optimized combination over a collection of basic estimators r1,…,rM, we use them as a collective indicator of the proximity between the training data and a test observation. This local distance approach is model-free and very fast. More specifically, the resulting nonparametric/nonlinear combined estimator is shown to perform asymptotically at least as well in the L2 sense as the best combination of the basic estimators in the collective. A companion R package called COBRA (standing for COmBined Regression Alternative) is presented (downloadable on http://cran.r-project.org/web/packages/COBRA/index.html). Substantial numerical evidence is provided on both synthetic and real data sets to assess the excellent performance and velocity of our method in a large variety of prediction problems.
Keywords: Combining estimators; Consistency; Nonlinearity; Nonparametric regression; Prediction