import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

np.random.seed(123)
X = np.linspace(0, 1, 20)
true_slope = 2.5
true_intercept = 1.0
sigma = 0.5
y = true_intercept + true_slope * X + np.random.normal(0, sigma, size=len(X))

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    epsilon = pm.HalfCauchy('epsilon', beta=5)

    mu = alpha + beta * X
    y_obs = pm.Normal('y_obs', mu=mu, sigma=epsilon, observed=y)

    trace = pm.sample(2000, tune=1000, cores=1, progressbar=False)

pm.summary(trace).round(2)

# Plot posterior distributions
pm.plot_posterior(trace, var_names=['alpha', 'beta'])
plt.show()
