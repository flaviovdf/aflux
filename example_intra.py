import matplotlib.pyplot as plt

import numpy as np
import lmfit

from aflux import intra

alpha = 5
beta = 1
num_states = 10
X = intra.get_matrix(alpha, beta, num_states)

x_ticks = np.arange(1, 1000)
eccdf_values = 1 - intra.get_exit_probability(X, x_ticks)

params = intra.fit_ccdf(eccdf_values, x_ticks, num_states=10, \
        residual='mse')
lmfit.report_fit(params)

alpha_estimate = params['alpha'].value
beta_estimate = params['beta'].value

X_estimate = intra.get_matrix(alpha_estimate, beta_estimate, num_states)
eccdf_estimate = 1 - intra.get_exit_probability(X_estimate, x_ticks)

f = (x_ticks[1:] * np.diff(1 - eccdf_values)).sum()
s = ((x_ticks[1:] ** 2) * np.diff(1 - eccdf_values)).sum()

print('Estimates using moments')
beta_moments = f / (f - 1)
alpha_moments = ((beta_moments ** 2) * \
        (3 - 4 * beta_moments - s + beta_moments * s)) / \
        (1 - 2 * beta_moments - s + beta_moments * s)

print('beta =', beta_moments)
print('alpha =', alpha_moments)

plt.ylabel('CCDF')
plt.xlabel('Time')
plt.loglog(x_ticks, eccdf_values, 'wo')
plt.loglog(x_ticks, eccdf_estimate, 'k-')
plt.show()
