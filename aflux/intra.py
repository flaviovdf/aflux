#-*- coding: utf8
from __future__ import division, print_function

import matplotlib
matplotlib.use('agg')

from matplotlib import pyplot as plt
from matplotlib import rc

from numpy import linalg as LA

from statsmodels.distributions.empirical_distribution import ECDF

import lmfit
import numpy as np
import plac
import sys

def get_min_value_alpha(num_states):
    '''
    Estimates the minimum value of alpha given the number of states.
    This is uselful to limit the search space for the alpha value.
    '''
    
    candidates = np.arange(1, 2, 0.000001, dtype='d')
    state_numbers = np.arange(num_states - 1, dtype='d') + 1

    for i, c in enumerate(candidates):
        powers = np.power(c, -state_numbers)
        if powers.sum() <= 1:
            break
    
    return candidates[i]

def get_matrix(alpha, beta, num_states=4):
    '''
    Creates the transition matrix.

    Parameters
    ----------
    alpha : double
        The value of valpha
    beta : double
        The value of beta
    num_states : int
        Number of states in the model
    '''

    n = num_states + 1
    X = np.zeros(shape=(n, n), dtype='d')
    
    states_numbers = np.arange(num_states, dtype='d')
    alpha_raised = np.power(alpha, -states_numbers[1:])

    X[0, 1:-1] = alpha_raised
    X[0, -1] = 1 - alpha_raised.sum()
    
    beta_over_alpha_raised = np.power(beta / alpha, states_numbers[1:])
    #beta_over_alpha_raised = np.ones(num_states - 1) * beta

    for row in xrange(1, num_states):
        X[row, row - 1] = beta_over_alpha_raised[row - 1]
        X[row, row] = 1 - beta_over_alpha_raised[row - 1]

    X[-1, -1] = 1
    return X

def get_exit_probability(X, x_ticks):
    '''
    Get's the probability of landing in each state for each
    value on the vector x_ticks

    Parameters
    ----------
    X : matrix
        The transition matrix
    x_ticks : array
        Each element in the array is the number of ticks to
        simulate the model
    '''

    start_probability = np.zeros(X.shape[0], dtype='d')
    start_probability[0] = 1.0
    
    result_arr = np.zeros(x_ticks.shape[0], dtype='d')
    P = np.identity(X.shape[0])
    p = 0
    for i, power in enumerate(x_ticks):
        A = LA.matrix_power(X, power - p)
        P = P.dot(A)
        p = power
        curr_probs = np.dot(start_probability, P)
        result_arr[i] = curr_probs[-1]
    
    return result_arr

def mse(true, pred):
    '''Used for estimating the mean squared error between the true
    values and the predicted values'''

    return (true - pred) / np.sqrt(true.shape[0])

def sse(true, pred):
    '''Used for estimating the sum squared error between the true
    values and the predicted values'''
    
    return true - pred

def mse_log(true, pred):
    '''Used for estimating the mean squared error between the log of true
    values and the log of the predicted values'''
    
    nonzero = (true > 0) & (pred > 0)
    sqrt_n = np.sqrt(true[nonzero].shape[0])
    return (np.log(true[nonzero]) - np.log(pred[nonzero])) / sqrt_n

def sse_log(true, pred):
    '''Used for estimating the sum squared error between the log of true
    values and the log of the predicted values'''
    
    nonzero = (true > 0) & (pred > 0)
    return np.log(true[nonzero]) - np.log(pred[nonzero])

def mrse(true, pred):
    '''Used for estimating the mean relative squared error between the true
    values and the predicted values'''
    
    nonzero = (true > 0) & (pred > 0)
    sqrt_n = np.sqrt(true[nonzero].shape[0])
    return ((true[nonzero] - pred[nonzero]) / true[nonzero]) / sqrt_n

def srse(true, pred):
    '''Used for estimating the sum relative squared error between the true
    values and the predicted values'''
    
    nonzero = (true > 0) & (pred > 0)
    return (true[nonzero] - pred[nonzero]) / true[nonzero]

def minfunc(params, eccdf_values, x_ticks, num_states, residual):
    '''
    Used for the least squares fit. Determines the objective
    to minimize.
    '''
    
    residuals = {
            'mse': mse,
            'sse': sse,
            'mse_log': mse_log,
            'sse_log': sse_log,
            'srse': srse,
            'mrse': mrse,
            }
    
    assert residual in residuals.keys()

    alpha = params['alpha'].value
    beta = params['beta'].value

    X = get_matrix(alpha, beta, num_states)
    eccdf_estimate = 1 - get_exit_probability(X, x_ticks)
    
    return residuals[residual](eccdf_values, eccdf_estimate)

def fit_ccdf(eccdf_values, x_ticks, num_states, residual):
    params = lmfit.Parameters()
    
    #With this expression beta <= alpha and we actually vary the difference
    #between them.
    params.add('beta', value=1, min=0, vary=True)
    params.add('delta_ineq', value=1, min=0, vary=True)
    params.add('alpha', expr='beta + delta_ineq')
    
    lmfit.minimize(minfunc, params, args=(eccdf_values, x_ticks, \
            num_states, residual))

    return params

def fit_data(data, min_value=0, max_value=np.inf, num_states=20, \
        residual='mse'):
    '''
    Fits the data to the model. This method will compute the CCDF and then 
    it will call the fit_ccdf method. The min_value and max_values 
    parameters dete are used to filter the data if necessary.
    
    Parameters
    ----------
    data : array
        The data to compute the cdf to fit
    min_value : int
        The mininum value of the data to consider.
    max_value : int
        The maximum value of the data to consider
    num_states : int
        The number of states in the chain
    '''

    data = np.asanyarray(data)
    data = data[(data >= min_value) & (data < max_value)]
    n = data.shape[0]
    
    if n < 30:
        raise Exception('Not enough points')

    #Log binning of the x-values
    log_min_size = np.log10(data.min())
    log_max_size = np.log10(data.max())
    nbins = np.ceil((log_max_size - log_min_size) * min(n, 20))
    bins = np.unique(np.floor(np.logspace(log_min_size, log_max_size, nbins, \
            base=10.0)))
    hist, edges = np.histogram(data, bins)
    x_ticks = (edges[1:] + edges[:-1]) / 2.0
    x_ticks = np.unique(np.round(x_ticks).astype(int))

    #x_ticks = np.unique(np.round(data).astype(int))
    ecdf = ECDF(data)
    eccdf_values = 1 - ecdf(x_ticks)

    params = fit_ccdf(eccdf_values, x_ticks, num_states, residual)
    
    #Computes the estimate
    alpha = params['alpha'].value
    beta = params['beta'].value
    X = get_matrix(alpha, beta, num_states)
    eccdf_estimate = 1 - get_exit_probability(X, x_ticks)
    
    first_moment = data.mean()
    second_moment = ((x_ticks[1:] ** 2) * np.diff(1 - eccdf_values)).sum()
    
    return params, eccdf_values, eccdf_estimate, x_ticks, \
            first_moment, second_moment

if __name__ == '__main__':
    main_sanity()
