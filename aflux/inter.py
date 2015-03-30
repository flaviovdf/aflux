#-*- coding: utf8
from __future__ import division, print_function

from aflux import dataio

from _inter import gibbs
from collections import OrderedDict

def fit(trace_fpath, num_topics, alpha_zh, beta_zs, beta_zd, num_iter, \
        burn_in):
    '''
    Learns the latent topics from a hypergraph trace. 

    Parameters
    ----------
    trace_fpath : str
        The path of the trace. Each line should be a \
                (hypernode, source, destination) tuple

    num_topics : int
        The number of latent spaces to learn

    alpha_zh : float
        The value of the alpha_zh hyperparameter

    beta_zs : float
        The value of the beta_zs (beta) hyperaparameter

    beta_zs : float
        The value of the beta_zd (beta') hyperparameter

    num_iter : int
        The number of iterations to learn the model from

    burn_in : int
        The burn_in of the chain
    
    Returns
    -------
    
    TODO: explain this better. For the time being, see the keys of the dict.
    A dictionary with the results.
    '''
    Trace, Count_zh, Count_sz, Count_dz, \
            count_h, count_z, prob_topics_aux, Theta_zh, Psi_sz, \
            Psi_dz, hyper2id, source2id, dest2id = \
            dataio.initialize_trace(trace_fpath, num_topics, num_iter)
    
    gibbs(Trace, Count_zh, Count_sz, Count_dz, count_h, count_z, \
            alpha_zh, beta_zs, beta_zd, \
            prob_topics_aux, Theta_zh, Psi_sz, Psi_dz, num_iter, burn_in)
    
    rv = OrderedDict()
    rv['Count_zh'] = Count_zh
    rv['Count_sz'] = Count_sz
    rv['Count_dz'] = Count_dz
    rv['count_h'] = count_h
    rv['count_z'] = count_z
    rv['Theta_zh'] = Theta_zh
    rv['Psi_sz'] = Psi_sz
    rv['Psi_dz'] = Psi_dz
    rv['assign'] = Trace[:, -1]
    rv['hyper2id'] = hyper2id
    rv['source2id'] = source2id
    rv['dest2id'] = dest2id
    
    return rv
