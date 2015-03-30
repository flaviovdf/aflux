#-*- coding: utf8
from __future__ import division, print_function

from collections import defaultdict
from collections import OrderedDict

import numpy as np
import pandas as pd

def save_model(out_fpath, model):
    store = pd.HDFStore(out_fpath, 'w')
    for model_key in model:
        model_val = model[model_key]
        
        if type(model_val) == np.ndarray:
            store[model_key] = pd.DataFrame(model_val)
        else:
            store[model_key] = pd.DataFrame(model_val.items(), \
                    columns=['Name', 'Id'])
    store.close()

def initialize_trace(trace_fpath, num_topics, burn_in):
    
    count_zh_dict = defaultdict(int)
    count_sz_dict = defaultdict(int)
    count_dz_dict = defaultdict(int)
    count_z_dict = defaultdict(int)
    count_h_dict = defaultdict(int)

    hyper2id = OrderedDict()
    source2id = OrderedDict()
    dest2id = OrderedDict()
    
    Trace = []
    with open(trace_fpath, 'r') as trace_file:
        for i, line in enumerate(trace_file):
            hyper_str, source_str, dest_str, c = line.strip().split('\t')
            c = int(c)

            for _ in xrange(c):
                if hyper_str not in hyper2id:
                    hyper2id[hyper_str] = len(hyper2id)
                
                if source_str not in source2id:
                    source2id[source_str] = len(source2id)
                
                if dest_str not in dest2id:
                    dest2id[dest_str] = len(dest2id)

                h = hyper2id[hyper_str]
                s = source2id[source_str]
                d = dest2id[dest_str]
                
                z = np.random.randint(num_topics)
                count_zh_dict[z, h] += 1
                count_sz_dict[s, z] += 1
                count_dz_dict[d, z] += 1
                count_z_dict[z] += 1
                count_h_dict[h] += 1
                
                Trace.append([h, s, d, z])
    
    Trace = np.asarray(Trace, dtype='i4', order='C')
    nh = len(hyper2id)
    ns = len(source2id)
    nd = len(dest2id)
    nz = num_topics

    Count_zh = np.zeros(shape=(nz, nh), dtype='i4')
    Count_sz = np.zeros(shape=(ns, nz), dtype='i4')
    Count_dz = np.zeros(shape=(nd, nz), dtype='i4')
    count_h = np.zeros(shape=(nh,), dtype='i4')
    count_z = np.zeros(shape=(nz,), dtype='i4')

    for z in xrange(Count_zh.shape[0]):
        count_z[z] = count_z_dict[z]

        for h in xrange(Count_zh.shape[1]):
            count_h[h] = count_h_dict[h]
            Count_zh[z, h] = count_zh_dict[z, h]

        for s in xrange(Count_sz.shape[0]):
            Count_sz[s, z] = count_sz_dict[s, z]

        for d in xrange(Count_dz.shape[0]):
            Count_dz[d, z] = count_dz_dict[d, z]
    
    prob_topics_aux = np.zeros(nz, dtype='f8')
    
    Theta_zh = np.zeros(shape=(nz, nh), dtype='f8')
    Psi_sz = np.zeros(shape=(ns, nz), dtype='f8')
    Psi_dz = np.zeros(shape=(nd, nz), dtype='f8')

    return Trace, Count_zh, Count_sz, Count_dz, count_h, count_z, \
            prob_topics_aux, Theta_zh, Psi_sz, Psi_dz, hyper2id, source2id, dest2id
