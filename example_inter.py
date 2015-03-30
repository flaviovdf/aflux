from __future__ import division, print_function

from aflux import dataio
from aflux import inter

import pandas as pd
import numpy as np

FPATH = './aflux/tests/sample_data/example.dat'
rv = inter.fit(FPATH, 40, 50.0 / 40, .001, .001, 300, 150)

out_fpath = './example_out/model.h5'
dataio.save_model(out_fpath, rv)

Psi_sz = rv["Psi_sz"]
Psi_dz = rv["Psi_dz"]

id2dest = dict((v, k) for k, v in rv["dest2id"].items())
id2source = dict((v, k) for k, v in rv["source2id"].items())

for z in xrange(40):
    top_source = Psi_sz[:, z].argsort()[::-1][:5]
    top_dest = Psi_dz[:, z].argsort()[::-1][:5]
    
    print(z)
    for i in xrange(5):
        print(id2source[top_source[i]])
    print()

    for i in xrange(5):
        print(id2dest[top_dest[i]])
    print()
    print()
    print()
