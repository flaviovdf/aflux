#-*- coding: utf8
from __future__ import division, print_function

from aflux import dataio
from aflux import inter
from aflux.tests import files

from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal

def test_full_learn_null():
    rv = inter.fit(files.LARGE, 20, .1, .1, .1, 800, 300)
    
    Count_zh = rv['Count_zh']
    Count_sz = rv['Count_sz'] 
    Count_dz = rv['Count_dz'] 
    
    assert_equal(Count_zh.sum(), 149819)
    assert_equal(Count_sz.sum(), 149819)
    assert_equal(Count_dz.sum(), 149819)
    
    count_h = rv['count_h']
    count_z = rv['count_z']

    assert_equal(count_h.sum(), 149819)
    assert_equal(count_z.sum(), 149819)

    assert rv['assign'].shape == (149819, )

    Theta_zh = rv['Theta_zh']
    Psi_sz = rv['Psi_sz']
    Psi_dz = rv['Psi_dz']
    
    assert_almost_equal(1, Theta_zh.sum(axis=0))
    assert_almost_equal(1, Psi_sz.sum(axis=0))
    assert_almost_equal(1, Psi_dz.sum(axis=0))
