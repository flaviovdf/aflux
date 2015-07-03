#-*- coding: utf8
'''
Unit tests for the main sherlock model.
'''
from __future__ import division, print_function

from aflux import dataio
from aflux import _inter
from aflux.tests import files

from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal

def test_aggregate():
    Trace, Count_zh, Count_sz, Count_dz, \
            count_h, count_z, prob_topics_aux, Theta_zh, Psi_sz, \
            Psi_dz, hyper2id, source2id, dest2id = \
            dataio.initialize_trace(files.SIZE10, 2, 10)
    
    assert_equal(Theta_zh, 0)
    assert_equal(Psi_sz, 0)
    assert_equal(Psi_dz, 0)
    
    _inter._aggregate(Count_zh, Count_sz, Count_dz, Theta_zh, Psi_sz, \
            Psi_dz)

    assert_equal(Theta_zh, Count_zh)
    assert_equal(Psi_sz, Count_sz)
    assert_equal(Count_dz, Psi_dz)

def test_average():
    Trace, Count_zh, Count_sz, Count_dz, \
            count_h, count_z, prob_topics_aux, Theta_zh, Psi_sz, \
            Psi_dz, hyper2id, source2id, dest2id = \
            dataio.initialize_trace(files.SIZE10, 2, 10)
    
    assert_equal(Theta_zh, 0)

    assert_equal(Theta_zh, 0)
    assert_equal(Psi_sz, 0)
    assert_equal(Psi_dz, 0)
    
    _inter._aggregate(Count_zh, Count_sz, Count_dz, Theta_zh, Psi_sz, \
            Psi_dz)

    assert_equal(Theta_zh, Count_zh)
    assert_equal(Psi_sz, Count_sz)
    assert_equal(Psi_dz, Count_dz)
    
    _inter._average(Theta_zh, Psi_sz, Psi_dz, 10)

    assert_equal(Theta_zh, Count_zh / 10)
    assert_equal(Psi_sz, Count_sz / 10)
    assert_equal(Psi_dz, Count_dz / 10)

def test_posterior():
    assert_equal(.6086956521739131, _inter._dir_posterior(2, 3, 2, 0.8))

def test_sample():
    Trace, Count_zh, Count_sz, Count_dz, \
            count_h, count_z, prob_topics_aux, Theta_zh, Psi_sz, \
            Psi_dz, hyper2id, source2id, dest2id = \
            dataio.initialize_trace(files.SIZE10, 2, 10)
     
    tstamp_idx = 3
    hyper = Trace[tstamp_idx, 0]
    source = Trace[tstamp_idx, 1]
    dest = Trace[tstamp_idx, 2]
    old_topic = Trace[tstamp_idx, 3]

    new_topic = _inter._sample(hyper, source, dest, \
            Count_zh, Count_sz, Count_dz, count_h, \
            count_z, .1, .1, .1, prob_topics_aux)
    
    assert new_topic <= 3

def test_estep():
    Trace, Count_zh, Count_sz, Count_dz, \
            count_h, count_z, prob_topics_aux, Theta_zh, Psi_sz, \
            Psi_dz, hyper2id, source2id, dest2id = \
            dataio.initialize_trace(files.SIZE10, 2, 10)
    
    
    alpha_zh = .1
    beta_zs = .1
    beta_zd = .1

    assert_equal(Count_zh.sum(), 10)
    assert_equal(Count_sz.sum(), 10)
    assert_equal(Count_dz.sum(), 10)
    
    assert_equal(count_h[0], 4)
    assert_equal(count_h[1], 4)
    assert_equal(count_h[2], 2)
    
    new_state = _inter._do_step(Trace, \
            Count_zh, Count_sz, Count_dz, count_h, count_z, alpha_zh, \
            beta_zs, beta_zd, prob_topics_aux)

    assert_equal(count_h[0], 4)
    assert_equal(count_h[1], 4)
    assert_equal(count_h[2], 2)
    

    assert_equal(Count_zh.sum(), 10)
    assert_equal(Count_sz.sum(), 10)
    assert_equal(Count_dz.sum(), 10)

def test_gibbs():
    
    Trace, Count_zh, Count_sz, Count_dz, \
            count_h, count_z, prob_topics_aux, Theta_zh, Psi_sz, \
            Psi_dz, hyper2id, source2id, dest2id = \
            dataio.initialize_trace(files.SIZE10, 2, 10)
    
    alpha_zh = .1
    beta_zs = .1
    beta_zd = .1
    
    assert (Theta_zh == 0).all()
    assert (Psi_sz == 0).all()
    assert (Psi_dz == 0).all()
    
    old_Count_zh = Count_zh.copy()
    old_Count_sz = Count_sz.copy()
    old_Count_dz = Count_dz.copy()
    old_count_h = count_h.copy()
    old_count_z = count_z.copy()

    _inter.gibbs(Trace, 
            Count_zh, Count_sz, Count_dz, 
            count_h, count_z, alpha_zh, beta_zs, beta_zd,
            prob_topics_aux, Theta_zh, Psi_sz, Psi_dz, 10, 2)
    
    assert (Theta_zh > 0).sum() > 0
    assert (Psi_sz > 0).sum() > 0
    assert (Psi_dz > 0).sum() > 0
    
    assert_almost_equal(1, Theta_zh.sum(axis=0))
    assert_almost_equal(1, Psi_sz.sum(axis=0))
    assert_almost_equal(1, Psi_dz.sum(axis=0))

    assert (old_Count_zh != Count_zh).any()
    assert (old_Count_sz != Count_sz).any()
    assert (old_Count_dz != Count_dz).any()
    
    assert (old_count_h == count_h).all() #the count_h should not change
    assert (old_count_z != count_z).any()
