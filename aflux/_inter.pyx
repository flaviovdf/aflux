#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: nonecheck = False
# cython: wraparound = False

from __future__ import division, print_function

from aflux.myrandom.random cimport rand

cdef extern from 'stdio.h':
    int printf(char *, ...) nogil

cdef void average(double[:,:] Theta_zh, double[:,:] Psi_sz, \
        double[:,:] Psi_dz, int n) nogil:

    cdef int nz = Theta_zh.shape[0]
    cdef int nh = Theta_zh.shape[1]
    cdef int ns = Psi_sz.shape[0]
    cdef int nd = Psi_dz.shape[0]
    
    cdef int z = 0
    cdef int h = 0
    cdef int s = 0 
    cdef int d = 0
    for z in xrange(nz):
        for h in xrange(nh):
            Theta_zh[z, h] /= n

        for s in xrange(ns):
            Psi_sz[s, z] /= n
    
        for d in xrange(nd):
            Psi_dz[d, z] /= n

def _average(Theta_zh, Psi_sz, Psi_dz, n):
    '''Wrapper used mostly for unit tests. Do not call directly otherwise'''
    average(Theta_zh, Psi_sz, Psi_dz, n)

cdef void aggregate(int[:,:] Count_zh, int[:,:] Count_sz, int[:,:] Count_dz, \
        double[:,:] Theta_zh, double[:,:] Psi_sz, double[:,:] Psi_dz) nogil:
    
    cdef int nz = Theta_zh.shape[0]
    cdef int nh = Theta_zh.shape[1]
    cdef int ns = Psi_sz.shape[0]
    cdef int nd = Psi_dz.shape[0]
    
    cdef int z = 0
    cdef int h = 0
    cdef int s = 0 
    cdef int d = 0
    for z in xrange(nz):
        for h in xrange(nh):
            Theta_zh[z, h] += Count_zh[z, h]

        for s in xrange(ns):
            Psi_sz[s, z] += Count_sz[s, z]
    
        for d in xrange(nd):
            Psi_dz[d, z] += Count_dz[d, z]

def _aggregate(Count_zh, Count_sz, Count_dz, Theta_zh, Psi_sz, Psi_dz):
    '''Wrapper used mostly for unit tests. Do not call directly otherwise'''
    aggregate(Count_zh, Count_sz, Count_dz, Theta_zh, Psi_sz, Psi_dz)

cdef double dir_posterior(double joint_count, double global_count, \
        double num_occurences, double smooth) nogil:

    cdef double numerator = smooth + joint_count
    cdef double denominator = global_count + (smooth * num_occurences)
    
    if denominator == 0:
        return 0
    else:
        return numerator / denominator

def _dir_posterior(joint_count, global_count, num_occurences, smooth):
    '''Wrapper used mostly for unit tests. Do not call directly otherwise'''
    return dir_posterior(joint_count, global_count, num_occurences, smooth)

cdef int sample(int hyper, int source, int dest, \
        int[:,:] Count_zh, int[:,:] Count_sz, int[:,:] Count_dz, \
        int[:] count_h, int[:] count_z, double alpha_zh, double beta_zs, \
        double beta_zd, double[:] prob_topics_aux) nogil:
    
    cdef int nz = prob_topics_aux.shape[0]
    cdef int ns = Count_sz.shape[0]
    cdef int nd = Count_dz.shape[0]
    cdef double sum_pzt = 0
    cdef int z = 0
    
    for z in xrange(nz):
        prob_topics_aux[z] = \
            dir_posterior(Count_zh[z, hyper], count_h[hyper], nz, alpha_zh) * \
            dir_posterior(Count_sz[source, z], count_z[z], ns, beta_zs) * \
            dir_posterior(Count_dz[dest, z], count_z[z], nd, beta_zd)
    
    #accumulate multinomial parameters
    for z in xrange(1, nz):
        prob_topics_aux[z] += prob_topics_aux[z - 1]
    
    cdef double u = rand() * prob_topics_aux[nz - 1]
    cdef int new_topic = nz - 1
    for z in xrange(nz):
        if u < prob_topics_aux[z]:
            new_topic = z
            break
    
    return new_topic

def _sample(hyper, source, dest, \
        Count_zh, Count_sz, Count_dz, count_h, count_z, \
        alpha_zh, beta_zs, beta_zd, prob_topics_aux):
    '''Wrapper used mostly for unit tests. Do not call directly otherwise'''
    return sample(hyper, source, dest, Count_zh, Count_sz, \
            Count_dz, count_h, count_z, alpha_zh, beta_zs, beta_zd, \
            prob_topics_aux)

cdef void do_step(int[:,:] Trace, 
        int[:,:] Count_zh, int[:,:] Count_sz, int[:,:] Count_dz, \
        int[:] count_h, int[:] count_z, double alpha_zh, double beta_zs, \
        double beta_zd, double[:] prob_topics_aux) nogil:
    
    cdef int hyper, source, dest, old_topic
    cdef int new_topic
    cdef int i

    for i in xrange(Trace.shape[0]):
        hyper = Trace[i, 0]
        source = Trace[i, 1]
        dest = Trace[i, 2]
        old_topic = Trace[i, 3]

        Count_zh[old_topic, hyper] -= 1
        Count_sz[source, old_topic] -= 1
        Count_dz[dest, old_topic] -= 1
        count_h[hyper] -= 1
        count_z[old_topic] -= 1

        new_topic = sample(hyper, source, dest, \
                Count_zh, Count_sz, Count_dz, count_h, count_z, \
                alpha_zh, beta_zs, beta_zd, prob_topics_aux)
        
        Trace[i, 3] = new_topic
        Count_zh[new_topic, hyper] += 1
        Count_sz[source, new_topic] += 1
        Count_dz[dest, new_topic] += 1
        count_h[hyper] += 1
        count_z[new_topic] += 1

def _do_step(Trace, Count_zh, Count_sz, Count_dz, count_h, count_z, \
        alpha_zh, beta_zs, beta_zd, prob_topics_aux):
    '''Wrapper used mostly for unit tests. Do not call directly otherwise'''
    do_step(Trace, Count_zh, Count_sz, Count_dz, count_h, count_z, alpha_zh, \
            beta_zs, beta_zd, prob_topics_aux)

cdef void row_normalize(double[:,:] X) nogil:
    
    cdef double sum_ = 0
    cdef int i, j
    for i in xrange(X.shape[0]):
        sum_ = 0

        for j in xrange(X.shape[1]):
            sum_ += X[i, j]

        for j in xrange(X.shape[1]):
            if sum_ > 0:
                X[i, j] = X[i, j] / sum_
            else:
                X[i, j] = 1.0 / X.shape[1]

cdef void fast_gibbs(int[:,:] Trace, 
        int[:,:] Count_zh, int[:,:] Count_sz, int[:,:] Count_dz, \
        int[:] count_h, int[:] count_z, double alpha_zh, double beta_zs, \
        double beta_zd, double[:] prob_topics_aux, \
        double[:,:] Theta_zh, double[:,:] Psi_sz, double[:,:] Psi_dz, \
        int num_iter, int burn_in) nogil:

    cdef int useful_iters = 0
    cdef int i
    for i in xrange(num_iter):
        do_step(Trace, 
                Count_zh, Count_sz, Count_dz, count_h, count_z, \
                alpha_zh, beta_zs, beta_zd, prob_topics_aux)
        
        #average everything out after burn_in
        if i >= burn_in:
            aggregate(Count_zh, Count_sz, Count_dz, Theta_zh, Psi_sz, Psi_dz)
            useful_iters += 1

    average(Theta_zh, Psi_sz, Psi_dz, useful_iters)
    row_normalize(Theta_zh)
    row_normalize(Psi_sz)
    row_normalize(Psi_dz)

def gibbs(Trace, Count_zh, Count_sz, Count_dz, count_h, count_z, \
        alpha_zh, beta_zs, beta_zd, \
        prob_topics_aux, Theta_zh, Psi_sz, Psi_dz, num_iter, burn_in):
    
    fast_gibbs(Trace, Count_zh, Count_sz, Count_dz, count_h, count_z, \
            alpha_zh, beta_zs, beta_zd, prob_topics_aux, \
            Theta_zh, Psi_sz, Psi_dz, num_iter, burn_in)
