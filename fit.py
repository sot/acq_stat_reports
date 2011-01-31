#!/usr/bin/env python

# Acquisition Statistics Report generation

import os
import sys
import numpy as np
import logging

# Matplotlib setup
# Use Agg backend for command-line (non-interactive) operation
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

from Ska.Matplotlib import plot_cxctime
from glob import glob
import json
from Chandra.Time import DateTime
import sherpa.ui as ui

import Ska.DBI

trend_start = 270691265.18
now =  412880113.73
csec_year = 86400 * 365.25
m = 0.00220747
b = 0.0175493



dbh = Ska.DBI.DBI(dbi='sybase', server='sybase', user='aca_read')
acq = dbh.fetchall("""select tstart, obc_id from acq_stats_data
                       where tstart >= %f and tstart <= %f """ 
                   % (trend_start, now))

times = (acq['tstart'] - trend_start) / csec_year
fail_mask = acq['obc_id'] == 'NOID'

# a log likelihood sum to be used as the user statistic
def llh(data, model, staterror=None,syserror=None,weight=None):
    prob = p(times, fail_mask, ypoly.c1.val, ypoly.c0.val)
#    print ypoly.c1.val, ypoly.c0.val
    return (np.sum(-np.log(prob)), np.ones_like(times))

# the probability per acquisition based on the given probability
# line... return the probability as a vector of the same length
# as the boolean acquisition and the times
def p( times, fail_mask, slope, intercept):
    # I tried ones_like here, but didn't have an easy dtype option...
    prob = np.ones(len(times),dtype=numpy.float64)
    fail_prob = slope * times + intercept
    success_prob = prob - fail_prob
    prob[fail_mask == False] = success_prob[fail_mask == False] 
    prob[fail_mask] = fail_prob[fail_mask]
    return prob

# I've got nothing for error ...
def my_err(data):
    return np.ones_like(data)

data_id = 0
ui.set_method('simplex')
ui.polynom1d.ypoly
ui.set_model(data_id, 'ypoly')
ui.thaw(ypoly.c0)
ui.thaw(ypoly.c1)
# set these to my guesses from the binned analysis
ypoly.c0 = b
ypoly.c1 = m


ui.load_arrays(data_id, times - trend_start, fail_mask == False)
#ui.set_staterror(data_id,
#                 np.max([rates[d]['fail']['err_h'][time_ok],
#                         rates[d]['fail']['err_l'][time_ok]], axis=0))

ui.load_user_stat("loglike", llh, my_err)
ui.set_stat(loglike)
ui.fit(data_id)
myfit = ui.get_fit_results()
import pickle
pickle.dump(myfit, open('fitfile.pkl', 'w'))



