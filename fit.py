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

date_start = '2006:292:00:00:00.000'
trend_start = DateTime(date_start).secs
csec_year = 86400 * 365.25
m = 0.00220747
b = 0.0175493


dbh = Ska.DBI.DBI(dbi='sybase', server='sybase', user='aca_read')
acq = dbh.fetchall("""select tstart, obc_id from acq_stats_data
                       where tstart >= %f"""
                   % (trend_start))

times = (acq['tstart'] - trend_start) / csec_year
fail_mask = acq['obc_id'] == 'NOID'

# a log likelihood sum to be used as the user statistic
def llh(data, model, staterror=None,syserror=None,weight=None):
    prob = p(times, data, model)
#    print ypoly.c1.val, ypoly.c0.val
    return (np.sum(-np.log(prob)), np.ones_like(times))

# the probability per acquisition based on the given probability
# line... return the probability as a vector of the same length
# as the boolean acquisition and the times
# where the.  fail_mask is true at the failurs.
# (probability of the value == True) = failure probability
# ( probability of the value == False ) = 1 - failure probability
def p( times, fail_mask, model):
    prob = np.ones(len(times),dtype=np.float64)
    fail_prob = model
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


ui.load_arrays(data_id, times, fail_mask)
#ui.set_staterror(data_id,
#                 np.max([rates[d]['fail']['err_h'][time_ok],
#                         rates[d]['fail']['err_l'][time_ok]], axis=0))

ui.load_user_stat("loglike", llh, my_err)
#ui.load_user_stat("loglike", llh)
ui.set_stat(loglike)
ui.fit(data_id)
myfit = ui.get_fit_results()

import json
if myfit.succeeded:
    rep_file = open('%s_fitfile.json' % 'acq_fail', 'w')
    rep_file.write(json.dumps(dict(time0=trend_start,
                                   date0=date_start,
                                   date_end=DateTime().date,
                                   m=ypoly.c1.val,
                                   b=ypoly.c0.val,
                                   comment="mx+b with b at time0 and m = (delta rate)/year"),
                              sort_keys=True,
                              indent=4))
    rep_file.close()
else:
    raise ValueError("Not fit")



