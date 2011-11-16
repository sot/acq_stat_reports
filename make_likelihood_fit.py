#!/usr/bin/env python

from __future__ import division
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

#trend_date_start = '2006:292:00:00:00.000'
trend_date_start = '2008:001:00:00:00.000'
#trend_date_start = '2011:246:15:41:04.973'
trend_start = DateTime(trend_date_start).secs
trend_date_stop = DateTime().date
trend_stop = DateTime().secs
#trend_stop = DateTime("2011:050:00:00:00.000").secs
#csec_year = 86400 * 365.25

dbh = Ska.DBI.DBI(dbi='sybase', server='sybase', user='aca_read', database='aca')
if 'stars' not in globals():
    stars = dbh.fetchall("""select obsid, slot, type, color, obc_id, mag, mag_obs, tstart, tstop
    from acq_stats_data
    where color is not null and
    tstart >= %f and tstart < %f
    order by tstart """
                         % (trend_start, trend_stop))
times = DateTime(stars['tstart']).frac_year - DateTime(trend_start).frac_year


figmap = { 'acq_fail' : 1,
           }


# masks of the stars matching the thresholds we've set (5% and 100% hardcoded)
failures = dict(acq_fail=( stars['obc_id'] == 'NOID' ))


# the probability per acquisition based on the given probability
# line... return the probability as a vector of the same length
# as the boolean acquisition and the times
def fail_prob(fail_mask, model):
    # I tried ones_like here, but didn't have an easy dtype option...
    prob = np.ones(len(fail_mask),dtype=np.float64)
    fail_prob = model
    success_prob = prob - fail_prob
    prob[fail_mask == False] = success_prob[fail_mask == False] 
    prob[fail_mask] = fail_prob[fail_mask]
    if (np.any( prob <= 0) or np.any( prob > 1)):
        raise ValueError
    return prob

# a log likelihood sum to be used as the user statistic
def llh(data, model, staterror=None,syserror=None,weight=None):
    prob = fail_prob(data, model)
    return (np.sum(-np.log(prob)), np.ones_like(data))



# I've got nothing for error ...
def my_err(data):
    return np.ones_like(data)*.25

def lim_line(pars, x):
    line = pars[0] * x + pars[1]
    line[line <= 0] = 1e-7
    line[line >= 1] = 1 - 1e-7
    return line

#axplot = {}
#ftype = 'obc_bad'
for ftype in failures:

    fail_mask = failures[ftype]
    data_id = figmap[ftype]
    ui.set_method('simplex')

    ui.load_user_model(lim_line, '%s_mod' % ftype)
    ui.add_user_pars('%s_mod' % ftype, ['m', 'b'])
    ui.set_model(data_id, '%s_mod' % ftype)

    ui.load_arrays(data_id,
                   times,
                   failures[ftype])

    fmod = ui.get_model_component('%s_mod' % ftype)

    fmod.b.min = 0
    fmod.b.max = 1
    fmod.m.min = 0
    fmod.m.max = 0.5
    fmod.b.val=1e-7


    ui.load_user_stat("loglike", llh, my_err)
    ui.set_stat(loglike)
    # the tricky part here is that the "model" is the probability polynomial
    # we've defined evaluated at the data x values.
    # the model and the data are passed to the user stat/ llh
    # function as it is minimized.
    ui.fit(data_id)
    myfit = ui.get_fit_results()
    #axplot[ftype] = ui.get_model_plot(data_id)
    if myfit.succeeded:
        import pickle
        pickle.dump(myfit, open('%s_fitfile.pkl' % ftype, 'w'))

        rep_file = open('%s_fitfile.json' % ftype, 'w')
        rep_file.write(json.dumps(dict(time0=trend_start,
                                       datestop=trend_date_stop,
                                       datestart=trend_date_start,
                                       data=ftype,
                                       bin='unbinned_likelihood',
                                       m=fmod.m.val,
                                       b=fmod.b.val,
                                       comment="mx+b with b at time0 and m = (delta rate)/year"),
                                      sort_keys=True,
                                  indent=4))
        rep_file.close()
    else:
        raise ValueError("Not fit")

## make some plots
## just grab the stars in equal-n chunks
#
#def chunks(l, n):
#    return [l[i:i+n] for i in range(0, len(l), n)]
#
#star_chunks = chunks(stars, 1000)
#s_dates = []
#s_bad_trak = []
#s_obc_bad = []
#s_no_trak = []
#for s in star_chunks:
#    # masks of the stars matching the thresholds we've set (5% and 100% hardcoded)
#    chunk_fails = dict(bad_trak=( s['not_tracking_samples']*1.0/s['n_samples']
#                              >= .05 ),
#                   obc_bad=( s['obc_bad_status_samples']*1.0/s['n_samples']
#                             >= .05 ),
#                   no_trak=( s['not_tracking_samples'] == s['n_samples']))
#    s_dates.append(DateTime(np.mean(s['tstart'])).frac_year)
#    s_bad_trak.append(np.count_nonzero(chunk_fails['bad_trak'])/len(s))
#    s_obc_bad.append(np.count_nonzero(chunk_fails['obc_bad'])/len(s))
#    s_no_trak.append(np.count_nonzero(chunk_fails['no_trak'])/len(s))
#
#
#s_dates = np.array(s_dates)
#s_rates = { 'bad_trak' : np.array(s_bad_trak),
#            'obc_bad' : np.array(s_obc_bad),
#            'no_trak' : np.array(s_no_trak) }



#    ui.clean()
