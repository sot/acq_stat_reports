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


trend_start = 270691265.18
trend_mxd = DateTime(trend_start).mxDateTime
trend_start_frac = trend_mxd.year + trend_mxd.day_of_year * 1.0 / 365
now = DateTime().mxDateTime
now_frac = now.year + now.day_of_year * 1 / 365

csec_year = 86400 * 365.25

import pickle
fit = pickle.load(open('fitfile.pkl'))
parnames = np.array(fit.parnames)
parvals = np.array(fit.parvals)
b_name = 'ypoly.c0'
m_name = 'ypoly.c1'
b = parvals[ parnames == b_name ][0]
m = parvals[ parnames == m_name ][0]

time_pad = .1

datadir = '/proj/sot/ska/data/acq_stat_reports'

data = { 'month': glob(os.path.join(datadir, '????', 'M??', 'rep.json')),
         'quarter': glob(os.path.join(datadir, '????', 'Q?', 'rep.json')),
         'semi': glob(os.path.join(datadir, '????', 'S?', 'rep.json')),
         'year': glob(os.path.join(datadir, '????', 'YEAR', 'rep.json')),}

#trend_start = 2006.8

rates = dict([(ttype, dict([ (ftype, dict(time=np.array([]),
                                          rate=np.array([]),
                                          err_h=np.array([]),
                                          err_l=np.array([])))
                             for ftype in ['fail',]]))
              for ttype in data.keys()])


for d in data.keys():

    data[d].sort()
    for p in data[d]:
        rep_file = open(p, 'r')
        rep_text = rep_file.read()
        rep = json.loads(rep_text)
        ftype='fail'
        mxd = DateTime( (DateTime(rep['datestart']).secs
                         +  DateTime(rep['datestop']).secs) / 2).mxDateTime
        frac_year = mxd.day_of_year * 1.0 / 365
        rates[d][ftype]['time'] = np.append(rates[d][ftype]['time'],
                                         mxd.year + frac_year)
        #DateTime(rep['datestart']).secs)
        rates[d][ftype]['rate'] = np.append(rates[d][ftype]['rate'],
                                         rep['fail_rate'])
        rates[d][ftype]['err_h'] = np.append(rates[d][ftype]['err_h'],
                                          rep['fail_rate_err_high'])
        rates[d][ftype]['err_l'] = np.append(rates[d][ftype]['err_l'],
                                          rep['fail_rate_err_low'])
        curr_color='black'
        #ptime = DateTime(rep['datestart']).secs
        

#    if d == 'quarter':
#        time_ok = rates[d]['fail']['time'] > trend_start
#        data_id = 0
#        ui.set_method('simplex')
#        ui.load_arrays(data_id,
#                       rates[d]['fail']['time'][time_ok] - trend_start,
#                       rates[d]['fail']['rate'][time_ok])
#        ui.set_staterror(data_id,
#                         np.max([rates[d]['fail']['err_h'][time_ok],
#                                 rates[d]['fail']['err_l'][time_ok]], axis=0))
#        ui.polynom1d.ypoly
#        ui.set_model(data_id, 'ypoly')
#        ui.thaw(ypoly.c0)
#        ui.thaw(ypoly.c1)
#        ui.fit(data_id)
#        myfit = ui.get_fit_results()
#        axplot = ui.get_model_plot(data_id)
#        
#
for d in data.keys():

    fig1 = plt.figure(1,figsize=(5,3))
    ax1 = fig1.gca()
    fig2 = plt.figure(2,figsize=(5,3))
    ax2 = fig2.gca()


    ax1.plot(rates[d]['fail']['time'],
                 rates[d]['fail']['rate'],
                 color=curr_color,
                 linestyle='',
                 marker='.',
                 markersize=5)
    ax2.errorbar(rates[d]['fail']['time'],
                 rates[d]['fail']['rate'],
                 yerr = np.array([rates[d]['fail']['err_l'],
                                  rates[d]['fail']['err_h']]),
                 color=curr_color,
                 linestyle='',
                 marker='.',
                 markersize=5)


#    #fig3= plt.figure(figsize=(5,3))
    for ax in [ax1, ax2]:
        #plot_cxctime(rates['fail']['time'],
        #             rates['fail']['rate'], ax=ax, fmt='b.')
        #ax.plot([axplot.x[0], axplot.x[-1]],
        #             [axplot.y[0], axplot.y[-1]], color='red' )
        ax.plot( [trend_start_frac,
                  now_frac],
                 [ b,
                   m * (now_frac - trend_start_frac) + b],
                 'r-')
    ax1.set_ylim(ax2.get_ylim())

    for ax in [ax1, ax2]:
        curr_xlims = ax.get_xlim()
        dxlim = curr_xlims[1]-curr_xlims[0]
        ax.set_xlim(curr_xlims[0]-time_pad*dxlim,
                    curr_xlims[1]+time_pad*dxlim)
    
        #    ax = fig.get_axes()[0]
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            label.set_size('small')
        ax.set_ylabel('Rate', fontsize=12)
        ax.set_title("%s acq_fail" % d, fontsize=12)


    fig1.savefig("summary_%s_acq_fail.png" % d)
    fig2.savefig("summary_%s_acq_fail_eb.png" % d)
    plt.close(fig1)
    plt.close(fig2)




import jinja2


#TASK_SHARE = os.path.join(os.environ['SKA'],'share', task)
TASK_SHARE = "."
#
jinja_env = jinja2.Environment(
	loader=jinja2.FileSystemLoader(os.path.join(TASK_SHARE, 'templates')))

outfile = 'acq_summary.html'
template = jinja_env.get_template('summary.html')
page = template.render()
f = open(outfile, 'w')
f.write(page)
f.close()
                
