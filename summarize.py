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
import Ska.report_ranges


time_pad = .1

datadir = '/proj/sot/ska/data/acq_stat_reports'

data = { 'month': glob(os.path.join(datadir, '????', 'M??', 'rep.json')),
         'quarter': glob(os.path.join(datadir, '????', 'Q?', 'rep.json')),
         'semi': glob(os.path.join(datadir, '????', 'S?', 'rep.json')),
         'year': glob(os.path.join(datadir, '????', 'YEAR', 'rep.json')),}


pred = json.load(open('acq_fail_fitfile.json'))

rates = dict([(ttype, dict([ (ftype, dict(time=[],
                                          rate=[],
                                          err_h=[],
                                          err_l=[]))
                             for ftype in ['fail',]]))
              for ttype in data.keys()])

now_mxd = DateTime().mxDateTime
now_frac = now_mxd.year + now_mxd.day_of_year / 365.25

for d in data.keys():

    data[d].sort()
    for p in data[d]:
        rep_file = open(p, 'r')
        rep_text = rep_file.read()
        rep = json.loads(rep_text)
        ftype='fail'
        mxd = DateTime( (DateTime(rep['datestart']).secs
                         +  DateTime(rep['datestop']).secs) / 2).mxDateTime
        frac_year = mxd.day_of_year / 365.25
        rates[d][ftype]['time'].append(mxd.year + frac_year)
        rates[d][ftype]['rate'].append(rep['fail_rate'])
        rates[d][ftype]['err_h'].append(rep['fail_rate_err_high'])
        rates[d][ftype]['err_l'].append(rep['fail_rate_err_low'])
        curr_color='black'
        #ptime = DateTime(rep['datestart']).secs
        
    for rkey in rates[d][ftype]:
        rates[d][ftype][rkey] = np.array(rates[d][ftype][rkey])


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



    # plot prediction for trending start through a year from now
    for ax in [ax1, ax2]:
        ax.plot( [pred['time0'],
                  now_frac + 1],
                 [ pred['b'],
                   pred['m'] * (now_frac + 1 - pred['time0']) + pred['b']],
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
                
