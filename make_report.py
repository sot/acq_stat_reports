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

import jinja2

import Ska.DBI
from Chandra.Time import DateTime


import histOutline
import timerange

task = 'acq_stat_reports'
#TASK_SHARE = os.path.join(os.environ['SKA'],'share', task)
TASK_SHARE = "."

jinja_env = jinja2.Environment(
	loader=jinja2.FileSystemLoader(os.path.join(TASK_SHARE, 'templates')))

logger = logging.getLogger(task)


def get_options():
    from optparse import OptionParser
    parser = OptionParser()
    parser.set_defaults()
    parser.add_option("--outdir",
                      default="/proj/sot/ska/www/ASPECT/acq_stat_reports/",
                      help="Output directory")
    parser.add_option("--url",
		      default="http://cxc.harvard.edu/mta/ASPECT/acq_stat_reports/")
    parser.add_option("--verbose",
                      type='int',
                      default=1,
                      help="Verbosity (0=quiet, 1=normal, 2=debug)")
    opt, args = parser.parse_args()
    return opt, args


def frac_points( acq, mag_bin, binstart=None ):

    x = []
    fracs = []
    err_high = []
    err_low = []
    
    for rstart in binstart:
        x.append(rstart+(mag_bin/2))
        range_acq = acq[ (acq.mag >= rstart) & (acq.mag < rstart+mag_bin)]
        if len(range_acq):
            good = (len(range_acq) - len(np.flatnonzero(range_acq.obc_id == 'NOID')))
            frac = good/(len(range_acq)*1.0)
            fracs.append(frac)
            err_low_lim = np.sqrt(good)/(len(range_acq)*1.0)
            if err_low_lim + frac < 1:
                err_high_lim = err_low_lim
            else:
                err_high_lim = 1 - frac
            #            print frac, '\t', err_low_lim, '\t', err_high_lim
            err_high.append(err_high_lim)
            err_low.append(err_low_lim)
        else:
            fracs.append(0)
            err_high.append(0)
            err_low.append(0)

    return (x, fracs, err_low, err_high)


def make_acq_plots( acqs, tstart=0, tstop=DateTime().secs, outdir="plots"):
    """Make range of acquisition statistics plots:
    mag_histogram.png - histogram of acq failures, full mag range
    zoom_mag_histogram.png - histogram of acq failures, tail mag range
    mag_pointhist.png
    zoom_mag_pointhist.png
    exp_mag_histogram.png
    delta_mag_scatter.png

    :param acqs: all mission acq stars as recarray from Ska.DBI.fetchall
    :param tstart: range of interest tstart (Chandra secs)
    :param tstop: range of interest tstop (Chandra secs)
    :param outdir: output directory for pngs
    :rtype: None
    
    """
    

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    figsize=(5,2.5)
    tiny_y = .1
    range_acqs = acqs[ (acqs.tstart >= tstart) & (acqs.tstart < tstop ) ]

    # Scaled Failure Histogram, full mag range
    plt.figure(figsize=figsize)
    mag_bin = .1
    good = range_acqs[ range_acqs.obc_id == 'ID' ]
    # use unfilled histograms from a scipy example
    (bins, data) = histOutline.histOutline(good.mag,
					   bins=np.arange(5.5-(mag_bin/2),
							  12+(mag_bin/2),mag_bin))
    plt.semilogy(bins, data+tiny_y, 'k-')
    bad = range_acqs[ range_acqs.obc_id == 'NOID' ]

    (bins, data) = histOutline.histOutline(bad.mag,
					   bins=np.arange(5.5-(mag_bin/2),
							  12+(mag_bin/2),mag_bin))
    plt.semilogy(bins, 100*data+tiny_y, 'r-')
    plt.xlabel('Star magnitude (mag)')
    plt.ylabel('N stars (red is x100)')
    plt.xlim(5,12)
    plt.title('N good (black) and bad (red) stars vs Mag')
    plt.subplots_adjust(top=.85, bottom=.17, right=.97)
    plt.savefig(os.path.join(outdir, 'mag_histogram.png'))

    # Scaled Failure Histogram, tail mag range
    plt.figure(figsize=figsize)
    mag_bin = .05
    (bins, data) = histOutline.histOutline(good.mag,
					   bins=np.arange(5.5-(mag_bin/2),
							  12+(mag_bin/2),mag_bin))
    plt.semilogy(bins, data+tiny_y, 'k-')
    bad = range_acqs[ range_acqs.obc_id == 'NOID' ]
    (bins, data) = histOutline.histOutline(bad.mag,
					   bins=np.arange(5.5-(mag_bin/2),
							  12+(mag_bin/2),mag_bin))
    plt.semilogy(bins, 100*data+tiny_y, 'r-')
#    plt.ylim(1,1000)
    plt.xlim(10,11)
    plt.xlabel('Star magnitude (mag)')
    plt.ylabel('N stars (red is x100)')
    plt.title('N good (black) and bad (red) stars vs Mag')
    plt.subplots_adjust(top=.85, bottom=.17, right=.97)
    plt.savefig(os.path.join(outdir, 'zoom_mag_histogram.png'))

    # Acquisition Success Fraction, full mag range
    plt.figure(figsize=figsize)
    mag_bin = .1
    (x, fracs, err_low, err_high ) = frac_points( range_acqs , mag_bin, binstart=np.arange(5.5-(mag_bin/2),12+(mag_bin/2),mag_bin))
    plt.errorbar( x, fracs, yerr=[err_low, err_high], color='black', marker='.', linestyle='None')    
    (x, fracs, err_low, err_high ) = frac_points( acqs, mag_bin, binstart=np.arange(5.5-(mag_bin/2),12+(mag_bin/2),mag_bin))
    plt.errorbar( x, fracs, yerr=[err_low, err_high], color='red', marker='.', linestyle='None')    
    plt.xlim(5,12)
    plt.ylim(-.05,1.05)
    plt.xlabel('Star magnitude (mag)')
    plt.ylabel('Fraction Acquired')
    plt.title('Acquisition Success vs Expected Mag')
    plt.subplots_adjust(top=.85, bottom=.17, right=.97)
    plt.savefig(os.path.join(outdir, 'mag_pointhist.png'))

    # Acquisition Success Fraction, tail mag range
    plt.figure(figsize=figsize)
    mag_bin = 0.05
    (x, fracs, err_low, err_high ) = frac_points( range_acqs , mag_bin, binstart=np.arange(10-(mag_bin/2),11+(mag_bin/2),mag_bin))
    plt.errorbar( x, fracs, yerr=[err_low, err_high], color='black', marker='.', linestyle='None')    
    (x, fracs, err_low, err_high ) = frac_points( acqs, mag_bin, binstart=np.arange(10-(mag_bin/2),11+(mag_bin/2),mag_bin))
    plt.errorbar( x, fracs, yerr=[err_low, err_high], color='red', marker='.', linestyle='None')    
    plt.ylim(-.05,1.05)
    plt.xlim(10,11)
    plt.xlabel('Star magnitude (mag)')
    plt.ylabel('Fraction Acquired')
    plt.title('Acquisition Success vs Expected Mag')
    plt.subplots_adjust(top=.85, bottom=.17, right=.97)
    plt.savefig(os.path.join(outdir, 'zoom_mag_pointhist.png'))


    plt.figure(figsize=figsize)
    plt.hist(range_acqs.mag, bins=np.arange(5.5-(mag_bin/2),12+(mag_bin/2),mag_bin), histtype='step',color='black',normed=True)
    plt.hist(acqs.mag, bins=np.arange(5.5-(mag_bin/2),12+(mag_bin/2),mag_bin), histtype='step',color='red',normed=True)
    plt.xlabel('Star magnitude (mag)')
    plt.ylabel('Fraction of All Acq Stars')
    plt.title('Expected Magnitudes of Acquisition Stars')
    plt.subplots_adjust(top=.85, bottom=.17, right=.97)
    plt.savefig(os.path.join(outdir, 'exp_mag_histogram.png'))

    plt.figure(figsize=figsize)
    ok = range_acqs.obc_id == 'ID'
    plt.plot(range_acqs[ok].mag, range_acqs[ok].mag_obs-range_acqs[ok].mag, 'k.',markersize=3)
    plt.xlabel('AGASC magnitude (mag)')
    plt.ylabel('Observed - AGASC mag')
    plt.title('Delta Mag vs Mag')
    plt.subplots_adjust(top=.85, bottom=.17, right=.97)
    plt.savefig(os.path.join(outdir, 'delta_mag_scatter.png'))



def make_id_plots( ids, tstart=0, tstop=DateTime().secs, outdir="plots"):
    """Make acquisition statistics plots related to IDs by obsid
    id_per_obsid_histogram.png
    id_per_obsid_mission_histogram.png

    :param ids: all mission ids by obsid recarray from Ska.DBI.fetchall
    :param tstart: range of interest tstart (Chandra secs)
    :param tstop: range of interest tstop (Chandra secs)
    :param outdir: output directory for pngs
    :rtype: None
    
    """
    

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # range N ids per obsid
    id_by_obsid = ids[ (ids.tstart >= tstart) & (ids.tstart < tstop ) ]

    figsize=(5,2.5)
    plt.figure(figsize=figsize)
    plt.hist(id_by_obsid.id_cnt, bins=np.arange(-.5,9,1), log=True, histtype='step',color='black')
    plt.hist(id_by_obsid.noid_cnt, bins=np.arange(-.5,9,1), log=True, histtype='step',color='red')
    ylims = plt.ylim()
    plt.ylim(.1,ylims[1]*2)
    plt.xlim(-1,9)
    plt.xlabel('Number of Acq Stars (red=NOID)')
    plt.ylabel('Number of obsids')
    plt.title('ID Acq Stars per obsid (Interval)')
    plt.subplots_adjust(top=.85, bottom=.17, right=.97)
    plt.savefig(os.path.join(outdir, 'id_per_obsid_histogram.png'))

    # N ids per obsid over mission
    id_by_obsid_all = ids

    plt.figure(figsize=figsize)
    plt.hist(id_by_obsid_all.id_cnt, bins=np.arange(-.5,9,1), log=True, histtype='step',color='black')
    plt.hist(id_by_obsid_all.noid_cnt, bins=np.arange(-.5,9,1), log=True, histtype='step',color='red')
    ylims = plt.ylim()
    plt.ylim(.1,ylims[1]*2)
    plt.xlim(-1,9)
    plt.xlabel('Number of Acq Stars (red=NOID)')
    plt.ylabel('Number of obsids')
    plt.title('ID Acq Stars per obsid (Mission)')
    plt.subplots_adjust(top=.85, bottom=.17, right=.97)
    plt.savefig(os.path.join(outdir, 'id_per_obsid_mission_histogram.png'))

def make_html( nav_dict, rep_dict, fail_dict, outdir):
    template = jinja_env.get_template('index.html')
    page = template.render(nav=nav_dict, fails=fail_dict, rep=rep_dict)
    f = open(os.path.join(outdir, 'index.html'), 'w')
    f.write(page)
    f.close()

class NoStarError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def acq_info( acqs, tname, mxdatestart, mxdatestop,
	      pred_start='2003:001:00:00:00.000' ):
    rep = { 'datestring' : tname,
	    'datestart' : DateTime(mxdatestart).date,
	    'datestop' : DateTime(mxdatestop).date,
	    'human_date_start' : mxdatestart.strftime("%d-%B-%Y"),
	    'human_date_stop' : mxdatestop.strftime("%d-%B-%Y"),
	    }

    range_acqs = acqs[ (acqs.tstart >= DateTime(mxdatestart).secs)
		       & (acqs.tstart < DateTime(mxdatestop).secs) ]
    pred_acqs =  acqs[ (acqs.tstart >= DateTime(pred_start).secs)
		       & (acqs.tstart < DateTime(mxdatestop).secs) ]

    rep['n_stars'] = len(range_acqs)
    if not len(range_acqs):
        raise NoStarError("No acq stars in range")
    rep['n_failed'] = len(np.flatnonzero(range_acqs.obc_id == 'NOID'))
    rep['fail_rate'] = rep['n_failed']*1.0/rep['n_stars']

    rep['fail_rate_pred'] = ( len(np.flatnonzero(pred_acqs.obc_id == 'NOID'))*1.0
                              / len(pred_acqs))
    rep['n_failed_pred'] = int(round(rep['fail_rate_pred'] * rep['n_stars']))

    import scipy.stats
    rep['prob_less'] = scipy.stats.poisson.cdf( rep['n_failed'], rep['n_failed_pred'] )
    rep['prob_more'] = 1 - scipy.stats.poisson.cdf( rep['n_failed'] - 1,
						    rep['n_failed_pred'] )
    
    return rep

def make_fail_html( stars, outfile):

    nav_dict = dict(star_cgi='https://icxc.harvard.edu/cgi-bin/aspect/get_stats/get_stats.cgi?id=',
	       starcheck_cgi='https://icxc.harvard.edu/cgi-bin/aspect/starcheck_print/starcheck_print.cgi?sselect=obsid;obsid1=')
    template = jinja_env.get_template('stars.html')
    page = template.render(nav=nav_dict, failed_stars=stars)
    f = open(outfile, 'w')
    f.write(page)
    f.close()



def acq_fails( acqs, tname, mxdatestart, mxdatestop,
	      pred_start='2003:001:00:00:00.000', outdir='out'):
    fails = []
    range_acqs = acqs[ (acqs.tstart >= DateTime(mxdatestart).secs)
		       & (acqs.tstart < DateTime(mxdatestop).secs) ]

    all_fails = range_acqs[ range_acqs.obc_id == 'NOID' ]
    failed_stars = []
    for fail in all_fails:
        star=dict(id=fail.agasc_id,
		  obsid=fail.obsid,
		  mag_exp=fail.mag,
		  mag_obs=fail.mag_obs,
		  acq_stat='NOID')
	failed_stars.append(star)
    make_fail_html(failed_stars, os.path.join(outdir, 'failed_acq_stars_list.html'))

    bin = .1
    for tmag_start in np.arange(10.0,10.8,.1):

        mag_range_acqs = range_acqs[ (range_acqs.mag >= tmag_start)
				     & (range_acqs.mag < (tmag_start + bin))]

        mfailed_stars = []
	mag_fail_acqs = mag_range_acqs[ mag_range_acqs.obc_id == 'NOID' ]
	for mfail in mag_fail_acqs:
	    star=dict(id=mfail.agasc_id,
		      obsid=mfail.obsid,
		      mag_exp=mfail.mag,
		      mag_obs=mfail.mag_obs,
		      acq_stat='NOID')
	mfailed_stars.append(star)
	failed_star_file = "failed_acq_%.1f_stars_list.html" % tmag_start
	make_fail_html(mfailed_stars, os.path.join(outdir, failed_star_file))


	mag_fail = dict(n_stars=len(mag_range_acqs),
			n_failed=len(np.flatnonzero(mag_range_acqs.obc_id == 'NOID')),			
		       )
	if mag_fail['n_stars'] == 0:
	    mag_fail['fail_rate'] = 0
	else:
	    mag_fail['fail_rate'] = mag_fail['n_failed']*1.0/mag_fail['n_stars']
	mag_fail['label'] = "%0.1f-%0.1f" % (tmag_start,tmag_start+bin)
	mag_fail['fail_file'] = failed_star_file
	fails.append(mag_fail)



    return fails





def main(opt):
    sqlaca = Ska.DBI.DBI(dbi='sybase', server='sybase', numpy=True)
    min_acq_time = DateTime('2003:001:00:00:00.000')

    all_acq = sqlaca.fetchall('select * from acq_stats_data where tstart >= %f'
                              % min_acq_time.secs )

    all_id = sqlaca.fetchall('select * from acq_stats_id_by_obsid where tstart >= %f' % 
                             min_acq_time.secs )

    to_update = timerange.get_update_ranges()

    for tname in to_update.keys():

        try:
	    mxdatestart = to_update[tname]['start']
	    mxdatestop = to_update[tname]['stop']


	    out = os.path.join(opt.outdir,
			       "%s" % to_update[tname]['year'],
			       to_update[tname]['subid'])

	    log.debug("Writing to %s" % out)
	    if not os.path.exists(out):
		    os.makedirs(out)

	    rep = acq_info(all_acq, tname, mxdatestart, mxdatestop)
		    
	    import json
	    rep_file = open(os.path.join(out, 'rep.json'), 'w')
	    rep_file.write(json.dumps(rep, sort_keys=True, indent=4))
	    rep_file.close()

	    fails = acq_fails(all_acq, tname, mxdatestart, mxdatestop, outdir=out)
	    fail_file = open(os.path.join(out, 'fail.json'), 'w')
	    fail_file.write(json.dumps(fails, sort_keys=True, indent=4))
	    fail_file.close()

	    prev_range = timerange.get_prev(to_update[tname])
	    next_range = timerange.get_next(to_update[tname])
	    nav = dict(main=opt.url,
		       next="%s/%s/%s/%s" % (opt.url,
					     next_range['year'],
					     next_range['subid'],
					     'index.html'),
		       prev="%s/%s/%s/%s" % (opt.url,
					     prev_range['year'],
					     prev_range['subid'],
					     'index.html'),
		       )

	    make_acq_plots( all_acq,
			    tstart=DateTime(mxdatestart).secs,
			    tstop=DateTime(mxdatestop).secs,
			    outdir=out)
	    make_id_plots( all_id,
			   tstart=DateTime(mxdatestart).secs,
			   tstop=DateTime(mxdatestop).secs,
			   outdir=out)

	    make_html(nav, rep, fails, outdir=out)
	except Exception, msg:
	    print "ERROR: Unable to process %s" % tname, msg
	



if __name__ == '__main__':
    opt, args = get_options()
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    if opt.verbose == 2:
	    ch.setLevel(logging.DEBUG)
    if opt.verbose == 0:
	    ch.setLevel(logging.ERROR)
    logger.addHandler(ch) 

    main(opt)
