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
import scipy.stats

import Ska.DBI
from Chandra.Time import DateTime

# local to project
import Ska.Matplotlib
import Ska.report_ranges
from star_error import high_low_rate

task = 'acq_stat_reports'
TASK_SHARE = os.path.join(os.environ['SKA'],'share', task)
TASK_DATA = os.path.join(os.environ['SKA'],'data', task)
#TASK_SHARE = "."

jinja_env = jinja2.Environment(
	loader=jinja2.FileSystemLoader(os.path.join(TASK_SHARE, 'templates')))

logger = logging.getLogger(task)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')


def get_options():
    from optparse import OptionParser
    parser = OptionParser()
    parser.set_defaults()
    parser.add_option("--datadir",
                      default="/proj/sot/ska/data/acq_stat_reports/",
                      help="Output directory")
    parser.add_option("--webdir",
                      default="/proj/sot/ska/www/ASPECT/acq_stat_reports/",
                      help="Output directory")
    parser.add_option("--url",
		      default="/mta/ASPECT/acq_stat_reports/")
    parser.add_option("--start_time",
                  default=None)
    parser.add_option("--days_back",
		      default=30,
		      type='int')
    parser.add_option("--verbose",
                      type='int',
                      default=1,
                      help="Verbosity (0=quiet, 1=normal, 2=debug)")
    opt, args = parser.parse_args()
    return opt, args


def frac_points( acq, mag_bin, binstart=None ):
    """
    Calculate the fraction of NOID stars in the bins requested, where acq is a recarray
    of the acq stars, mag_bin is the mag range width of the binning, and binstart is
    a list of the left start of the bins.

    Return the left edge of the bins (x), the fraction of good acq stars (fracs),
    the low limit on the error of fracs (err_low), and the high limit on the error
    of fraces (err_high).  Returns a list of lists.
    """
    
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
    h=plt.figure(figsize=figsize)
    mag_bin = .1
    good = range_acqs[ range_acqs.obc_id == 'ID' ]
    # use unfilled histograms from a scipy example
    (bins, data) = Ska.Matplotlib.hist_outline(good.mag,
					   bins=np.arange(5.5-(mag_bin/2),
							  12+(mag_bin/2),mag_bin))
    plt.semilogy(bins, data+tiny_y, 'k-')
    bad = range_acqs[ range_acqs.obc_id == 'NOID' ]

    (bins, data) = Ska.Matplotlib.hist_outline(bad.mag,
					   bins=np.arange(5.5-(mag_bin/2),
							  12+(mag_bin/2),mag_bin))
    plt.semilogy(bins, 100*data+tiny_y, 'r-')
    plt.xlabel('Star magnitude (mag)')
    plt.ylabel('N stars (red is x100)')
    plt.xlim(5,12)
    plt.title('N good (black) and bad (red) stars vs Mag')
    plt.subplots_adjust(top=.85, bottom=.17, right=.97)
    plt.savefig(os.path.join(outdir, 'mag_histogram.png'))
    plt.close(h)

    # Scaled Failure Histogram, tail mag range
    h=plt.figure(figsize=figsize)
    mag_bin = .05
    (bins, data) = Ska.Matplotlib.hist_outline(good.mag,
					   bins=np.arange(5.5-(mag_bin/2),
							  12+(mag_bin/2),mag_bin))
    plt.semilogy(bins, data+tiny_y, 'k-')
    bad = range_acqs[ range_acqs.obc_id == 'NOID' ]
    (bins, data) = Ska.Matplotlib.hist_outline(bad.mag,
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
    plt.close(h)

    # Acquisition Success Fraction, full mag range
    h=plt.figure(figsize=figsize)
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
    plt.close(h)

    # Acquisition Success Fraction, tail mag range
    h=plt.figure(figsize=figsize)
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
    plt.close(h)

    h=plt.figure(figsize=figsize)
    plt.hist(range_acqs.mag, bins=np.arange(5.5-(mag_bin/2),12+(mag_bin/2),mag_bin), histtype='step',color='black',normed=True)
    plt.hist(acqs.mag, bins=np.arange(5.5-(mag_bin/2),12+(mag_bin/2),mag_bin), histtype='step',color='red',normed=True)
    plt.xlabel('Star magnitude (mag)')
    plt.ylabel('Fraction of All Acq Stars')
    plt.title('Expected Magnitudes of Acquisition Stars')
    plt.ylim(0,1.0)
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
    plt.close(h)


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
    h=plt.figure(figsize=figsize)
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
    plt.close(h)

    # N ids per obsid over mission
    id_by_obsid_all = ids

    h=plt.figure(figsize=figsize)
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
    plt.close(h)

def make_html( nav_dict, rep_dict, fail_dict, outdir):
    """
    Render and write the basic page, where nav_dict is a dictionary of the
    navigation elements (locations of UP_TO_MAIN, NEXT, PREV), rep_dict is
    a dictionary of the main data elements (n failures etc), fail_dict
    contains the elements required for the extra table of failures at the
    bottom of the page, and outdir is the destination directory.
    """

    template = jinja_env.get_template('index.html')
    page = template.render(nav=nav_dict, fails=fail_dict, rep=rep_dict)
    f = open(os.path.join(outdir, 'index.html'), 'w')
    f.write(page)
    f.close()

class NoStarError(Exception):
    """
    Special error for the case when no acquisition stars are found.
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def acq_info( acqs, tname, mxdatestart, mxdatestop, pred):
    """
    Generate a report dictionary for the time range.

    :param acqs: recarray of all acquisition stars available in the table
    :param tname: timerange string (e.g. 2010-M05)
    :param mxdatestart: mx.DateTime of start of reporting interval
    :param mxdatestop: mxDateTime of end of reporting interval
    :param pred_start: date for beginning of time range for predictions based
    on average from pred_start to now()

    :rtype: dict of report values
    """
	
    rep = { 'datestring' : tname,
	    'datestart' : DateTime(mxdatestart).date,
	    'datestop' : DateTime(mxdatestop).date,
	    'human_date_start' : mxdatestart.strftime("%d-%B-%Y"),
	    'human_date_stop' : mxdatestop.strftime("%d-%B-%Y"),
	    }

    range_acqs = acqs[ (acqs.tstart >= DateTime(mxdatestart).secs)
		       & (acqs.tstart < DateTime(mxdatestop).secs) ]
    #pred_acqs =  acqs[ (acqs.tstart >= DateTime(pred_start).secs)
	#	       & (acqs.tstart < DateTime(mxdatestop).secs) ]

    rep['n_stars'] = len(range_acqs)
    if not len(range_acqs):
        raise NoStarError("No acq stars in range")
    rep['n_failed'] = len(np.flatnonzero(range_acqs.obc_id == 'NOID'))
    rep['fail_rate'] = 1.0*rep['n_failed']/rep['n_stars']

    mean_time = (DateTime(rep['datestart']).secs + DateTime(rep['datestop']).secs) / 2.0
    mean_time_d_year = (mean_time - pred['time0']) / (86400 * 365.25)
    rep['fail_rate_pred'] = mean_time_d_year * pred['m'] + pred['b']
    rep['n_failed_pred'] = int(round(rep['fail_rate_pred'] * rep['n_stars']))


    rep['prob_less'] = scipy.stats.poisson.cdf( rep['n_failed'], rep['n_failed_pred'] )
    rep['prob_more'] = 1 - scipy.stats.poisson.cdf( rep['n_failed'] - 1,
						    rep['n_failed_pred'] )

    rep['fail_rate_err_high'], rep['fail_rate_err_low'] = high_low_rate( rep['n_failed'],
                                                                         rep['n_stars'])

    return rep

def make_fail_html( stars, outfile):
    """
    Render and write the expanded table of failed stars
    """
    nav_dict = dict(star_cgi='https://icxc.harvard.edu/cgi-bin/aspect/get_stats/get_stats.cgi?id=',
	       starcheck_cgi='https://icxc.harvard.edu/cgi-bin/aspect/starcheck_print/starcheck_print.cgi?sselect=obsid;obsid1=')
    template = jinja_env.get_template('stars.html')
    page = template.render(nav=nav_dict, failed_stars=stars)
    f = open(outfile, 'w')
    f.write(page)
    f.close()



def acq_fails( acqs, tname, mxdatestart, mxdatestop, outdir='out'):
    """
    Find the failures over the interval and find the tail mag failures.
    Pass the failures to make_fail_html for the main star table and the
    little ones (by mag bin).
    """

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
                        n_failed=len(
            np.flatnonzero(mag_range_acqs.obc_id == 'NOID')),			
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
    """
    Update acquisition statistics plots.  Mission averages are computed with all stars
    from 2000:001 to the end of the interval.
    """

    import time
    nowdate = time.ctime()
    logger.info("---------- acq stat reports update at %s ----------" % (nowdate))
    
    sqlaca = Ska.DBI.DBI(dbi='sybase', server='sybase', user='aca_read', database='aca', numpy=True)
    min_acq_time = DateTime('2000:001:00:00:00.000')

    all_acq = sqlaca.fetchall('select * from acq_stats_data where tstart >= %f'
                              % min_acq_time.secs )

    # use the acq_stats_id_by_obsid view for a quick count of the number of ID/NOID
    # stars in each obsid.  Used by make_id_plots()
    all_id = sqlaca.fetchall('select * from acq_stats_id_by_obsid where tstart >= %f' % 
                             min_acq_time.secs )

    if opt.start_time is None:
        to_update = Ska.report_ranges.get_update_ranges(opt.days_back)
    else:
        import time
        now = DateTime(time.time(), format='unix').mxDateTime
        start = DateTime(opt.start_time).mxDateTime
        delta = now - start
        to_update = Ska.report_ranges.get_update_ranges(int(delta.days))
        

    for tname in sorted(to_update.keys()):
        logger.debug("Attempting to update %s" % tname )
        mxdatestart = to_update[tname]['start']
        mxdatestop = to_update[tname]['stop']

        # ignore acquisition stars that are newer than the end of the range
        # in question (happens during reprocessing) for consistency
        all_acq_upto = all_acq[ all_acq.tstart <= DateTime(mxdatestop).secs ]
        all_id_upto = all_id[ all_id.tstart <= DateTime(mxdatestop).secs ]

        webout = os.path.join(opt.webdir,
                              "%s" % to_update[tname]['year'],
                              to_update[tname]['subid'])
        if not os.path.exists(webout):
            os.makedirs(webout)
        dataout = os.path.join(opt.datadir,
                               "%s" % to_update[tname]['year'],
                               to_update[tname]['subid'])
        if not os.path.exists(dataout):
            os.makedirs(dataout)

        logger.debug("Plots and HTML to %s" % webout)
        logger.debug("JSON to  %s" % dataout)


        import json
        pred = json.load(open(os.path.join(TASK_DATA, 'acq_fail_fitfile.json')))
        rep = acq_info(all_acq_upto, tname, mxdatestart, mxdatestop, pred)

        rep_file = open(os.path.join(dataout, 'rep.json'), 'w')
        rep_file.write(json.dumps(rep, sort_keys=True, indent=4))
        rep_file.close()

        fails = acq_fails(all_acq_upto,
                          tname,
                          mxdatestart,
                          mxdatestop,
                          outdir=webout)
        fail_file = open(os.path.join(dataout, 'fail.json'), 'w')
        fail_file.write(json.dumps(fails, sort_keys=True, indent=4))
        fail_file.close()

        prev_range = Ska.report_ranges.get_prev(to_update[tname])
        next_range = Ska.report_ranges.get_next(to_update[tname])
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

        make_acq_plots( all_acq_upto,
                        tstart=DateTime(mxdatestart).secs,
                        tstop=DateTime(mxdatestop).secs,
                        outdir=webout)
        make_id_plots( all_id_upto,
                       tstart=DateTime(mxdatestart).secs,
                       tstop=DateTime(mxdatestop).secs,
                       outdir=webout)
        make_html(nav, rep, fails, outdir=webout)
        #except Exception, msg:
	    #print "ERROR: Unable to process %s" % tname, msg
	



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
