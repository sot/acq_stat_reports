
import Ska.DBI
from Chandra.Time import DateTime
import os
import numpy as np

# Matplotlib setup
# Use Agg backend for command-line (non-interactive) operation
import matplotlib
if __name__ == '__main__':
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        

import histOutline


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

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    figsize=(5,2.5)
    tiny_y = .1
    range_acqs = acqs[ (acqs.tstart >= tstart) & (acqs.tstart < datestop ) ]

    plt.figure(figsize=figsize)
    mag_bin = .1
    good = range_acqs[ range_acqs.obc_id == 'ID' ]
    (bins, data) = histOutline.histOutline(good.mag,
					   bins=np.arange(5.5-(mag_bin/2),
							  12+(mag_bin/2),mag_bin))
    semilogy(bins, data+tiny_y, 'k-')
    bad = range_acqs[ range_acqs.obc_id == 'NOID' ]
    (bins, data) = histOutline.histOutline(bad.mag,
					   bins=np.arange(5.5-(mag_bin/2),
							  12+(mag_bin/2),mag_bin))
    semilogy(bins, 100*data+tiny_y, 'r-')
    plt.xlabel('Star magnitude (mag)')
    plt.ylabel('N stars (red is x100)')
    plt.title('N good (black) and bad (red) stars vs Mag')
    plt.savefig(os.path.join(outdir, 'mag_histogram.png'))

    plt.figure(figsize=figsize)
    mag_bin = .05
    (bins, data) = histOutline.histOutline(good.mag,
					   bins=np.arange(5.5-(mag_bin/2),
							  12+(mag_bin/2),mag_bin))
    semilogy(bins, data+tiny_y, 'k-')
    bad = range_acqs[ range_acqs.obc_id == 'NOID' ]
    (bins, data) = histOutline.histOutline(bad.mag,
					   bins=np.arange(5.5-(mag_bin/2),
							  12+(mag_bin/2),mag_bin))
    semilogy(bins, 100*data+tiny_y, 'r-')
#    plt.ylim(1,1000)
    plt.xlim(10,11)
    plt.xlabel('Star magnitude (mag)')
    plt.ylabel('N stars (red is x100)')
    plt.title('N good (black) and bad (red) stars vs Mag')
    plt.savefig(os.path.join(outdir, 'zoom_mag_histogram.png'))


    plt.figure(figsize=figsize)
    mag_bin = .1
    (x, fracs, err_low, err_high ) = frac_points( range_acqs , mag_bin, binstart=np.arange(5.5-(mag_bin/2),12+(mag_bin/2),mag_bin))
    plt.errorbar( x, fracs, yerr=[err_low, err_high], color='black', marker='.', linestyle='None')    
    (x, fracs, err_low, err_high ) = frac_points( acqs, mag_bin, binstart=np.arange(5.5-(mag_bin/2),12+(mag_bin/2),mag_bin))
    plt.errorbar( x, fracs, yerr=[err_low, err_high], color='red', marker='.', linestyle='None')    
    plt.ylim(-.05,1.05)
    plt.xlabel('Star magnitude (mag)')
    plt.ylabel('Fraction Acquired')
    plt.title('Acquisition Success vs Expected Mag')
    plt.savefig(os.path.join(outdir, 'mag_pointhist.png'))


    plt.figure(figsize=figsize)
    mag_bin = 0.05
    (x, fracs, err_low, err_high ) = frac_points( range_acqs , mag_bin, binstart=np.arange(10-(mag_bin/2),11+(mag_bin/2),mag_bin))
    plt.errorbar( x, fracs, yerr=[err_low, err_high], color='black', marker='.', linestyle='None')    
    (x, fracs, err_low, err_high ) = frac_points( acqs, mag_bin, binstart=np.arange(10-(mag_bin/2),11+(mag_bin/2),mag_bin))
    plt.errorbar( x, fracs, yerr=[err_low, err_high], color='red', marker='.', linestyle='None')    
    plt.ylim(-.05,1.05)
    plt.xlabel('Star magnitude (mag)')
    plt.ylabel('Fraction Acquired')
    plt.title('Acquisition Success vs Expected Mag')
    plt.savefig(os.path.join(outdir, 'zoom_mag_pointhist.png'))


    plt.figure(figsize=figsize)
    plt.hist(range_acqs.mag, bins=np.arange(5.5-(mag_bin/2),12+(mag_bin/2),mag_bin), histtype='step',color='black',normed=True)
    plt.hist(acqs.mag, bins=np.arange(5.5-(mag_bin/2),12+(mag_bin/2),mag_bin), histtype='step',color='red',normed=True)
    plt.xlabel('Star magnitude (mag)')
    plt.ylabel('Fraction of All Acq Stars')
    plt.title('Expected Magnitudes of Acquisition Stars')
    plt.savefig(os.path.join(outdir, 'exp_mag_histogram.png'))

    plt.figure(figsize=figsize)
    ok = range_acqs.obc_id == 'ID'
    plt.plot(range_acqs[ok].mag, range_acqs[ok].mag_obs-range_acqs[ok].mag, 'k.',markersize=3)
    plt.xlabel('AGASC magnitude (mag)')
    plt.ylabel('Observed - AGASC mag')
    plt.title('Delta Mag vs Mag')
    plt.savefig(os.path.join(outdir, 'delta_mag_scatter.png'))



def make_id_plots( ids, tstart=0, tstop=DateTime().secs, outdir="plots"):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

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
    plt.savefig(os.path.join(outdir, 'id_per_obsid_histogram.png'))


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
    plt.savefig(os.path.join(outdir, 'id_per_obsid_mission_histogram.png'))



sqlaca = Ska.DBI.DBI(dbi='sybase', server='sybase', numpy=True)
min_acq_time = DateTime('2003:001:00:00:00.000')

if 'all_acq' not in globals():
    all_acq = sqlaca.fetchall('select * from acq_stats_data where tstart >= %f'
                              % min_acq_time.secs )

if 'all_id' not in globals():
    all_id = sqlaca.fetchall('select * from acq_stats_id_by_obsid where tstart >= %f' % 
                             min_acq_time.secs )


datestart = DateTime('2010-01-01T00:00:00.000')
datestop = DateTime('2010-02-01T00:00:00.000')

make_acq_plots( all_acq, tstart=datestart.secs, tstop=datestop.secs)
make_id_plots( all_id, tstart=datestart.secs, tstop=datestop.secs)
