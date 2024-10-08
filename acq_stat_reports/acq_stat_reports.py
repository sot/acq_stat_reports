"""
Generate acquisition statistics report.
"""

import argparse
import json
import os
from pathlib import Path

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import ska_matplotlib
import ska_report_ranges
from astropy import units as u
from astropy.table import Table, join
from chandra_aca.star_probs import binomial_confidence_interval
from cxotime import CxoTime
from ska_helpers import logging

SKA = Path(os.environ["SKA"])

JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates" / "acq_stats")
)
logger = logging.basic_logger("acq_stat_reports", level="INFO")


class NoStarError(Exception):
    """
    Special error for the case when no acquisition stars are found.
    """


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datadir",
        default=Path(SKA / "data" / "acq_stat_reports"),
        help="Output directory",
        type=Path,
    )
    parser.add_argument(
        "--webdir",
        default=Path(SKA / "www" / "ASPECT" / "acq_stat_reports"),
        help="Output directory",
        type=Path,
    )
    parser.add_argument("--url", default="/mta/ASPECT/acq_stat_reports/")
    parser.add_argument("--start_time", default=None)
    parser.add_argument("--days_back", default=30, type=int)
    parser.add_argument(
        "-v",
        default="INFO",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
            "debug",
            "info",
            "warning",
            "error",
            "critical",
        ],
        help="Verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    return parser


def frac_points(acq, mag_bin, binstart=None):
    """Calculate the fraction of NOID stars in the bins requested.

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
        x.append(rstart + (mag_bin / 2))
        range_acq = acq[(acq["mag"] >= rstart) & (acq["mag"] < rstart + mag_bin)]
        if len(range_acq):
            good = len(range_acq) - len(np.flatnonzero(range_acq["acqid"] == 0))
            frac = good / (len(range_acq) * 1.0)
            fracs.append(frac)
            err_low_lim = np.sqrt(good) / (len(range_acq) * 1.0)
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


def make_acq_plots(acqs, tstart=0, tstop=None, outdir=None, close_figures=False):  # noqa: PLR0912, PLR0915
    """Make acquisition statistics plots.

    Make range of acquisition statistics plots:
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

    if tstop is None:
        tstop = CxoTime().cxcsec

    if outdir is not None and not outdir.exists():
        outdir.mkdir(parents=True)

    figsize = (5, 2.5)
    tiny_y = 0.1
    range_acqs = acqs[(acqs["tstart"] >= tstart) & (acqs["tstart"] < tstop)]

    # Scaled Failure Histogram, full mag range
    h = plt.figure(figsize=figsize)
    mag_bin = 0.1
    good = range_acqs[range_acqs["acqid"] == 1]
    # use unfilled histograms from a scipy example
    (bins, data) = ska_matplotlib.hist_outline(
        good["mag"], bins=np.arange(5.5 - (mag_bin / 2), 12 + (mag_bin / 2), mag_bin)
    )
    plt.semilogy(bins, data + tiny_y, "k-")
    bad = range_acqs[range_acqs["acqid"] == 0]

    (bins, data) = ska_matplotlib.hist_outline(
        bad["mag"], bins=np.arange(5.5 - (mag_bin / 2), 12 + (mag_bin / 2), mag_bin)
    )
    plt.semilogy(bins, 100 * data + tiny_y, "r-")
    plt.xlabel("Star magnitude (mag)")
    plt.ylabel("N stars (red is x100)")
    plt.xlim(5, 12)
    plt.title("N good (black) and bad (red) stars vs Mag")
    plt.subplots_adjust(top=0.85, bottom=0.17, right=0.97)
    if outdir:
        plt.savefig(outdir / "mag_histogram.png")
    if close_figures:
        plt.close(h)

    # Scaled Failure Histogram, tail mag range
    h = plt.figure(figsize=figsize)
    mag_bin = 0.05
    (bins, data) = ska_matplotlib.hist_outline(
        good["mag"], bins=np.arange(5.5 - (mag_bin / 2), 12 + (mag_bin / 2), mag_bin)
    )
    plt.semilogy(bins, data + tiny_y, "k-")
    bad = range_acqs[range_acqs["acqid"] == 0]
    (bins, data) = ska_matplotlib.hist_outline(
        bad["mag"], bins=np.arange(5.5 - (mag_bin / 2), 12 + (mag_bin / 2), mag_bin)
    )
    plt.semilogy(bins, 100 * data + tiny_y, "r-")
    #    plt.ylim(1,1000)
    plt.xlim(10, 11)
    plt.xlabel("Star magnitude (mag)")
    plt.ylabel("N stars (red is x100)")
    plt.title("N good (black) and bad (red) stars vs Mag")
    plt.subplots_adjust(top=0.85, bottom=0.17, right=0.97)
    if outdir:
        plt.savefig(outdir / "zoom_mag_histogram.png")
    if close_figures:
        plt.close(h)

    # Acquisition Success Fraction, full mag range
    h = plt.figure(figsize=figsize)
    mag_bin = 0.1
    (x, fracs, err_low, err_high) = frac_points(
        range_acqs,
        mag_bin,
        binstart=np.arange(5.5 - (mag_bin / 2), 12 + (mag_bin / 2), mag_bin),
    )
    plt.errorbar(
        x, fracs, yerr=[err_low, err_high], color="black", marker=".", linestyle="None"
    )
    (x, fracs, err_low, err_high) = frac_points(
        acqs,
        mag_bin,
        binstart=np.arange(5.5 - (mag_bin / 2), 12 + (mag_bin / 2), mag_bin),
    )
    plt.errorbar(
        x, fracs, yerr=[err_low, err_high], color="red", marker=".", linestyle="None"
    )
    plt.xlim(5, 12)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Star magnitude (mag)")
    plt.ylabel("Fraction Acquired")
    plt.title("Acquisition Success vs Expected Mag")
    plt.subplots_adjust(top=0.85, bottom=0.17, right=0.97)
    if outdir:
        plt.savefig(outdir / "mag_pointhist.png")
    if close_figures:
        plt.close(h)

    # Acquisition Success Fraction, tail mag range
    h = plt.figure(figsize=figsize)
    mag_bin = 0.05
    (x, fracs, err_low, err_high) = frac_points(
        range_acqs,
        mag_bin,
        binstart=np.arange(10 - (mag_bin / 2), 11 + (mag_bin / 2), mag_bin),
    )
    plt.errorbar(
        x, fracs, yerr=[err_low, err_high], color="black", marker=".", linestyle="None"
    )
    (x, fracs, err_low, err_high) = frac_points(
        acqs,
        mag_bin,
        binstart=np.arange(10 - (mag_bin / 2), 11 + (mag_bin / 2), mag_bin),
    )
    plt.errorbar(
        x, fracs, yerr=[err_low, err_high], color="red", marker=".", linestyle="None"
    )
    plt.ylim(-0.05, 1.05)
    plt.xlim(10, 11)
    plt.xlabel("Star magnitude (mag)")
    plt.ylabel("Fraction Acquired")
    plt.title("Acquisition Success vs Expected Mag")
    plt.subplots_adjust(top=0.85, bottom=0.17, right=0.97)
    if outdir:
        plt.savefig(outdir / "zoom_mag_pointhist.png")
    if close_figures:
        plt.close(h)

    h = plt.figure(figsize=figsize)
    plt.hist(
        range_acqs["mag"],
        bins=np.arange(5.5 - (mag_bin / 2), 12 + (mag_bin / 2), mag_bin),
        histtype="step",
        color="black",
        density=True,
    )
    plt.hist(
        acqs["mag"],
        bins=np.arange(5.5 - (mag_bin / 2), 12 + (mag_bin / 2), mag_bin),
        histtype="step",
        color="red",
        density=True,
    )
    plt.xlabel("Star magnitude (mag)")
    plt.ylabel("Fraction of All Acq Stars")
    plt.title("Expected Magnitudes of Acquisition Stars")
    plt.ylim(0, 1.0)
    plt.subplots_adjust(top=0.85, bottom=0.17, right=0.97)
    if outdir:
        plt.savefig(outdir / "exp_mag_histogram.png")

    plt.figure(figsize=figsize)
    ok = range_acqs["acqid"] == 1
    plt.plot(
        range_acqs[ok]["mag"],
        range_acqs[ok]["mag_obs"] - range_acqs[ok]["mag"],
        "k.",
        markersize=3,
    )
    plt.xlabel("AGASC magnitude (mag)")
    plt.ylabel("Observed - AGASC mag")
    plt.title("Delta Mag vs Mag")
    plt.grid(True)
    plt.subplots_adjust(top=0.85, bottom=0.17, right=0.97)
    if outdir:
        plt.savefig(outdir / "delta_mag_scatter.png")
    if close_figures:
        plt.close(h)

    long_acqs = Table(acqs[acqs["tstart"] > (CxoTime() - 2 * 365 * u.day).secs])[
        ["obsid", "acqid", "tstart"]
    ]
    acqs_id = long_acqs[long_acqs["acqid"] == 1]
    gacqs = acqs_id.group_by("obsid")
    n_acqs = gacqs.groups.aggregate(np.size)
    # Delete bogus column from table meant to count id stars
    del n_acqs["tstart"]
    # Delete acqid column from source table because it doesn't
    # aggregate (for t(ime)_acqs) and isn't necessary now that we
    # have the counts in n_acqs
    del gacqs["acqid"]
    t_acqs = gacqs.groups.aggregate(np.mean)
    out = join(n_acqs, t_acqs, keys="obsid")
    h = plt.figure(figsize=(10, 2.5))
    ska_matplotlib.plot_cxctime(out["tstart"], out["acqid"], ".")
    plt.grid()
    plt.ylim(0, 9)
    plt.margins(0.05)
    plt.ylabel("Identified acq stars")
    plt.tight_layout()
    if outdir:
        plt.savefig(outdir / "id_acq_stars.png")
    if close_figures:
        plt.close(h)


def make_html(nav_dict, rep_dict, fail_dict, outdir):
    """Render and write the basic page.

    Render and write the basic page, where nav_dict is a dictionary of the
    navigation elements (locations of UP_TO_MAIN, NEXT, PREV), rep_dict is
    a dictionary of the main data elements (n failures etc), fail_dict
    contains the elements required for the extra table of failures at the
    bottom of the page, and outdir is the destination directory.
    """
    template = JINJA_ENV.get_template("index.html")
    page = template.render(nav=nav_dict, fails=fail_dict, rep=rep_dict)
    with open(outdir / "index.html", "w") as fh:
        fh.write(page)


def acq_info(acqs, tname, range_datestart, range_datestop):
    """
    Generate a report dictionary for the time range.

    :param acqs: recarray of all acquisition stars available in the table
    :param tname: timerange string (e.g. 2010-M05)
    :param range_datestart: cxotime.CxoTime of start of reporting interval
    :param range_datestop: cxotime.CxoTime of end of reporting interval

    :rtype: dict of report values
    """
    pred = json.load(open(SKA / "data" / "acq_stat_reports" / "acq_fail_fitfile.json"))

    rep = {
        "datestring": tname,
        "datestart": range_datestart.date,
        "datestop": range_datestop.date,
        "human_date_start": range_datestart.datetime.strftime("%Y-%b-%d"),
        "human_date_stop": range_datestop.datetime.strftime("%Y-%b-%d"),
    }

    range_acqs = acqs[
        (acqs["tstart"] >= range_datestart.cxcsec)
        & (acqs["tstart"] < range_datestop.cxcsec)
    ]
    # pred_acqs =  acqs[ (acqs["tstart"] >= CxoTime(pred_start).cxcsec)
    # 	       & (acqs["tstart"] < CxoTime(mxdatestop).cxcsec) ]

    rep["n_stars"] = len(range_acqs)
    if not len(range_acqs):
        raise NoStarError("No acq stars in range")
    rep["n_failed"] = len(np.flatnonzero(range_acqs["acqid"] == 0))
    rep["fail_rate"] = 1.0 * rep["n_failed"] / rep["n_stars"]

    mean_time = (
        CxoTime(rep["datestart"]).cxcsec + CxoTime(rep["datestop"]).cxcsec
    ) / 2.0
    mean_time_d_year = (mean_time - pred["time0"]) / (86400 * 365.25)
    rep["fail_rate_pred"] = mean_time_d_year * pred["m"] + pred["b"]
    rep["n_failed_pred"] = int(round(rep["fail_rate_pred"] * rep["n_stars"]))

    rep["prob_less"] = scipy.stats.poisson.cdf(rep["n_failed"], rep["n_failed_pred"])
    rep["prob_more"] = 1 - scipy.stats.poisson.cdf(
        rep["n_failed"] - 1, rep["n_failed_pred"]
    )

    r, low, high = binomial_confidence_interval(rep["n_failed"], rep["n_stars"])
    rep["fail_rate_err_high"], rep["fail_rate_err_low"] = high - r, r - low

    return rep


def make_fail_html(stars, outfile):
    """
    Render and write the expanded table of failed stars
    """
    template = JINJA_ENV.get_template("stars.html")
    page = template.render(failed_stars=stars)
    f = open(outfile, "w")
    f.write(page)
    f.close()


def acq_fails(acqs, range_datestart, range_datestop, outdir="out"):  #  noqa: ARG001  (commented out)
    """Find the failures over the interval and find the tail mag failures.

    Pass the failures to make_fail_html for the main star table and the
    little ones (by mag bin).
    """

    fails = []
    range_acqs = acqs[
        (acqs["tstart"] >= range_datestart.cxcsec)
        & (acqs["tstart"] < range_datestop.cxcsec)
    ]

    all_fails = range_acqs[range_acqs["acqid"] == 0]
    failed_stars = []
    for fail in all_fails:
        star = {
            "id": fail["agasc_id"],
            "obsid": fail["obsid"],
            "mag_exp": fail["mag"],
            "mag_obs": fail["mag_obs"],
            "acq_stat": "NOID",
        }
        failed_stars.append(star)
    make_fail_html(failed_stars, outdir / "failed_acq_stars_list.html")

    bin = 0.1
    for tmag_start in np.arange(10.0, 10.8, 0.1):
        mag_range_acqs = range_acqs[
            (range_acqs["mag"] >= tmag_start) & (range_acqs["mag"] < (tmag_start + bin))
        ]
        mfailed_stars = []

        mag_fail_acqs = mag_range_acqs[mag_range_acqs["acqid"] == 0]
        for mfail in mag_fail_acqs:
            star = {
                "id": mfail["agasc_id"],
                "obsid": mfail["obsid"],
                "mag_exp": mfail["mag"],
                "mag_obs": mfail["mag_obs"],
                "acq_stat": "NOID",
            }
            mfailed_stars.append(star)
        failed_star_file = outdir / f"failed_acq_{tmag_start:.1f}_stars_list.html"
        make_fail_html(mfailed_stars, failed_star_file)

        mag_fail = {
            "n_stars": len(mag_range_acqs),
            "n_failed": len(np.flatnonzero(mag_range_acqs["acqid"] == 0)),
        }
        if mag_fail["n_stars"] == 0:
            mag_fail["fail_rate"] = 0
        else:
            mag_fail["fail_rate"] = mag_fail["n_failed"] * 1.0 / mag_fail["n_stars"]
        mag_fail["label"] = "%0.1f-%0.1f" % (tmag_start, tmag_start + bin)
        mag_fail["fail_file"] = failed_star_file.name
        fails.append(mag_fail)

    return fails


def get_data():
    mica_acq_stats_file = (
        Path(os.environ["SKA"]) / "data" / "acq_stats" / "acq_stats.h5"
    )
    mica_acq_stats = Table.read(mica_acq_stats_file, path="/data")
    mica_acq_stats["tstart"] = CxoTime(mica_acq_stats["acq_start"]).cxcsec

    all_acq = mica_acq_stats[
        [
            "agasc_id",
            "obsid",
            "tstart",
            "mag_aca",
            "mag_obs",
            "color1",
            "halfw",
            "ccd_temp",
            "acqid",
        ]
    ].copy()
    all_acq.rename_columns(
        ["mag_aca", "color1", "halfw", "ccd_temp"],
        ["mag", "color", "halfwidth", "t_ccd"],
    )

    return all_acq


def main():
    args = get_parser().parse_args()

    logger.setLevel(args.v.upper())

    now = CxoTime()
    logger.info("---------- acq stat reports update at %s ----------" % (now.iso))

    if args.start_time is None:
        to_update = ska_report_ranges.get_update_ranges(args.days_back)
    else:
        now = CxoTime()
        start = CxoTime(args.start_time)
        delta = now - start
        to_update = ska_report_ranges.get_update_ranges(int(delta))

    all_acq = get_data()

    for tname in sorted(to_update.keys()):
        logger.debug("Attempting to update %s" % tname)
        range_datestart = CxoTime(to_update[tname]["start"])
        range_datestop = CxoTime(to_update[tname]["stop"])

        # ignore acquisition stars that are newer than the end of the range
        # in question (happens during reprocessing) for consistency
        all_acq_upto = all_acq[all_acq["tstart"] <= CxoTime(range_datestop).secs]

        webout = args.webdir / f"{to_update[tname]['year']}" / to_update[tname]["subid"]
        webout.mkdir(parents=True, exist_ok=True)

        dataout = (
            args.datadir / f"{to_update[tname]['year']}" / to_update[tname]["subid"]
        )
        dataout.mkdir(parents=True, exist_ok=True)

        logger.debug("Plots and HTML to %s" % webout)
        logger.debug("JSON to  %s" % dataout)

        rep = acq_info(all_acq_upto, tname, range_datestart, range_datestop)
        with open(dataout / "rep.json", "w") as rep_file:
            rep_file.write(json.dumps(rep, sort_keys=True, indent=4))

        fails = acq_fails(all_acq_upto, range_datestart, range_datestop, outdir=webout)
        with open(dataout / "fail.json", "w") as fail_file:
            fail_file.write(json.dumps(fails, sort_keys=True, indent=4))

        prev_range = ska_report_ranges.get_prev(to_update[tname])
        next_range = ska_report_ranges.get_next(to_update[tname])
        nav = {
            "main": args.url,
            "next": f"{args.url}/{next_range['year']}/{next_range['subid']}/index.html",
            "prev": f"{args.url}/{prev_range['year']}/{prev_range['subid']}/index.html",
        }

        make_acq_plots(
            all_acq_upto,
            tstart=range_datestart.secs,
            tstop=range_datestop.secs,
            outdir=webout,
            close_figures=True,
        )
        make_html(nav, rep, fails, outdir=webout)


if __name__ == "__main__":
    main()
