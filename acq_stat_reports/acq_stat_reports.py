"""
Generate acquisition statistics report.
"""

import argparse
import itertools
import json
import os
import warnings
from pathlib import Path

import agasc
import jinja2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import ska_matplotlib
import ska_report_ranges
from astropy import units as u
from astropy.table import Table, join
from chandra_aca.star_probs import acq_success_prob, binomial_confidence_interval
from cxotime import CxoTime
from ska_helpers import logging

from acq_stat_reports.config import OPTIONS

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
        help="Output directory (default is $SKA/data/acq_stat_reports)",
        type=Path,
    )
    parser.add_argument(
        "--webdir",
        default=Path(SKA / "www" / "ASPECT" / "acq_stat_reports"),
        help="Output directory (default is $SKA/www/ASPECT/acq_stat_reports)",
        type=Path,
    )
    parser.add_argument(
        "--url",
        help="URL root of the deployed page (default is /mta/ASPECT/acq_stat_reports/)",
        default="/mta/ASPECT/acq_stat_reports/",
    )
    parser.add_argument(
        "--start_time", help="Start time (default is NOW)", default=None
    )
    parser.add_argument(
        "--days_back",
        help="How many days to look back (default is 30)",
        default=30,
        type=int,
    )
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


def make_acq_plots(acqs, tstart=0, tstop=None, outdir=None, close_figures=False):
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

    range_acqs = acqs[(acqs["tstart"] >= tstart) & (acqs["tstart"] < tstop)]

    make_mag_histogram_plot(range_acqs)
    make_mag_distribution_plot(range_acqs)
    make_mag_pointhist_plot(acqs, range_acqs)
    make_delta_mag_scatter_plot(range_acqs)
    make_id_acq_stars_plot(range_acqs, outdir=outdir, close_figures=close_figures)


def make_mag_distribution_plot(range_acqs):
    figsize = (OPTIONS.figure_width, OPTIONS.figure_height)
    outdir = Path(OPTIONS.data_dir)
    close_figures = OPTIONS.close_figures
    mag_bin = 0.1

    h = plt.figure(figsize=figsize)
    bins = np.arange(5.5 - (mag_bin / 2), 12 + (mag_bin / 2), mag_bin)

    plt.hist(
        range_acqs["mag"][range_acqs["acqid"] == 1],
        bins=bins,
        histtype="step",
        color="k",
        density=True,
        label="Acquired",
    )
    plt.hist(
        range_acqs["mag"][range_acqs["acqid"] == 0],
        bins=bins,
        histtype="step",
        color="r",
        weights=np.full(np.count_nonzero(range_acqs["acqid"] == 0), 100),
        density=True,
        label="Not acquired",
    )

    plt.title("Star Magnitude Distribution")
    plt.xlabel("Star magnitude (mag)")
    plt.ylabel(r"# stars/$\Delta mag$")
    plt.subplots_adjust(top=0.85, bottom=0.17, right=0.97)
    plt.xlim(5, 12)
    plt.yscale("linear")
    plt.legend(loc="best")
    plt.tight_layout()
    if outdir:
        plt.savefig(outdir / "mag_distribution.png")
    if close_figures:
        plt.close(h)


def make_mag_pointhist_plot(range_acqs):
    figsize = (OPTIONS.figure_width, OPTIONS.figure_height)
    outdir = Path(OPTIONS.data_dir)
    close_figures = OPTIONS.close_figures

    # Acquisition Success Fraction, full mag range
    h = plt.figure(figsize=figsize)
    mag_bin = 0.1
    bins = np.arange(5.5 - (mag_bin / 2), 12 + (mag_bin / 2), mag_bin)

    quantiles = get_histogram_quantile_ranges(range_acqs, {"mag": bins})

    sel = quantiles["n"] > 0
    quantiles["acqid_frac"] = np.zeros(len(quantiles))
    quantiles["low_frac"] = np.zeros(len(quantiles))
    quantiles["high_frac"] = np.zeros(len(quantiles))
    quantiles["acqid_frac"][sel] = quantiles["acqid"][sel] / quantiles["n"][sel]
    quantiles["low_frac"][sel] = quantiles["low"][sel] / quantiles["n"][sel]
    quantiles["high_frac"][sel] = quantiles["high"][sel] / quantiles["n"][sel]

    plt.plot(quantiles["mag"], quantiles["acqid_frac"], ".", color="k")
    plt.fill_between(
        quantiles["mag"],
        quantiles["low_frac"],
        quantiles["high_frac"],
        color="gray",
        alpha=0.8,
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


def make_mag_histogram_plot(
    range_acqs,
    draw_good=True,
    draw_bad=True,
    density=True,
    draw_ranges=True,
    filename="mag_histogram.png",
):
    figsize = (OPTIONS.figure_width, OPTIONS.figure_height)
    outdir = Path(OPTIONS.data_dir)
    close_figures = OPTIONS.close_figures
    mag_bin = 0.1

    # Scaled Failure Histogram, full mag range
    h = plt.figure(figsize=figsize)
    bins = np.arange(5.5 - (mag_bin / 2), 12 + (mag_bin / 2), mag_bin)

    quantiles = get_histogram_quantile_ranges(range_acqs, {"mag": bins})
    x = np.array([quantiles["mag_low"], quantiles["mag_high"]]).T.flatten()
    # y = np.array([quantiles["median"], quantiles["median"]]).T.flatten()
    y1 = np.array([quantiles["low"], quantiles["low"]]).T.flatten()
    y2 = np.array([quantiles["high"], quantiles["high"]]).T.flatten()
    n = np.array([quantiles["n"], quantiles["n"]]).T.flatten()
    acqid = np.array([quantiles["acqid"], quantiles["acqid"]]).T.flatten()

    if draw_good:
        scale = 1 / (mag_bin * np.sum(acqid)) if density else 1
        # plt.plot(x, scale * y, "-", color="gray")
        if draw_ranges:
            plt.fill_between(x, scale * y1, scale * y2, color="gray", alpha=0.8)
        plt.plot(x, scale * acqid, "-", color="k")

    if draw_bad:
        scale = 1 / (mag_bin * np.sum(n - acqid)) if density else 1
        # plt.plot(x, scale * (n-y), "-", color="orange")
        if draw_ranges:
            plt.fill_between(
                x, scale * (n - y1), scale * (n - y2), color="r", alpha=0.3
            )
        plt.plot(x, scale * (n - acqid), "-", color="r")

    plt.xlabel("Star magnitude (mag)")
    plt.ylabel("N stars")
    plt.title("N good (black) and bad (red) stars vs Mag")
    plt.xlim(bins[0], bins[-1])
    plt.ylim(ymin=0)
    plt.tight_layout()
    if outdir:
        plt.savefig(outdir / filename)
    if close_figures:
        plt.close(h)


def make_delta_mag_scatter_plot(range_acqs):
    figsize = (OPTIONS.figure_width, OPTIONS.figure_height)
    outdir = Path(OPTIONS.data_dir)
    close_figures = OPTIONS.close_figures

    h = plt.figure(figsize=figsize)
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


def make_id_acq_stars_plot(range_acqs):
    outdir = Path(OPTIONS.data_dir)
    close_figures = OPTIONS.close_figures

    long_acqs = Table(
        range_acqs[range_acqs["tstart"] > (CxoTime() - 2 * 365 * u.day).secs]
    )[["obsid", "acqid", "tstart"]]
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


def _get_quantiles_(group, n_samples=10000):
    samples = (
        np.random.uniform(size=n_samples * len(group)).reshape((-1, len(group)))
        < group["p_acq_model"][None]
    )
    n = np.sum(samples, axis=1)
    return np.percentile(n, [5, 50, 95])


def get_histogram_quantile_ranges(data, bin_edges, n_samples=10000):
    cols = list(bin_edges)
    bin_cols = [f"{bin_col}_bin" for bin_col in cols]
    for bin_col in cols:
        bin_edges[bin_col] = np.atleast_1d(bin_edges[bin_col])
        if len(bin_edges[bin_col].shape) != 1:
            raise ValueError("bin_edges must be a dict of 1D arrays")

    # data = data.copy()
    bin_edges = bin_edges.copy()

    # for bin_col in cols:
    #     # no underflow (otherwise I one has to be careful with bin == 0)
    #     data = data[data[bin_col] > bin_edges[bin_col][0]]

    sizes = [len(b) for b in bin_edges.values()]
    offsets = {cols[0]: 0}
    offsets.update({cols[i]: np.prod(sizes[:i]) for i in range(1, len(cols))})

    global_bin_idx = np.zeros(len(data), dtype=int)
    for bin_col in cols:
        data[f"{bin_col}_bin"] = np.digitize(data[bin_col], bin_edges[bin_col])
        global_bin_idx += offsets[bin_col] + data[f"{bin_col}_bin"]
    data["bin"] = global_bin_idx

    # now this is very inefficient, because we are making a list of all possible combinations of bin
    # indices. Note that we are iterating over cols in reverse order so that the bin calculation
    # agrees with the one on the data array.
    bin_idx = dict(
        zip(
            cols[::-1],
            np.array(
                list(
                    zip(
                        *[
                            list(j)
                            for j in itertools.product(
                                *[range(len(bin_edges[col]) + 1) for col in cols[::-1]]
                            )
                        ],
                        strict=True,
                    )
                )
            ),
            strict=True,
        )
    )

    # bins are the bin edges, and np.digitize returns indices assuming there are (n_bins - 1) bins,
    # plus the under/overflow bins. That's (n_bins + 1) bins.
    # for book-keeping, we will create padded arrays with -inf and +inf
    padded_bins = {
        col: np.concatenate([[-np.inf], bin_edges[col], [np.inf]]) for col in cols
    }

    quantiles = Table()
    quantiles["bin"] = np.arange(len(bin_idx[cols[0]]))
    for bin_col in cols:
        j = bin_idx[bin_col]
        quantiles[f"{bin_col}_bin"] = j
        quantiles[f"{bin_col}_low"] = padded_bins[bin_col][j]
        quantiles[f"{bin_col}_high"] = padded_bins[bin_col][j + 1]
        quantiles[f"{bin_col}"] = (
            quantiles[f"{bin_col}_low"] + quantiles[f"{bin_col}_high"]
        ) / 2

    # this is a sanity check that the bin indices as calculated above are correct
    global_bin_idx = np.zeros(len(quantiles), dtype=int)
    for bin_col in cols:
        global_bin_idx += offsets[bin_col] + quantiles[f"{bin_col}_bin"]
    assert np.all(quantiles["bin"] == global_bin_idx)
    # end sanity check

    g = data.group_by(bin_cols)
    assert len(g.groups.indices) < len(quantiles)  # sanity check
    idx = g[["bin"] + cols].groups.aggregate(np.mean)["bin"].astype(int)
    mean = g[cols].groups.aggregate(np.mean)
    n = np.diff(g.groups.indices)
    acqid = g["acqid"].groups.aggregate(np.count_nonzero)
    low, median, high = np.vstack(
        [_get_quantiles_(group, n_samples) for group in g.groups]
    ).T

    quantiles["low"] = 0
    quantiles["median"] = 0
    quantiles["high"] = 0
    quantiles["n"] = 0
    quantiles["acqid"] = 0

    quantiles["low"][idx] = low
    quantiles["median"][idx] = median
    quantiles["high"][idx] = high
    quantiles["n"][idx] = n
    quantiles["acqid"][idx] = acqid

    for col in cols:
        quantiles[f"{col}_mean"] = np.nan
        quantiles[f"{col}_mean"][idx] = mean[col]
        quantiles[f"{col}_delta"] = quantiles[f"{col}_high"] - quantiles[f"{col}_low"]
    return quantiles


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
            "known_bad",
        ]
    ].copy()
    all_acq.rename_columns(
        ["mag_aca", "color1", "halfw", "ccd_temp"],
        ["mag", "color", "halfwidth", "t_ccd"],
    )

    # Coerce uint8 columns (which are all actually bool) to bool
    for col in all_acq.itercols():
        if col.dtype.type is np.uint8:
            col.dtype = bool

    # Get latest mag estimates from the AGASC with supplement
    mags_supp = agasc.get_supplement_table("mags")
    mags_supp = dict(zip(mags_supp["agasc_id"], mags_supp["mag_aca"], strict=True))
    all_acq["mag"] = [
        mags_supp.get(agasc_id, mag_aca)
        for agasc_id, mag_aca in zip(all_acq["agasc_id"], all_acq["mag"], strict=True)
    ]

    # Remove known bad stars
    bad_stars = agasc.get_supplement_table("bad")
    bad = np.isin(all_acq["agasc_id"], bad_stars["agasc_id"]) | all_acq["known_bad"]
    if OPTIONS.remove_bad_stars:
        all_acq = all_acq[~bad]
    else:
        all_acq["bad_star"] = bad

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="\n.*clipping input.*outside that range"
        )
        all_acq["p_acq_model"] = acq_success_prob(
            date=all_acq["tstart"],
            t_ccd=all_acq["t_ccd"],
            mag=all_acq["mag"],
            color=all_acq["color"],
            spoiler=False,
            halfwidth=all_acq["halfwidth"],
        )

        all_acq["p_acq_model"].format = ".3f"
        all_acq["t_ccd"].format = ".2f"
        all_acq["mag"].format = ".2f"
        all_acq["mag_obs"].format = ".2f"
        all_acq["color"].format = ".2f"

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

        OPTIONS.data_dir = webout
        OPTIONS.close_figures = True
        make_acq_plots(
            all_acq_upto,
            tstart=range_datestart.secs,
            tstop=range_datestop.secs,
        )
        make_html(nav, rep, fails, outdir=webout)


if __name__ == "__main__":
    main()
