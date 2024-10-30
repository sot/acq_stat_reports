"""
Generate acquisition statistics report.
"""

import argparse
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

from acq_stat_reports import utils
from acq_stat_reports.config import conf

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


def make_acq_plots(acqs, tstart=0, tstop=None, outdir=None):
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
    two_year_acqs = acqs[
        (acqs["tstart"] >= (CxoTime(tstop) - 2 * 365 * u.day).secs)
        & (acqs["tstart"] < tstop)
    ]

    t_ccd_bins = np.linspace(-14, 6, 21)
    if tstart < CxoTime("2020:001").cxcsec:
        # hack to make sure historicall data fits in the range
        mt = np.mean(range_acqs["t_ccd"])
        mt = 2 * np.round(mt / 2)
        t_ccd_bins = np.linspace(mt - 6, mt + 6, 21)

    mag_bins = np.concatenate(
        [np.linspace(5.3, 8.3, 16), [8.5, 8.7, 8.9, 9.2, 9.5, 9.8, 10.2, 11, 12]]
    )

    datasets = {
        "all_acq": range_acqs,
        "two_year_acqs": two_year_acqs,
        "binned_p_acq": utils.BinnedData(
            data=range_acqs,
            bins={"p_acq_model": np.linspace(0, 1, 10)},
        ),
        "binned_t_ccd": utils.BinnedData(
            data=range_acqs,
            bins={"t_ccd": t_ccd_bins},
        ),
        "variable_binned_mag": utils.BinnedData(
            data=range_acqs,
            bins={"mag": mag_bins},
        ),
        "binned_mag": utils.BinnedData(
            data=range_acqs,
            bins={"mag": np.arange(5.5 - (0.2 / 2), 12 + (0.2 / 2), 0.2)},
        ),
        "binned_time": utils.BinnedData(
            data=two_year_acqs,
            bins={"tstart": np.arange(tstop, tstop - 2 * 365 * 86400, -86400 * 30)},
        ),
    }

    plot_params = [
        {
            "description": "Timeline of the number of identified acq stars in the last two years",
            "data": "two_year_acqs",
            "function": acq_stars_plot,
            "parameters": {"filename": "id_acq_stars.png", "figscale": (2, 1)},
        },
        {
            "description": "Timeline of the acquisition failure rate in the last two years",
            "data": "binned_time",
            "function": fail_rate_plot,
            "parameters": {"filename": "fail_rate_plot.png", "figscale": (2, 1)},
        },
        {
            "description": "Scatter plot of observed magnitude Vs catalog magnitude",
            "data": "all_acq",
            "function": mag_scatter_plot,
            "parameters": {
                "filename": "delta_mag_scatter.png",
            },
        },
        {
            "description": "Histogram of magnitudes of acquisition failures",
            "data": "binned_mag",
            "function": utils.binned_data_plot,
            "parameters": {
                "xlabel": "Star magnitude (mag)",
                "ylabel": "N stars",
                "title": "Failed Acquisitions vs Mag",
                "draw_good": False,
                "draw_bad": True,
                "density": False,
                "draw_ranges": True,
                "filename": "mag_histogram.png",
            },
        },
        {
            "description": "Histogram of T_CCD of acquisition failures",
            "data": "binned_t_ccd",
            "function": utils.binned_data_plot,
            "parameters": {
                "xlabel": "T$_{CCD}$",
                "ylabel": "N stars",
                "title": "Failed Acquisition vs T$_{CCD}$",
                "draw_good": False,
                "draw_bad": True,
                "density": False,
                "draw_ranges": True,
                "filename": "t_ccd_histogram.png",
            },
        },
        {
            "description": "",
            "data": "variable_binned_mag",
            "function": utils.binned_data_fraction_plot,
            "parameters": {
                "xlabel": "Star magnitude (mag)",
                "ylabel": "Fraction Acquired",
                "title": "Acquisition Success vs Mag",
                "filename": "mag_pointhist.png",
            },
        },
        {
            "description": "Fraction of acquisition successes vs T_CCD",
            "data": "binned_t_ccd",
            "function": utils.binned_data_fraction_plot,
            "parameters": {
                "xlabel": "T$_{CCD}$",
                "ylabel": "Fraction Acquired",
                "title": "Acquisition Success vs T$_{CCD}$",
                "filename": "t_ccd_pointhist.png",
            },
        },
        {
            "description": "Fraction of acquisition successes vs Magnitude",
            "data": "binned_p_acq",
            "function": utils.binned_data_probability_plot,
            "parameters": {
                "filename": "prob_scatter.png",
            },
        },
    ]

    for params in plot_params:
        params["function"](datasets[params["data"]], **params["parameters"])


@utils.mpl_plot(
    xlabel="AGASC magnitude (mag)",
    ylabel="Observed - AGASC mag",
    title="Delta Mag vs Mag",
)
def mag_scatter_plot(data, **kwargs):  # noqa: ARG001 (kwargs is neeeded by the decorator)
    ok = data["acqid"] == 1
    plt.plot(
        data[ok]["mag"],
        data[ok]["mag_obs"] - data[ok]["mag"],
        "k.",
        markersize=3,
    )
    plt.grid(True)


@utils.mpl_plot(
    ylabel="Failed Acq. Rate (%)",
    figscale=(2, 1),
)
def fail_rate_plot(data, **kwargs):  # noqa: ARG001 (kwargs is neeeded by the decorator)
    d = data.binned_data[np.isfinite(data.binned_data["tstart"])]
    sel = d["n"] != 0  # rate will only be plotted where n != 0
    x = utils._mpl_hist_steps(
        ska_matplotlib.cxctime2plotdate(d["tstart_low"]),
        ska_matplotlib.cxctime2plotdate(d["tstart_high"]),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero")
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        y = np.where(sel, 100 * (d["n"] - d["acqid"]) / d["n"], np.nan)
        y_high = np.where(sel, 100 * (d["n"] - d["sigma_1_high"]) / d["n"], np.nan)
        y_low = np.where(sel, 100 * (d["n"] - d["sigma_1_low"]) / d["n"], np.nan)
        y_2_high = np.where(sel, 100 * (d["n"] - d["sigma_2_high"]) / d["n"], np.nan)
        y_2_low = np.where(sel, 100 * (d["n"] - d["sigma_2_low"]) / d["n"], np.nan)

    sigma_band_2 = plt.fill_between(
        x,
        utils._mpl_hist_steps(y_2_high),
        utils._mpl_hist_steps(y_2_low),
        alpha=0.25,
        color="gray",
    )
    sigma_band_1 = plt.fill_between(
        x,
        utils._mpl_hist_steps(y_high),
        utils._mpl_hist_steps(y_low),
        alpha=0.5,
        color="gray",
    )
    ska_matplotlib.plot_cxctime(d["tstart"], y, ".")

    plt.legend(
        [sigma_band_1, sigma_band_2],
        ["68.2% range", "95.4% range"],
        loc="best",
    )


@utils.mpl_plot(
    ylabel="Identified acq stars",
)
def acq_stars_plot(data, **kwargs):  # noqa: ARG001 (kwargs is neeeded by the decorator)
    acqs_id = data[data["acqid"] == 1]
    gacqs = acqs_id.group_by("obsid")
    n_acqs = gacqs.groups.aggregate(np.size)[["obsid", "acqid"]]
    t_acqs = gacqs.groups.aggregate(np.mean)[["obsid", "tstart"]]
    out = join(n_acqs, t_acqs, keys="obsid")
    ska_matplotlib.plot_cxctime(out["tstart"], out["acqid"], ".")
    plt.yticks([1, 2, 3, 4, 5, 6, 7, 8], minor=True)
    plt.grid(which="minor", axis="y", color="gray")
    plt.grid(which="major", axis="x", color="gray")
    plt.ylim(0, 9)


def make_html(nav_dict, acq_info, outdir):
    """Render and write the basic page.

    Render and write the basic page, where nav_dict is a dictionary of the
    navigation elements (locations of UP_TO_MAIN, NEXT, PREV), rep_dict is
    a dictionary of the main data elements (n failures etc), fail_dict
    contains the elements required for the extra table of failures at the
    bottom of the page, and outdir is the destination directory.
    """
    fail_list = [fail for fail in acq_info["fails"].values() if fail["label"] != ""]

    for label in acq_info["fails"]:
        if label:
            outfile = outdir / f"failed_acq_{label}_stars_list.html"
        else:
            outfile = outdir / "failed_acq_stars_list.html"
        acq_info["fails"][label]["fail_file"] = outfile

        template = JINJA_ENV.get_template("stars.html")
        page = template.render(failed_stars=acq_info["fails"][label]["failed_stars"])
        with open(outfile, "w") as fh:
            fh.write(page)

    template = JINJA_ENV.get_template("index.html")
    page = template.render(nav=nav_dict, fails=fail_list, rep=acq_info["info"])
    with open(outdir / "index.html", "w") as fh:
        fh.write(page)


def get_acq_info(acqs, tname, range_datestart, range_datestop):
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

    result = {
        "info": rep,
        "fails": acq_fails(acqs, range_datestart, range_datestop),
    }
    return result


def acq_fails(acqs, range_datestart, range_datestop, outdir="out"):  #  noqa: ARG001  (commented out)
    """Find the failures over the interval and find the tail mag failures.

    Pass the failures to make_fail_html for the main star table and the
    little ones (by mag bin).
    """

    fails = {}
    range_acqs = acqs[
        (acqs["tstart"] >= range_datestart.cxcsec)
        & (acqs["tstart"] < range_datestop.cxcsec)
    ]

    failed_stars = [
        {
            "id": int(mfail["agasc_id"]),
            "obsid": int(mfail["obsid"]),
            "mag_exp": mfail["mag"],
            "mag_obs": mfail["mag_obs"],
            "acq_stat": "NOID",
        }
        for mfail in range_acqs[range_acqs["acqid"] == 0]
    ]
    n_stars = len(range_acqs)
    n_failed = len(np.flatnonzero(range_acqs["acqid"] == 0))
    fails[""] = {
        # "mag_range": (-np.inf, np.inf),
        "n_stars": n_stars,
        "n_failed": n_failed,
        "fail_rate": 0 if n_stars == 0 else n_failed / n_stars,
        "label": "",
        "failed_stars": failed_stars,
    }

    bin = 0.1
    for tmag_start in np.arange(10.0, 10.8, 0.1):
        label = f"{tmag_start:.1f}-{tmag_start + bin:.1f}"
        mag_range_acqs = range_acqs[
            (range_acqs["mag"] >= tmag_start) & (range_acqs["mag"] < (tmag_start + bin))
        ]

        mfailed_stars = [
            {
                "id": int(mfail["agasc_id"]),
                "obsid": int(mfail["obsid"]),
                "mag_exp": mfail["mag"],
                "mag_obs": mfail["mag_obs"],
                "acq_stat": "NOID",
            }
            for mfail in mag_range_acqs[mag_range_acqs["acqid"] == 0]
        ]

        n_stars = len(mag_range_acqs)
        n_failed = len(np.flatnonzero(mag_range_acqs["acqid"] == 0))
        fails[label] = {
            "mag_range": (tmag_start, tmag_start + bin),
            "n_stars": n_stars,
            "n_failed": n_failed,
            "fail_rate": 0 if n_stars == 0 else n_failed / n_stars,
            "label": label,
            "failed_stars": mfailed_stars,
        }

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
    if conf.remove_bad_stars:
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
    import matplotlib

    matplotlib.use("agg")

    args = get_parser().parse_args()

    logger.setLevel(args.v.upper())

    now = CxoTime()
    logger.info("---------- acq stat reports update at %s ----------" % (now.iso))

    if args.start_time is None:
        to_update = ska_report_ranges.get_update_ranges(args.days_back)
    else:
        start = CxoTime(args.start_time)
        to_update = ska_report_ranges.get_update_ranges(
            int((now - start).to(u.day).value)
        )

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

        acq_info = get_acq_info(all_acq_upto, tname, range_datestart, range_datestop)
        with open(dataout / "acq_info.json", "w") as rep_file:
            rep_file.write(json.dumps(acq_info, sort_keys=True, indent=4))

        prev_range = ska_report_ranges.get_prev(to_update[tname])
        next_range = ska_report_ranges.get_next(to_update[tname])
        nav = {
            "main": args.url,
            "next": f"{args.url}/{next_range['year']}/{next_range['subid']}/index.html",
            "prev": f"{args.url}/{prev_range['year']}/{prev_range['subid']}/index.html",
        }

        conf.data_dir = str(webout)
        conf.close_figures = True
        make_acq_plots(
            all_acq_upto,
            tstart=range_datestart.secs,
            tstop=range_datestop.secs,
        )
        make_html(nav, acq_info, outdir=webout)


if __name__ == "__main__":
    main()
