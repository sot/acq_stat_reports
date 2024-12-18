"""
Generate acquisition statistics report.
"""

import os
import warnings
from pathlib import Path

import agasc
import jinja2
import matplotlib.pyplot as plt
import numpy as np
import ska_matplotlib
from astropy import units as u
from astropy.table import Table, join
from chandra_aca.star_probs import acq_success_prob
from cxotime import CxoTime
from ska_helpers import logging

from acq_stat_reports import utils

__all__ = [
    "NoStarError",
    "logger",
    "get_data",
    "get_binned_data",
    "get_plot_params",
    "mag_scatter_plot",
    "fail_rate_plot",
    "acq_stars_plot",
    "make_html",
    "make_acq_plots",
]


JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates" / "acq_stats")
)
logger = logging.basic_logger("acq_stat_reports", level="INFO")


class NoStarError(Exception):
    """
    Special error for the case when no acquisition stars are found.
    """


def get_binned_data(acqs, tstart=0, tstop=None):
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
    :rtype: dict

    """

    if tstop is None:
        tstop = CxoTime().cxcsec

    tstart = CxoTime(tstart).cxcsec
    tstop = CxoTime(tstop).cxcsec

    range_acqs = acqs[(acqs["tstart"] >= tstart) & (acqs["tstart"] < tstop)]
    two_year_acqs = acqs[
        (acqs["tstart"] >= (CxoTime(tstop) - 2 * 365 * u.day).secs)
        & (acqs["tstart"] < tstop)
    ]
    two_year_acqs_borderline = acqs[
        (acqs["tstart"] >= (CxoTime(tstop) - 2 * 365 * u.day).secs)
        & (acqs["tstart"] < tstop)
        & (acqs["p_acq_model"] > 0.25)
        & (acqs["p_acq_model"] < 0.75)
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
        "binned_time_borderline": utils.BinnedData(
            data=two_year_acqs_borderline,
            bins={"tstart": np.arange(tstop, tstop - 2 * 365 * 86400, -86400 * 30)},
        ),
    }
    return datasets


def get_plot_params():
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
            "parameters": {
                "filename": "fail_rate_plot.png",
                "figscale": (2, 1),
                "title": "Failure Rate History",
            },
        },
        {
            "description": "Timeline of the acquisition failure rate in the last two years",
            "data": "binned_time_borderline",
            "function": fail_rate_plot,
            "parameters": {
                "filename": "fail_rate_plot_borderline.png",
                "figscale": (2, 1),
                "title": "Failure Rate History ($0.25 < p_{acq} < 0.75$)",
            },
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

    return plot_params


def make_acq_plots(datasets):
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

    plot_params = get_plot_params()

    for params in plot_params:
        params["function"](datasets[params["data"]], **params["parameters"])

    for pp in plot_params:
        pp["function"] = f"{pp['function'].__module__}.{pp['function'].__name__}"

    result = {
        "datasets": {
            k: v.json() for k, v in datasets.items() if isinstance(v, utils.BinnedData)
        },
        "plot_params": plot_params,
    }

    return result


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

    ymax = np.max([y, y_high, y_low, y_2_high, y_2_low])
    plt.ylim(0, 1.1 * ymax)

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


def make_html(data, outdir):
    """Render and write the basic page.

    Render and write the basic page, where nav_dict is a dictionary of the
    navigation elements (locations of UP_TO_MAIN, NEXT, PREV), rep_dict is
    a dictionary of the main data elements (n failures etc), fail_dict
    contains the elements required for the extra table of failures at the
    bottom of the page, and outdir is the destination directory.
    """
    template = JINJA_ENV.get_template("index.html")
    page = template.render(time_ranges=data["time_ranges"])

    with open(outdir / "index.html", "w") as fh:
        fh.write(page)


def get_data(remove_bad_stars=False):
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
    if remove_bad_stars:
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
