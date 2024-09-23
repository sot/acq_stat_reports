import functools
import itertools
from pathlib import Path
from typing import Any

# from typing import Callable
import matplotlib.pyplot as plt
import numpy as np

# import numpy.typing as npt
from astropy.table import Table
from chandra_aca.star_probs import binomial_confidence_interval

from acq_stat_reports.config import conf


class BinnedData:
    def __init__(self, data, bins, extra_cols=None, selection=None):
        self.bins = bins
        self.selection = selection if selection is not None else (lambda x: x)
        self.data = self.selection(data)
        self.binned_data = get_histogram_quantile_ranges(
            self.selection(data),
            bin_edges=self.bins,
            extra_cols=[] if extra_cols is None else extra_cols,
        )


def _mpl_hist_steps(arr, arr2=None):
    """
    Utility to generate a step plot from histogram values.

    Parameters
    ----------
    arr : array-like
    arr2 : array-like, optional

    Returns
    -------
    array-like
    """
    if arr2 is None:
        return np.array([arr, arr]).T.flatten()
    return np.array([arr, arr2]).T.flatten()


def mpl_plot(**defaults):
    """
    Decorator to handle common matplotlib plotting tasks.
    """

    def wrap(func):
        @functools.wraps(func)
        def wrapped_function(*args, **kwargs):
            options = defaults.copy()
            options.update(kwargs)

            filename = options.pop("filename", None)
            ax = options.pop("ax", None)
            close_figures = options.pop("close_figures", conf.close_figures)
            figsize = options.pop("figsize", (conf.figure_width, conf.figure_height))
            figscale = options.pop("figscale", (1, 1))
            figscale, _ = np.broadcast_arrays(figscale, [1, 1])
            figsize = (figsize[0] * figscale[0], figsize[1] * figscale[1])

            outdir = options.pop("outdir", Path(conf.data_dir))

            with plt.style.context(options.pop("style", conf.mpl_style)):
                if ax is None:
                    fig = plt.figure(figsize=figsize)

                func(*args, **options)
                plt.xlabel(options.get("xlabel", ""))
                plt.ylabel(options.get("ylabel", ""))
                plt.title(options.get("title", ""))

                if options.get("legend", False):
                    plt.legend(loc="best")

                if ax is None:
                    plt.tight_layout()

                if outdir and filename:
                    plt.savefig(outdir / filename)
                if close_figures:
                    plt.close(fig)

        return wrapped_function

    return wrap


@mpl_plot()
def binned_data_plot(binned_data: BinnedData, **kwargs):
    """
    Plot binned data.

    This function plots histograms of the binned data, where the data must have been binned using
    a single column. The data is expected to be a Table with columns:

    - `bin`: the bin index
    - `{col}_low`: the lower edge of the bin
    - `{col}_high`: the upper edge of the bin
    - `{col}`: the center of the bin
    - `{col}_delta`: the width of the bin
    - `n`: the number of samples in the bin
    - `acqid`: the number of "good" samples in the bin
    - `low`: the lower quantile (usually 5%)
    - `high`: the upper quantile (usually 95%)

    Optional arguments:

    - density. Normalize the histograms.
    - draw_good. Draw a histogram of the "good" samples.
    - draw_bad. Draw a histogram of the "bad" samples.
    - draw_ranges. Draw a shaded region between the lower and upper quantiles.

    """
    if len(binned_data.bins) > 1:
        raise Exception("Only one axis allowed in binned_data_plot")
    bins = list(binned_data.bins.values())[0]
    col = list(binned_data.bins.keys())[0]
    density = kwargs.get("density", False)
    draw_good = kwargs.get("draw_good", False)
    draw_bad = kwargs.get("draw_bad", False)
    draw_ranges = kwargs.get("draw_ranges", False)
    draw_legend = kwargs.get("draw_legend", True)

    quantiles = binned_data.binned_data
    x = _mpl_hist_steps(quantiles[f"{col}_low"], quantiles[f"{col}_high"])
    sigma_1_low = _mpl_hist_steps(quantiles["sigma_1_low"])
    sigma_1_high = _mpl_hist_steps(quantiles["sigma_1_high"])
    sigma_2_low = _mpl_hist_steps(quantiles["sigma_2_low"])
    sigma_2_high = _mpl_hist_steps(quantiles["sigma_2_high"])
    n = _mpl_hist_steps(quantiles["n"])
    acqid = _mpl_hist_steps(quantiles["acqid"])
    diff = _mpl_hist_steps(quantiles[f"{col}_delta"])

    labels: dict[Any, str] = {}
    if draw_good and np.any(binned_data.data["acqid"]):
        scale = 1 / (diff * np.sum(quantiles["acqid"])) if density else 1
        if draw_ranges:
            sigma_band_2 = plt.fill_between(
                x,
                scale * sigma_2_low,
                scale * sigma_2_high,
                color="gray",
                alpha=0.3,
            )
            sigma_band_1 = plt.fill_between(
                x,
                scale * sigma_1_low,
                scale * sigma_1_high,
                color="gray",
                alpha=0.8,
            )
            labels.update(
                {
                    sigma_band_1: "68.2% range",
                    sigma_band_2: "95.4% range",
                }
            )
        (scatter,) = plt.plot(
            x,
            scale * acqid,
            "-",
            color="k",
            label="Acquired",
        )
        labels[scatter] = "Acquired"

    if draw_bad and np.any(~binned_data.data["acqid"]):
        scale = 1 / (diff * np.sum(n - acqid)) if density else 1
        if draw_ranges:
            sigma_band_1 = plt.fill_between(
                x,
                scale * (n - sigma_2_low),
                scale * (n - sigma_2_high),
                color="r",
                alpha=0.1,
            )
            sigma_band_2 = plt.fill_between(
                x,
                scale * (n - sigma_1_low),
                scale * (n - sigma_1_high),
                color="r",
                alpha=0.3,
            )
            labels.update(
                {
                    sigma_band_1: "68.2% range",
                    sigma_band_2: "95.4% range",
                }
            )
        (scatter,) = plt.plot(
            x,
            scale * (n - acqid),
            "-",
            color="r",
            label="Not Acquired",
        )
        labels[scatter] = "Not Acquired"

    bins = bins[np.isfinite(bins)]
    plt.xlim(bins[0], bins[-1])
    plt.ylim(ymin=0)

    if draw_legend:
        plt.legend(
            labels.keys(),
            labels.values(),
            loc="best",
        )


@mpl_plot()
def binned_data_fraction_plot(
    binned_data: BinnedData,
    **kwargs,  # noqa: ARG001 (kwargs is neeeded by the decorator)
):
    """
    Plot binned data.

    This function plots ratios from the binned data (acqid/n), where the data must have been binned
    using a single column. The data is expected to be a Table with columns:

    - `bin`: the bin index
    - `{col}_low`: the lower edge of the bin
    - `{col}_high`: the upper edge of the bin
    - `{col}`: the center of the bin
    - `{col}_delta`: the width of the bin
    - `n`: the number of samples in the bin
    - `acqid`: the number of "good" samples in the bin
    - `low`: the lower quantile (usually 5%)
    - `high`: the upper quantile (usually 95%)

    The plot includes a shaded region between the lower and upper quantiles (low/n and high/n).

    """
    if len(binned_data.bins) > 1:
        raise Exception("Only one axis allowed in FractionPlot")
    bins = list(binned_data.bins.values())[0]
    col = list(binned_data.bins.keys())[0]

    quantiles = binned_data.binned_data

    sel = quantiles["n"] > 0
    acqid_frac = np.zeros(len(quantiles))
    sigma_1_low_frac = np.zeros(len(quantiles))
    sigma_2_low_frac = np.zeros(len(quantiles))
    sigma_1_high_frac = np.zeros(len(quantiles))
    sigma_2_high_frac = np.zeros(len(quantiles))
    acqid_frac[sel] = quantiles["acqid"][sel] / quantiles["n"][sel]
    sigma_1_low_frac[sel] = quantiles["sigma_1_low"][sel] / quantiles["n"][sel]
    sigma_2_low_frac[sel] = quantiles["sigma_2_low"][sel] / quantiles["n"][sel]
    sigma_1_high_frac[sel] = quantiles["sigma_1_high"][sel] / quantiles["n"][sel]
    sigma_2_high_frac[sel] = quantiles["sigma_2_high"][sel] / quantiles["n"][sel]
    acqid_frac[~sel] = np.nan
    sigma_1_low_frac[~sel] = np.nan
    sigma_2_low_frac[~sel] = np.nan
    sigma_1_high_frac[~sel] = np.nan
    sigma_2_high_frac[~sel] = np.nan

    (scatter,) = plt.plot(quantiles[col], acqid_frac, ".", color="k")

    x = _mpl_hist_steps(
        quantiles[f"{col}_low"],
        quantiles[f"{col}_high"],
    )
    sigma_1_low = _mpl_hist_steps(sigma_1_low_frac)
    sigma_2_low = _mpl_hist_steps(sigma_2_low_frac)
    sigma_1_high = _mpl_hist_steps(sigma_1_high_frac)
    sigma_2_high = _mpl_hist_steps(sigma_2_high_frac)

    sigma_band_1 = plt.fill_between(
        x, sigma_1_low, sigma_1_high, color="gray", alpha=0.8
    )
    sigma_band_2 = plt.fill_between(
        x, sigma_2_low, sigma_2_high, color="gray", alpha=0.3
    )

    bins = bins[np.isfinite(bins)]
    plt.xlim(bins[0], bins[-1])
    plt.ylim(-0.05, 1.05)

    labels = {
        scatter: "Observed",
        sigma_band_1: "68.2% range",
        sigma_band_2: "95.4% range",
    }
    plt.legend(
        labels.keys(),
        labels.values(),
        loc="best",
    )


@mpl_plot(
    title="Acquisition Probability",
    xlabel="Expected p$_{acq}$",
    ylabel="Observed p$_{acq}$",
    figscale=(1, 2),
)
def binned_data_probability_plot(
    binned_data: BinnedData,
    **kwargs,  # noqa: ARG001 (kwargs is neeeded by the decorator)
):
    quantiles = binned_data.binned_data

    sel = quantiles["n"] > 0
    prob_exp = quantiles["p_acq_model_mean"][sel]
    prob_obs, low, high = binomial_confidence_interval(
        quantiles["acqid"][sel], quantiles["n"][sel]
    )

    plt.gca().set_aspect("equal")
    plt.errorbar(
        prob_exp,
        prob_obs,
        yerr=[prob_obs - low, high - prob_obs],
        xerr=quantiles["p_acq_model_std"][sel],
        fmt=".",
        color="k",
        linewidth=1,
    )
    x = np.linspace(0, 1, 10)
    plt.plot(x, x, "k--")
    plt.ylim(0, 1)
    plt.xlim(0, 1)


def _get_quantiles_(p_acq_model, quantiles=None, n_realizations=10000):
    """
    Generate the 5th, 50th, and 95th percentiles of the expected number of successes in a series.

    This function assumes the input is an array of probabilities, one for each draw.
    It generates `n_realizations` realizations , and uses them to estimate the percentiles.

    Parameters
    ----------
    p_acq_model : array-like
        The binomial probability.
    n_realizations : int, optional
        The number of realizations of the series to use for the Monte Carlo estimation.

    Returns
    -------
    array-like
        The quantiles
    """
    if quantiles is None:
        quantiles = [15.9, 50, 84.1]
    samples = (
        np.random.uniform(size=n_realizations * len(p_acq_model)).reshape(
            (-1, len(p_acq_model))
        )
        < p_acq_model[None]
    )
    n = np.sum(samples, axis=1)
    return np.percentile(n, quantiles)


def get_histogram_quantile_ranges(  # noqa: PLR0915
    data,
    bin_edges,
    extra_cols=(),
    n_samples=10000,
    success_column="acqid",
    prob_column="p_acq_model",
):
    """
    Summarize data in bins and calculate quantiles.

    This function expects a Table with `success_column` and `prob_column` columns. The
    `success_column` column tells whether this sample is a success, and `prob_column` is the
    a-priori probability of success.

    The function bins the data according to the columns given in `bin_edges`, and creates the
    following columns:
    - `bin`: the unique bin index
    - `n`: the number of samples in the bin
    - `{col}_low`: the lower edge of the bin (repeated for all bin_edges keys)
    - `{col}_high`: the upper edge of the bin (repeated for all bin_edges keys)
    - `{col}`: the center of the bin (repeated for all bin_edges keys)
    - `{col}_delta`: the width of the bin (repeated for all bin_edges keys)
    - `{success_column}`: the number of "good" samples in the bin
    - `low`: the lower quantile (usually 5%)
    - `median`: the median
    - `high`: the upper quantile (usually 95%)

    NOTE: the result is NOT sparse (it will contain all bins, even if there are no samples in it).

    Parameters
    ----------
    data : Table
        The data to summarize. Must include a `prob_column` column and the corresponding columns
        for the bin edges.
    bin_edges : dict
        A dictionary of bin edges for each column.
    n_samples : int, optional
        The number of realizations to use in the Monte Carlo estimation of the expected successes.
    success_column: str, optional
        The name of the column containing the success/failure boolean flag.
    prob_column: str, optional
        The name of the column containing the probabilities.

    Returns
    -------
    Table
        The quantiles of the data in the bins.
    """
    cols = list(bin_edges)
    extra_cols = list(extra_cols)
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

    sizes = [len(b) + 1 for b in bin_edges.values()]
    offsets = {cols[0]: 1}
    offsets.update({cols[i]: np.prod(sizes[:i]) for i in range(1, len(cols))})

    global_bin_idx = np.zeros(len(data), dtype=int)
    for bin_col in cols:
        data[f"{bin_col}_bin"] = np.digitize(data[bin_col], bin_edges[bin_col])
        global_bin_idx += offsets[bin_col] * data[f"{bin_col}_bin"]
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
        global_bin_idx += offsets[bin_col] * quantiles[f"{bin_col}_bin"]
    assert np.all(quantiles["bin"] == global_bin_idx)
    # end sanity check

    g = data.group_by(bin_cols)
    assert len(g.groups.indices) <= len(quantiles) + 1  # sanity check
    idx = g[["bin"] + cols].groups.aggregate(np.mean)["bin"].astype(int)
    mean = g[cols + extra_cols].groups.aggregate(np.mean)
    std = g[cols + extra_cols].groups.aggregate(np.std)
    n = np.diff(g.groups.indices)
    acqid = g[success_column].groups.aggregate(np.count_nonzero)
    sigma_2_low, sigma_1_low, median, sigma_1_high, sigma_2_high = np.vstack(
        [
            _get_quantiles_(
                group[prob_column], [[2.27, 15.9, 50, 84.1, 97.7]], n_samples
            )
            for group in g.groups
        ]
    ).T

    quantiles["sigma_1_low"] = 0
    quantiles["sigma_2_low"] = 0
    quantiles["median"] = 0
    quantiles["sigma_1_high"] = 0
    quantiles["sigma_2_high"] = 0
    quantiles["n"] = 0
    quantiles[success_column] = 0

    quantiles["sigma_1_low"][idx] = sigma_1_low
    quantiles["sigma_1_low"][idx] = sigma_1_low
    quantiles["sigma_2_low"][idx] = sigma_2_low
    quantiles["median"][idx] = median
    quantiles["sigma_1_high"][idx] = sigma_1_high
    quantiles["sigma_1_high"][idx] = sigma_1_high
    quantiles["sigma_2_high"][idx] = sigma_2_high
    quantiles["n"][idx] = n
    quantiles[success_column][idx] = acqid

    for col in cols:
        quantiles[f"{col}_mean"] = np.nan
        quantiles[f"{col}_mean"][idx] = mean[col]
        quantiles[f"{col}_delta"] = quantiles[f"{col}_high"] - quantiles[f"{col}_low"]
        quantiles[f"{col}_std"] = np.nan
        quantiles[f"{col}_std"][idx] = std[col]
    for col in extra_cols:
        quantiles[f"{col}_mean"] = np.nan
        quantiles[f"{col}_mean"][idx] = mean[col]
        quantiles[f"{col}_std"] = np.nan
        quantiles[f"{col}_std"][idx] = std[col]

    quantiles.meta["shape"] = tuple(sizes)[::-1]
    return quantiles
