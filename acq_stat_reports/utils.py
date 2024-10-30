import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# from typing import Callable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# import numpy.typing as npt
from astropy.table import Table
from chandra_aca.star_probs import binomial_confidence_interval

from acq_stat_reports.config import conf


@dataclass
class BinnedData:
    """
    Class to store the data and results from get_histogram_quantile_ranges.
    """

    bins: dict[str, np.ndarray]
    data: Table
    binned_data: Table

    def __init__(self, data, bins, extra_cols=None):
        self.bins = bins
        self.data = data
        self.binned_data = get_histogram_quantile_ranges(
            data,
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

    This decorator can take kwargs to serve as defaults for the wrapped function.

    Before calling the wrapped function, the following parameters are popped from kwargs:

        * filename: str. Used to save the figure.
        * ax: matplotlib.axes.Axes. The matplotlib axes to use. If not given, a new figure is
          created.
        * figsize: tuple. Figure size in inches. Default: (conf.figure_width, conf.figure_height).
        * figscale: tuple. Scale factor applied to figsize. Defaults to (1, 1).
        * outdir: Path. The directory where to save the image. Default: conf.data_dir.
        * style: str. The matplotlib style to use. Default: conf.mpl_style.
        * xlabel: str. Label for the x-axis. Default: "".
        * ylabel: str. Label for the y-axis. Default: "".
        * title: str. The title of the plot. Default: "".
        * legend: bool. Whether to draw the legend. Default: False.

    All other kwargs are passed to the wrapped function.

    Prameters
    ---------
    defaults : dict
        Default values for the parameters.
    """

    def wrap(func):
        @functools.wraps(func)
        def wrapped_function(*args, **kwargs):
            options = defaults.copy()
            options.update(kwargs)

            filename = options.pop("filename", None)
            ax = options.pop("ax", None)
            figsize = options.pop("figsize", (conf.figure_width, conf.figure_height))
            figscale = options.pop("figscale", (1, 1))
            figscale, _ = np.broadcast_arrays(figscale, [1, 1])
            figsize = (figsize[0] * figscale[0], figsize[1] * figscale[1])

            outdir = options.pop("outdir", Path(conf.output_dir))

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
                if matplotlib.pyplot.isinteractive():
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
    - `sigma_1_low`: the lower 1-sigma quantile (usually 15.9%)
    - `sigma_1_high`: the upper 1-sigma quantile (usually 84.1%)
    - `sigma_2_low`: the lower 2-sigma quantile (usually 2.27%)
    - `sigma_2_high`: the upper 2-sigma quantile (usually 97.7%)

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
    - `sigma_1_low`: the lower 1-sigma quantile (usually 15.9%)
    - `sigma_1_high`: the upper 1-sigma quantile (usually 84.1%)
    - `sigma_2_low`: the lower 2-sigma quantile (usually 2.27%)
    - `sigma_2_high`: the upper 2-sigma quantile (usually 97.7%)

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


def get_quantiles(p_acq_model, quantiles=None, n_realizations=10000):
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
    p_acq_model = np.atleast_1d(p_acq_model).flatten()

    if quantiles is None:
        quantiles = [15.9, 50, 84.1]
    # The input is an array of N probabilities, one for each draw,
    # and we want to generate n_realizations realizations of the series.
    # That is an array with shape (n_realizations, N)
    samples = (
        np.random.uniform(size=n_realizations * len(p_acq_model)).reshape(
            (-1, len(p_acq_model))
        )
        < p_acq_model[None]
    )
    # we get the number of successes for each realization by summing along the p_acq axis (axis 1)
    n = np.sum(samples, axis=1)
    return np.percentile(n, quantiles)


def _check_arguments(indices, bin_edges):
    """
    Utility to verify that indices and bin edges are valid and consistent.

    This function also does some argument transformations to interpret typical input formats.

    Indices must not be negative, and must not be larger than the number of bins (including under-
    and over-flow bins). For example, if the bin edges are [1, 2, 3], bin indices can not be larger
    than 4.

    Parameters
    ----------
    indices : array-like
        The indices to check. It can be a structured or plain array.
        The array must be 1- or 2-dimensional. If it is 2-dimensional, the dimensionality of the
        data is given by the number of columns.
    bin_edges : array-like or list of array-like
        The bin edges. If it is a single array, the bins are assumed to apply in all dimensions.
        If it is a list, it can have length 1 or have the same length as the number of columns in
        indices.
    """
    indices = np.asarray(indices)
    if indices.dtype.names:
        # this is a structured array and we want a plain array
        dtype = {indices.dtype[i] for i in range(len(indices.dtype))}
        if len(set(dtype)) > 1:
            raise ValueError(
                f"All d-dimensional indices must be of the same type. Got {dtype}"
            )
        indices = np.array(
            [indices[indices.dtype.names[i]] for i in range(len(indices.dtype.names))]
        ).T
    if len(indices.shape) == 1:
        indices = indices[:, np.newaxis]

    # Check and prepare bin_edges
    try:
        # this tries to figure out if the bins are a numpy 1d array
        if len(np.shape(bin_edges)) == 1:
            bin_edges = [bin_edges]
    except ValueError:
        pass

    for i, edges in enumerate(bin_edges):
        if np.any(indices[:, i] < 0):
            raise ValueError("Indices must be non-negative")
        if np.any(indices[:, i] >= len(edges) + 1):
            raise ValueError("Indices must be less than the number of bins")

    if len(bin_edges) != 1 and indices.shape[1] != len(bin_edges):
        raise ValueError(
            "bin_edges must be a list of length 1 or the number of columns in indices. "
            f"Got len(bin_edges)={len(bin_edges)}) and n_cols={indices.shape[1]}"
        )

    if len(bin_edges) == 1 and indices.shape[1] > 1:
        # if there is only one bin array, we assume it applies to all columns
        bin_edges = [bin_edges[0] for _ in range(indices.shape[1])]

    bin_edges = [np.atleast_1d(col_edges) for col_edges in bin_edges]
    bin_shapes = [len(col_edges.shape) for col_edges in bin_edges]
    if np.any(np.not_equal(bin_shapes, 1)):
        raise ValueError(f"bin_edges must be a list of 1D arrays. Got {bin_shapes}")

    return indices, bin_edges


def get_global_bin_idx(indices, bin_edges):
    """
    Utility to calculate a unique bin index from a set of indices.

    n-dimensional binning is done on n axes separately. That gives a tuple unique indices, one index
    for each axis. Sometimes it is usefull to have a single index.

    Parameters
    ----------
    indices : array-like
        The indices to check. It can be a structured or plain array.
        The array must be 1- or 2-dimensional. If it is 2-dimensional, the dimensionality of the
        data is given by the number of columns.
    bin_edges : array-like or list of array-like
        The bin edges. If it is a single array, the bins are assumed to apply in all dimensions.
        If it is a list, it can have length 1 or have the same length as the number of columns in
        indices.
    """
    # Check and normalize indices and bin edges
    indices, bin_edges = _check_arguments(indices, bin_edges)

    n_cols = len(bin_edges)
    sizes = [len(b) + 1 for b in bin_edges]
    offsets = {i: (np.prod(sizes[:i]) if i > 0 else 1) for i in range(n_cols)}

    global_bin_idx = np.zeros(len(indices), dtype=int)
    for bin_col in range(n_cols):
        global_bin_idx += offsets[bin_col] * indices[:, bin_col]
    return global_bin_idx


def _get_dtype(cols, extra_cols, success_column):
    """
    The standard dtype for get_histogram_quantile_ranges
    """
    dtype = [
        ("bin", np.int64),
        ("n", int),
        (success_column, int),
    ]
    for col in cols:
        dtype += [
            (f"{col}_bin", np.int64),
            (f"{col}_low", float),
            (f"{col}_high", float),
            (f"{col}", float),
            (f"{col}_delta", float),
        ]
    for col in cols + extra_cols:
        dtype += [
            (f"{col}_mean", float),
            (f"{col}_std", float),
        ]
    dtype += [
        ("sigma_1_low", int),
        ("sigma_1_high", int),
        ("sigma_2_low", int),
        ("sigma_2_high", int),
        ("median", int),
    ]
    return np.dtype(dtype)


def get_histogram_quantile_ranges(
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
    a-priori probability of success (used to calculate expected coverage intervals).

    The data is binned using the columns given in `bin_edges`. The bin edges are given as a
    dictionary of column names and bin edges. The bin edges must be 1D arrays.

    Summary statistics are calculated for the columns in `bin_edges` and `extra_cols`.

    The function bins the data according to the columns given in `bin_edges`, and creates the
    following columns:
    - `bin`: the unique bin index
    - `n`: the number of samples in the bin
    - `{col}_low`: the lower edge of the bin (repeated for all bin_edges keys and extra_cols)
    - `{col}_high`: the upper edge of the bin (repeated for all bin_edges keys and extra_cols)
    - `{col}`: the center of the bin (repeated for all bin_edges keys and extra_cols)
    - `{col}_delta`: the width of the bin (repeated for all bin_edges keys and extra_cols)
    - `{success_column}`: the number of "good" samples in the bin
    - `median`: the median
    - `sigma_1_low`: the lower 1-sigma quantile (15.9%)
    - `sigma_1_high`: the upper 1-sigma quantile (84.1%)
    - `sigma_2_low`: the lower 2-sigma quantile (2.27%)
    - `sigma_2_high`: the upper 2-sigma quantile (97.7%)

    NOTE: the result is NOT sparse (it will contain all bins, even if there are no samples in it).

    Parameters
    ----------
    data : Table
        The data to summarize. Must include a `prob_column` column and the corresponding columns
        for the bin edges.
    bin_edges : dict
        A dictionary of bin edges for each column.
    extra_cols : list, optional
        Extra columns to include in the output. These columns will have the mean and standard
        deviation within each bin calculated.
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

    data = data.copy()

    sizes = [len(b) + 1 for b in bin_edges.values()]

    # np.digitize returns indices assuming there are (N - 1) bins plus the under/overflow bins.
    # That's (N + 1) bins.
    # To easily set the bin ranges to all, including the under/over-flow, we will add -inf and +inf
    # to the bin edges.
    padded_bins = {
        col: np.concatenate([[-np.inf], bin_edges[col], [np.inf]]) for col in cols
    }

    bins = np.array(
        [np.digitize(data[bin_col], bin_edges[bin_col]) for bin_col in cols]
    ).T

    bin_cols = [f"{bin_col}_bin" for bin_col in cols]
    # this is a unique bin index added for convenience
    data["bin"] = get_global_bin_idx(bins, bin_edges.values())

    # this creates an array with all possible combinations of indices (sorted by global bin)
    bin_indices = np.meshgrid(*[list(range(s)) for s in sizes[::-1]], indexing="ij")[
        ::-1
    ]

    quantiles = np.zeros(
        np.prod(sizes),
        dtype=_get_dtype(cols, extra_cols, success_column),
    )

    # default non-zero values
    for col in cols + extra_cols:
        quantiles[f"{col}_mean"] = np.nan
        quantiles[f"{col}_std"] = np.nan
        quantiles[f"{col}_mean"] = np.nan
        quantiles[f"{col}_std"] = np.nan

    # bin index values
    for idx, bin_col in enumerate(cols):
        quantiles[f"{bin_col}_bin"] = bin_indices[idx].flatten()
    quantiles["bin"] = get_global_bin_idx(quantiles[bin_cols], bin_edges.values())

    # bin edges
    for col in cols:
        quantiles[f"{col}_low"] = padded_bins[col][quantiles[f"{col}_bin"]]
        quantiles[f"{col}_high"] = padded_bins[col][quantiles[f"{col}_bin"] + 1]
        quantiles[f"{col}"] = (quantiles[f"{col}_low"] + quantiles[f"{col}_high"]) / 2
        quantiles[f"{col}_delta"] = quantiles[f"{col}_high"] - quantiles[f"{col}_low"]

    # the actual data
    for bin_idx in np.unique(data["bin"]):
        # this is the only place where the global bin index is used.
        # one can replace this by a mask using the bin indices of each column if so desired.
        sel = data["bin"] == bin_idx

        sigma_2_low, sigma_1_low, median, sigma_1_high, sigma_2_high = get_quantiles(
            data[prob_column][sel], [2.27, 15.9, 50, 84.1, 97.7], n_samples
        )

        for col in cols + extra_cols:
            quantiles[f"{col}_mean"][bin_idx] = np.mean(data[col][sel])
            quantiles[f"{col}_std"][bin_idx] = np.std(data[col][sel])

        quantiles["sigma_1_low"][bin_idx] = sigma_1_low
        quantiles["sigma_1_low"][bin_idx] = sigma_1_low
        quantiles["sigma_2_low"][bin_idx] = sigma_2_low
        quantiles["median"][bin_idx] = median
        quantiles["sigma_1_high"][bin_idx] = sigma_1_high
        quantiles["sigma_1_high"][bin_idx] = sigma_1_high
        quantiles["sigma_2_high"][bin_idx] = sigma_2_high
        quantiles["n"][bin_idx] = np.count_nonzero(sel)
        quantiles[success_column][bin_idx] = np.count_nonzero(
            sel & data[success_column]
        )

    quantiles = Table(quantiles)
    quantiles.meta["shape"] = tuple(sizes)[::-1]

    return quantiles
