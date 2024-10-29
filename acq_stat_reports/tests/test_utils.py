import numpy as np
import pytest
from astropy.table import Table

from acq_stat_reports import utils


def test_histogram():
    example = Table(
        {
            "a": [
                8.26548573,
                8.86763105,
                5.49371611,
                3.61050205,
                9.7853743,
                1.13293811,
                2.06440575,
                3.59795432,
                4.94548012,
                9.06282222,
            ],
            "b": [
                3.9890854,
                1.40180462,
                6.73164771,
                5.61232687,
                5.17822797,
                5.84638758,
                3.584898,
                4.53757365,
                0.4635047,
                0.36043916,
            ],
            "success": [False, False, True, True, True, False, True, True, True, True],
            "prob": [0.5, 0.9, 0.3, 0.7, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    )
    example_copy = example.copy()

    bin_edges = {
        "a": np.array([1, 5, 9]),
        "b": np.array([1, 3.5, 7]),
    }
    hist = utils.get_histogram_quantile_ranges(
        example,
        bin_edges,
        success_column="success",
        prob_column="prob",
    )

    cols = [
        "a",
        "a_bin",
        "a_delta",
        "a_high",
        "a_low",
        "a_mean",
        "a_std",
        "b",
        "b_bin",
        "b_delta",
        "b_high",
        "b_low",
        "b_mean",
        "b_std",
        "bin",
        "median",
        "n",
        "sigma_1_high",
        "sigma_1_low",
        "sigma_2_high",
        "sigma_2_low",
        "success",
    ]

    # check that the input table was not modified
    assert example.colnames == example_copy.colnames
    for col in example.colnames:
        assert np.all(example[col] == example_copy[col])

    assert cols == sorted(hist.colnames)

    n_a = len(bin_edges["a"]) + 1
    n_b = len(bin_edges["b"]) + 1
    assert hist.meta["shape"] == (n_b, n_a)  # note that this is reversed
    # all bins are there (even empty bins) and none are repeated
    assert len(hist) == n_a * n_b
    assert np.unique(hist["a_bin"]).size == len(bin_edges["a"]) + 1
    assert np.unique(hist["b_bin"]).size == len(bin_edges["b"]) + 1
    # the global bin ID is sorted and goes from 0 to len(hist) - 1
    assert np.all(hist["bin"] == np.arange(len(hist)))

    # counts
    ref = np.array([[0, 1, 0, 1], [0, 0, 1, 0], [0, 4, 2, 1], [0, 0, 0, 0]])
    assert np.all(ref == np.asarray(hist["n"]).reshape((n_a, n_b)))
    ref = np.array([[0.0, 1.0, 0.0, 1.0], [0, 0, 0, 0], [0, 3, 1, 1], [0, 0, 0, 0]])
    assert np.all(ref == np.asarray(hist["success"]).reshape((n_a, n_b)))

    # bin centers along a-axis
    ref = np.array(
        [
            [-np.inf, 3.0, 7.0, np.inf],
            [-np.inf, 3.0, 7.0, np.inf],
            [-np.inf, 3.0, 7.0, np.inf],
            [-np.inf, 3.0, 7.0, np.inf],
        ]
    )
    assert np.all(np.asarray(hist["a"]).reshape((n_a, n_b)) == ref)

    # bin centers along b-axis
    ref = np.array(
        [
            [-np.inf, -np.inf, -np.inf, -np.inf],
            [2.25, 2.25, 2.25, 2.25],
            [5.25, 5.25, 5.25, 5.25],
            [np.inf, np.inf, np.inf, np.inf],
        ]
    )
    assert np.all(np.asarray(hist["b"]).reshape((n_a, n_b)) == ref)

    # quantiles
    ref = np.array([[0, 1, 0, 1], [0, 0, 1, 0], [0, 4, 1, 1], [0, 0, 0, 0]])
    assert np.all(np.asarray(hist["sigma_1_high"]).reshape((n_a, n_b)) == ref)

    ref = np.array([[0, 1, 0, 1], [0, 0, 1, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
    assert np.all(np.asarray(hist["sigma_1_low"]).reshape((n_a, n_b)) == ref)


def test_histogram_3d():
    example = Table()
    a, b, c = np.broadcast_arrays(
        np.array([0, 1, 2, 3, 4])[None, None, :],
        np.array([10, 11, 12, 13])[None, :, None],
        np.array([20, 21, 22])[:, None, None],
    )
    example["a"] = a.flatten()
    example["b"] = b.flatten()
    example["c"] = c.flatten()
    example["success"] = True
    example["prob"] = 1

    bin_edges = {
        "a": np.array([0.5, 1.5, 2.5, 3.5]),
        "b": np.array([10.5, 11.5, 12.5]),
        "c": np.array([20.5, 21.5]),
    }
    hist = utils.get_histogram_quantile_ranges(
        example,
        bin_edges,
        success_column="success",
        prob_column="prob",
    )

    assert np.all(hist["n"] == 1)
    assert np.all(hist["success"] == 1)

    cols = [
        "a",
        "a_bin",
        "a_delta",
        "a_high",
        "a_low",
        "a_mean",
        "a_std",
        "b",
        "b_bin",
        "b_delta",
        "b_high",
        "b_low",
        "b_mean",
        "b_std",
        "bin",
        "c",
        "c_bin",
        "c_delta",
        "c_high",
        "c_low",
        "c_mean",
        "c_std",
        "sigma_1_low",
        "sigma_1_high",
        "sigma_2_low",
        "sigma_2_high",
        "median",
        "n",
        "success",
    ]

    assert sorted(cols) == sorted(hist.colnames)


def test_global_bin():
    # Basic arguments (no implicit conversions)

    # 1d case with 10 bins plus under/over-flow
    # under-flow is 0, over-flow is 11 (what actually matters is the shape of the bins)
    bin_ranges = [
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    ]  # list of arrays
    indices = np.array([[2, 3, 2, 11, 11, 0, 4, 9, 8, 4]]).T  # 2d array

    # in 1d, the global bin is trivial
    assert np.all(utils.get_global_bin_idx(indices, bin_ranges) == indices.flatten())

    # 2d case with 4 bins in the first dimension and 3 bins in the second
    bin_ranges = [list(range(5)), list(range(4))]
    # these are all the possible bin indices (sorted by global bin)
    indices = np.array(
        [
            [
                0,
                1,
                2,
                3,
                4,
                5,
                0,
                1,
                2,
                3,
                4,
                5,
                0,
                1,
                2,
                3,
                4,
                5,
                0,
                1,
                2,
                3,
                4,
                5,
                0,
                1,
                2,
                3,
                4,
                5,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                4,
            ],
        ]
    ).T
    idx = np.arange(len(indices))
    assert np.all(utils.get_global_bin_idx(indices, bin_ranges) == idx)

    # Table is a possible input
    table = Table(
        {
            "a": indices[:, 0],
            "b": indices[:, 1],
        }
    )
    assert np.all(utils.get_global_bin_idx(table, bin_ranges) == idx)

    # 3d case
    shape = [5, 6, 4]
    bins = [list(range(s)) for s in shape]
    idx1, idx2, idx3 = np.broadcast_arrays(
        np.arange(shape[0] + 1)[np.newaxis, np.newaxis, :],
        np.arange(shape[1] + 1)[np.newaxis, :, np.newaxis],
        np.arange(shape[2] + 1)[:, np.newaxis, np.newaxis],
    )
    indices = np.vstack([idx1.flatten(), idx2.flatten(), idx3.flatten()]).T
    idx = np.arange(len(indices))
    assert np.all(utils.get_global_bin_idx(indices, bins) == idx)

    # Argument validation

    # 1d case. It should be possible to pass 1d arrays (both indices and bin_ranges)
    bin_ranges = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # array
    indices = np.array([2, 3, 2, 11, 11, 0, 4, 9, 8, 4])  # 1d array
    assert np.all(utils.get_global_bin_idx(indices, bin_ranges) == np.array(indices))

    # 2d case. If only one array of bin ranges is passed, it is used for all dimensions
    bin_ranges = list(range(4))
    indices = np.array(
        [
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        ]
    ).T
    idx = np.arange(len(indices))

    assert np.all(utils.get_global_bin_idx(indices, [bin_ranges, bin_ranges]) == idx)
    assert np.all(utils.get_global_bin_idx(indices, [bin_ranges]) == idx)
    assert np.all(utils.get_global_bin_idx(indices, bin_ranges) == idx)

    # invalid bins
    bin_ranges = [list(range(3)), list(range(4))]
    with pytest.raises(ValueError):
        utils.get_global_bin_idx([[-1, 2]], bin_ranges)

    with pytest.raises(ValueError):
        utils.get_global_bin_idx([[1, -1]], bin_ranges)

    utils.get_global_bin_idx([[3, 4]], bin_ranges)
    with pytest.raises(ValueError):
        utils.get_global_bin_idx([[3, 5]], bin_ranges)

    utils.get_global_bin_idx([[3, 2]], bin_ranges)
    with pytest.raises(ValueError):
        utils.get_global_bin_idx([[4, 2]], bin_ranges)
