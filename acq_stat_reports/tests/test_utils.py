import numpy as np
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
