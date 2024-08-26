#!/usr/bin/env python
"""
Create summary page.
"""

import argparse
import json
import os
import time
from pathlib import Path

import jinja2
import matplotlib
import numpy as np
from cxotime import CxoTime
from ska_helpers import logging

if __name__ == "__main__":
    # Matplotlib setup
    # Use Agg backend for command-line (non-interactive) operation
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates" / "acq_stats")
)
SKA = Path(os.environ["SKA"])

logger = logging.basic_logger("acq_stat_reports", level="INFO")


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


def main():  # noqa: PLR0915
    args = get_parser().parse_args()

    logger.setLevel(args.v.upper())

    nowdate = time.ctime()
    logger.info(f"---------- acq stat reports summary update at {nowdate} ----------")

    datadir = args.datadir
    plotdir = args.webdir / "summary"
    plotdir.mkdir(exist_ok=True, parents=True)

    time_pad = 0.05

    data = {
        "month": datadir.glob("????/M??/rep.json"),
        "quarter": datadir.glob("????/Q?/rep.json"),
        "semi": datadir.glob("????/S?/rep.json"),
        "year": datadir.glob("????/YEAR/rep.json"),
    }

    for d in data:
        data[d] = sorted(data[d])
        rates = {
            ftype: {
                "time": np.array([]),
                "rate": np.array([]),
                "err_h": np.array([]),
                "err_l": np.array([]),
            }
            for ftype in ["fail_rate"]
        }

        for p in data[d]:
            with open(p, "r") as rep_file:
                rep_text = rep_file.read()
            rep = json.loads(rep_text)
            start = CxoTime(rep["datestart"])
            stop = CxoTime(rep["datestop"])
            datetime = start + (stop - start) / 2
            for ftype in rates:
                rates[ftype]["time"] = np.append(
                    rates[ftype]["time"], datetime.frac_year
                )
                for ftype2 in ["fail_rate"]:
                    rates[ftype2]["rate"] = np.append(
                        rates[ftype2]["rate"], rep["fail_rate"]
                    )
                    rates[ftype2]["err_h"] = np.append(
                        rates[ftype2]["err_h"], rep["fail_rate_err_high"]
                    )
                    rates[ftype2]["err_l"] = np.append(
                        rates[ftype2]["err_l"], rep["fail_rate_err_low"]
                    )

        for ftype in ["fail_rate"]:
            fig1 = plt.figure(1, figsize=(5, 3))
            ax1 = fig1.gca()
            fig2 = plt.figure(2, figsize=(5, 3))
            ax2 = fig2.gca()

            ax1.plot(
                rates[ftype]["time"],
                rates[ftype]["rate"],
                color="black",
                linestyle="",
                marker=".",
                markersize=5,
            )
            ax1.grid()
            ax2.errorbar(
                rates[ftype]["time"],
                rates[ftype]["rate"],
                yerr=np.array([rates[ftype]["err_l"], rates[ftype]["err_h"]]),
                color="black",
                linestyle="",
                marker=".",
                markersize=5,
            )
            ax2.grid()

            now_frac = CxoTime().frac_year

            ax2_ylim = ax2.get_ylim()
            # pad a bit below 0 relative to ylim range
            ax2.set_ylim(ax2_ylim[0] - 0.025 * (ax2_ylim[1] - ax2_ylim[0]))
            ax1.set_ylim(ax2.get_ylim())

            for ax in [ax1, ax2]:
                dxlim = now_frac - 2000
                ax.set_xlim(2000, now_frac + time_pad * dxlim)
                #    ax = fig.get_axes()[0]
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                for label in labels:
                    label.set_size("small")
                ax.set_ylabel("Rate", fontsize=12)
                ax.set_title("%s %s" % (d, ftype), fontsize=12)

            fig1.subplots_adjust(left=0.15)
            fig2.subplots_adjust(left=0.15)
            fig1.savefig(plotdir / f"summary_{d}_{ftype}.png")
            fig2.savefig(plotdir / f"summary_{d}_{ftype}_eb.png")
            plt.close(fig1)
            plt.close(fig2)

    outfile = plotdir / "acq_summary.html"
    template = JINJA_ENV.get_template("summary.html")
    page = template.render()
    with open(outfile, "w") as fh:
        fh.write(page)

    logger.info("---------- acq stat reports summary update complete ----------")


if __name__ == "__main__":
    main()
