"""
Generate acquisition statistics report.
"""

import argparse
import json
import os
from pathlib import Path

from astropy import units as u
from cxotime import CxoTime
from ska_helpers import logging

from acq_stat_reports import config as conf
from acq_stat_reports import get_data, make_acq_plots, make_html

SKA = Path(os.environ["SKA"])


logger = logging.basic_logger("acq_stat_reports", level="INFO")


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=Path(SKA / "www" / "ASPECT" / "acq_stat_reports"),
        help="Output directory (default is $SKA/www/ASPECT/acq_stat_reports)",
        type=Path,
        dest="output_dir",
    )
    parser.add_argument(
        "--stop",
        help="End of the time range to consider (default is NOW)",
        default=None,
    )
    parser.add_argument(
        "--log-level",
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


def main(sys_args=None):
    import matplotlib

    matplotlib.use("agg")

    args = get_parser().parse_args(sys_args)

    logger.setLevel(args.log_level.upper())

    now = CxoTime()
    logger.info("---------- acq stat reports update at %s ----------" % (now.iso))

    stop = now if args.stop is None else CxoTime(args.stop)

    all_acq = get_data()

    time_ranges = [
        {
            "name": "month",
            "start": stop - 30 * u.day,
            "stop": stop,
        },
        {
            "name": "quarter",
            "start": stop - 90 * u.day,
            "stop": stop,
        },
        {
            "name": "half-year",
            "start": stop - 182 * u.day,
            "stop": stop,
        },
        {
            "name": "year",
            "start": stop - 365 * u.day,
            "stop": stop,
        },
    ]

    data = {
        "time": now.iso,
        "args": {k: str(val) for k, val in vars(args).items()},
        "time_ranges": [],
    }
    for time_range in time_ranges:
        tname = time_range["name"]
        output_dir = args.output_dir / tname
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Creating {tname} plots in {output_dir}")
        range_datestart = CxoTime(time_range["start"])
        range_datestop = CxoTime(time_range["stop"])

        # ignore acquisition stars that are newer than the end of the range
        # in question (happens during reprocessing) for consistency
        all_acq_upto = all_acq[all_acq["tstart"] <= CxoTime(range_datestop).secs]

        conf.output_dir = str(output_dir)
        conf.close_figures = True
        plots = make_acq_plots(
            all_acq_upto,
            tstart=range_datestart.secs,
            tstop=range_datestop.secs,
        )

        data["time_ranges"].append(
            {
                "name": tname,
                "datestring": tname,
                "datestart": range_datestart.date,
                "datestop": range_datestop.date,
                "human_date_start": range_datestart.datetime.strftime("%Y-%b-%d"),
                "human_date_stop": range_datestop.datetime.strftime("%Y-%b-%d"),
                "plots": plots,
            }
        )

    logger.debug(f"JSON to {args.output_dir}")
    with open(args.output_dir / "data.json", "w") as rep_file:
        rep_file.write(json.dumps(data, sort_keys=True, indent=2))

    make_html(data, outdir=args.output_dir)


if __name__ == "__main__":
    main()
