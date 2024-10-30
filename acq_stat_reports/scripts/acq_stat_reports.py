"""
Generate acquisition statistics report.
"""

import argparse
import json
import os
from pathlib import Path

import ska_report_ranges
from astropy import units as u
from cxotime import CxoTime
from ska_helpers import logging

from acq_stat_reports import get_acq_info, get_data, make_acq_plots, make_html
from acq_stat_reports.config import conf

SKA = Path(os.environ["SKA"])


logger = logging.basic_logger("acq_stat_reports", level="INFO")


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
        dest="output_dir",
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

        output_dir = (
            args.output_dir / f"{to_update[tname]['year']}" / to_update[tname]["subid"]
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        dataout = (
            args.datadir / f"{to_update[tname]['year']}" / to_update[tname]["subid"]
        )
        dataout.mkdir(parents=True, exist_ok=True)

        logger.debug("Plots and HTML to %s" % output_dir)
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

        conf.output_dir = str(output_dir)
        conf.close_figures = True
        make_acq_plots(
            all_acq_upto,
            tstart=range_datestart.secs,
            tstop=range_datestop.secs,
        )
        make_html(nav, acq_info, outdir=output_dir)


if __name__ == "__main__":
    main()
