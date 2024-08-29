"""
Create the table of contents for the acquisition stat reports.
"""

import argparse
import os
from pathlib import Path

import jinja2

WEBDATA = Path(os.environ["SKA"]) / "www" / "ASPECT" / "acq_stat_reports"

JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates" / "acq_stats")
)

MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


CELL_SPAN = {
    "M": 1,
    "Q": 3,
    "S": 6,
    "Y": 12,
}
"""
The number of months in each interval type
"""


START_MONTH = {
    "M01": 0,
    "M02": 1,
    "M03": 2,
    "M04": 3,
    "M05": 4,
    "M06": 5,
    "M07": 6,
    "M08": 7,
    "M09": 8,
    "M10": 9,
    "M11": 10,
    "M12": 11,
    "Q1": -2,
    "Q2": 1,
    "Q3": 4,
    "Q4": 7,
    "S1": -5,
    "S2": 1,
    "YEAR": 0,
}
"""
The starting month of each interval, relative to January
"""


def get_cells(year, interval):
    span = CELL_SPAN[interval[0]]
    start = START_MONTH[interval]
    year = int(year)
    if start < 0:
        return [
            {"year": (year - 1), "start": start % 12, "span": -start},
            {"year": year, "start": 0, "span": span + start},
        ]
    else:
        return [{"year": year, "start": start, "span": span}]


def get_toc(data_dir=None):
    if data_dir is None:
        data_dir = WEBDATA

    # these are all the expected intervals
    all_years = [int(p.name) for p in sorted(data_dir.glob("????"))]
    all_semi = [f"S{semi:01d}" for semi in range(1, 3)]
    all_quarters = [f"Q{quarter:01d}" for quarter in range(1, 5)]
    all_months = [f"M{month:02d}" for month in range(1, 13)]

    all_cells = {
        (year, interval): get_cells(year, interval)
        for year in all_years
        for interval in all_months + all_quarters + all_semi + ["YEAR"]
    }

    # these are the actual data directories
    directories = (
        sorted(data_dir.glob("*/M??"))
        + sorted(data_dir.glob("*/Q?"))
        + sorted(data_dir.glob("*/S?"))
        + sorted(data_dir.glob("*/YEAR"))
    )
    for path in directories:
        for cell in all_cells[(int(path.parent.name), path.name)]:
            cell["path"] = path.relative_to(data_dir)

    # and these are the table cells (one interval can be split into two cells)
    values = [
        [val["year"], interval, val["start"], val["span"], val.get("path", "")]
        for (_, interval), val1 in all_cells.items()
        for val in val1
        if val["year"] >= all_years[0]
    ]

    # this merges empty cells
    remove = []
    stack = []
    stack_year = None
    stack_interval_type = None
    for idx, (year, interval, _, _, path) in enumerate(values):
        if year != stack_year or interval[0] != stack_interval_type:
            stack_year = year
            stack_interval_type = interval[0]
            if stack:
                values[stack[0]][3] = sum([values[idx][3] for idx in stack])
                remove += stack[1:]
                stack = []
        if not path:
            stack.append(idx)

    for idx in remove[::-1]:
        del values[idx]

    return values


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--webdir",
        default=Path(WEBDATA),
        help="Output directory",
        type=Path,
    )
    return parser


def main():
    args = get_parser().parse_args()

    values = get_toc(data_dir=args.webdir)
    all_years = sorted({val[0] for val in values})
    semi_data = [
        [
            [val[3], str(val[4]), val[1], val[0]]
            for val in values
            if val[0] == year and val[1][0] == "S"
        ]
        for year in all_years
    ]
    monthly_data = [
        [
            [val[3], str(val[4]), MONTH_NAMES[int(val[1].replace("M", "")) - 1], val[0]]
            for val in values
            if val[0] == year and val[1][0] == "M"
        ]
        for year in all_years
    ]
    quarterly_data = [
        [
            [val[3], str(val[4]), val[1], val[0]]
            for val in values
            if val[0] == year and val[1][0] == "Q"
        ]
        for year in all_years
    ]
    yearly_data = [
        [
            [val[3], str(val[4]), val[1], val[0]]
            for val in values
            if val[0] == year and val[1][0] == "Y"
        ]
        for year in all_years
    ]

    template = JINJA_ENV.get_template("toc.html")
    text = template.render(
        monthly_data=monthly_data,
        quarterly_data=quarterly_data,
        semi_data=semi_data,
        yearly_data=yearly_data,
    )
    with open(args.webdir / "index.html", "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
