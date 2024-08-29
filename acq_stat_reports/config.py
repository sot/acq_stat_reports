import astropy.config as _config_


class ConfigNamespace(_config_.ConfigNamespace):
    rootname = "ska_trending"

    data_dir = _config_.ConfigItem(
        ".",
        "Top-level data directory",
    )
    figure_width = _config_.ConfigItem(
        5,
        "Default figure width in inches",
    )
    figure_height = _config_.ConfigItem(
        2.5,
        "Default figure height in inches",
    )
    close_figures = _config_.ConfigItem(
        False,
        "Close matplotlib figures after plotting",
    )
    remove_bad_stars = _config_.ConfigItem(
        False,
        "Do not include bad stars in the reports",
    )
    mpl_style = _config_.ConfigItem(
        "bmh",
        "Matplotlib style to use",
    )

    def __str__(self):
        from pprint import pformat

        return pformat({k: getattr(self, k) for k in self})


OPTIONS = ConfigNamespace()
