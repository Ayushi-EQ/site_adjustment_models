"""Site adjustment factor calculations.

This module implements site adjustment factors from "Parameterized
models of systematic site effects from hybrid ground-motion
simulations in New Zealand". The models are provided as a reference
for calculations made in the paper.

The classifications of hill, valley, basin, unmodelled basin, and
basin edge are defined in the referenced paper.

Functions
---------
site_adjustment_hill
    Site adjustment factor calculations for hill sites. Requires both
    a period (seconds) and an H1250 value (m).
site_adjustment_unmodelled_basin
    Site adjustment factor calculations for unmodelled basin sites.
    Requires both period (seconds) and fundamental period T0 (seconds).
site_adjustment_basin_edge
    Site adjustment factor calculations for basin edge sites.
site_adjustment
    Site adjustment factor calculations for sites, dispatched to one
    of the above functions using the `site_class` parameter.

Classes
-------
SiteClass
    The site class, one of `HILL`, `VALLEY`, `BASIN`, `UNMODELLED_BASIN`, and `BASIN_EDGE`.
"""

from enum import Enum, auto
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

COEFFICIENT_TABLE = pd.read_csv(
    Path(__file__).parent.parent / "data" / "Coefficients_Paper1_Final.csv"
).set_index("T")


def _closest_coefficients_for_period(
    period: npt.ArrayLike, coefficient: str
) -> npt.NDArray[np.float64]:
    """Find the closest defined coefficient for a given period.

    Given a period (p) and a coefficient, lookup the first row i in
    the coefficient table such that ``p[i - 1] < p <= p[i]``. Then,
    select the value closest to `p` out of ``p[i - 1]`` and ``p[i]``.

    Parameters
    ----------
    period : npt.ArrayLike
        The period(s) to lookup.
    coefficient : str
        The coefficient to lookup.


    Returns
    -------
    npt.NDArray[np.float64]
        The coefficient value(s) for the given period(s).

    Examples
    --------
    >>> _closest_coefficients_for_period([0.01, 0.2 + 1e-6], 'h1')
    array([0.        , 0.00569733])
    """
    period = np.asarray(period)
    coefficients = COEFFICIENT_TABLE[coefficient]
    idx_right = np.searchsorted(coefficients.index.values, period, side="left")
    idx_left = np.clip(idx_right - 1, 0, len(coefficients.index.values) - 1)

    left_diff = np.abs(coefficients.index.values[idx_left] - period)
    right_diff = np.abs(coefficients.index.values[idx_right] - period)
    idx = np.where(left_diff < right_diff, idx_left, idx_right)

    return coefficients.values[idx]


def site_adjustment_hill(period: npt.ArrayLike, h_1250: npt.ArrayLike) -> npt.ArrayLike:
    """Calculate site adjustment factors for hill sites.

    This implements Equation 6 for Hill sites.

    Parameters
    ----------
    period : npt.ArrayLike
        The period(s) to calculate the site adjustment factors for.
    h_1250 : npt.ArrayLike
        The H1250 value for the given site. H1250 is defined as the
        difference between site elevation, and mean elevation of the
        DEM in a radius of 1250m.


    Returns
    -------
    npt.ArrayLike
        The site adjustment factor for the given period(s) and H1250
        value.

    Examples
    --------
    >>> site_adjustment_hill(0.1, 0)
    array(0.)
    >>> site_adjustment_hill([0.1, 0.2], -1.0)
    array([-0.        , -0.00569733])
    """
    h1 = _closest_coefficients_for_period(period, "h1")
    h2 = _closest_coefficients_for_period(period, "h2")
    return np.where(
        h_1250 < 0, h1 * np.maximum(-50, h_1250), h2 * np.minimum(70, h_1250)
    )


def site_adjustment_unmodelled_basin(
    period: npt.ArrayLike, t0: npt.ArrayLike
) -> npt.ArrayLike:
    """Calculate site adjustment factors for unmodelled basins.

    This function implements equations 7 and 8 for sites unmodelled
    basins.

    Parameters
    ----------
    period : npt.ArrayLike
        The period(s) to calculate adjustment factors for.
    t0 : npt.ArrayLike
        The fundamental period at the site in the unmodelled basin.

    Returns
    -------
    npt.ArrayLike
        The adjustment factor(s) for the period(s).

    Examples
    --------
    >>> site_adjustment_unmodelled_basin([0.1, 0.2, 0.5, 2.0], 1.5)
    array([0.        , 0.        , 0.        , 0.36025334])
    """
    m1 = _closest_coefficients_for_period(period, "m1")
    m2 = _closest_coefficients_for_period(period, "m2")
    n1 = _closest_coefficients_for_period(period, "n1")
    n2 = _closest_coefficients_for_period(period, "n2")
    period = np.asarray(period)

    return np.where(
        period <= 1,
        m1 * np.log(np.clip(t0, 0.4, 0.95)) + n1,
        m2 * np.log(np.clip(t0, 0.4, 1.1)) + n2,
    )


def site_adjustment_basin_edge(period: npt.ArrayLike) -> npt.ArrayLike:
    """Site adjustment factor calculations for basin edge sites.

    This function linearly interpolates values in the 'mean' column of
    the coefficients table. A subset of this table is present in Table 2.

    Parameters
    ----------
    period : npt.ArrayLike
        The period(s) to calculate adjustment factors for.


    Returns
    -------
    npt.ArrayLike
        The adjustment factors for the give period(s).

    Examples
    --------
    >>> site_adjustment_basin_edge([0.1, 0.2, 0.5, 2.0])
    array([0.22198625, 0.32827271, 0.22368033, 0.14316994])
    """
    mean = COEFFICIENT_TABLE["mean"]
    return np.interp(period, mean.index, mean.values)


class SiteClass(Enum):
    """Site class enumeration."""

    UNMODELLED_BASIN = auto()
    HILL = auto()
    BASIN_EDGE = auto()
    BASIN = auto()
    VALLEY = auto()


def site_adjustment(
    site_class: SiteClass,
    period: npt.ArrayLike,
    t0: npt.ArrayLike | None = None,
    h_1250: npt.ArrayLike | None = None,
) -> npt.ArrayLike:
    """Calculate site class adjustments for a site of a given class.

    Parameters
    ----------
    site_class : SiteClass
        The site geomorphological class.
    period : npt.ArrayLike
        The period(s) to calculate for.
    t0 : npt.ArrayLike | None
        The fundamental period of the site. Required if `site_class`
        is `SiteClass.UNMODELLED_BASIN`.
    h_1250 : npt.ArrayLike | None
        The H1250 value of the site. Required if `site_class` is
        `SiteClass.HILL`.

    Returns
    -------
    npt.ArrayLike
        The site adjustment factor(s) for the given period(s).

    Examples
    --------
    >>> site_adjustment(SiteClass.HILL, [0.1, 0.2], h_1250=-1.0)
    array([-0.        , -0.00569733])
    >>> site_adjustment(SiteClass.UNMODELLED_BASIN, [0.1, 0.2, 0.5, 2.0], t0=1.5)
    array([0.        , 0.        , 0.        , 0.36025334])
    >>> site_adjustment(SiteClass.BASIN_EDGE, [0.1, 0.2, 0.5, 2.0])
    array([0.22198625, 0.32827271, 0.22368033, 0.14316994])
    >>> site_adjustment(SiteClass.BASIN, [0.1, 0.2, 0.5])
    array([0., 0., 0.])
    >>> site_adjustment(SiteClass.VALLEY, 0.1)
    array(0.)
    """
    match site_class:
        case SiteClass.UNMODELLED_BASIN:
            if t0 is None:
                return ValueError("t0 must be supplied for unmodelled basin sites.")
            return site_adjustment_unmodelled_basin(period, t0)
        case SiteClass.HILL:
            if h_1250 is None:
                return ValueError("t0 must be supplied for hill basin sites.")
            return site_adjustment_hill(period, h_1250)
        case SiteClass.BASIN_EDGE:
            return site_adjustment_basin_edge(period)
        case _:
            return np.zeros_like(period)
