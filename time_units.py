from __future__ import annotations

import math

MONTHS_PER_YEAR = 12
_FLOAT_TOLERANCE = 1e-6


def years_to_months(years: float) -> int:
    """Convert a simulation time expressed in years to an integer timestep in months."""
    if not isinstance(years, (int, float)):
        raise TypeError("Years must be a numeric value.")
    if not math.isfinite(years):
        raise ValueError("Years must be a finite number.")
    if years < 0:
        raise ValueError("Years must be non-negative.")

    months = years * MONTHS_PER_YEAR
    rounded = round(months)
    if not math.isclose(months, rounded, rel_tol=_FLOAT_TOLERANCE, abs_tol=_FLOAT_TOLERANCE):
        raise ValueError("Years value must correspond to a whole number of months.")
    return int(rounded)


def months_to_years(months: int) -> float:
    """Convert an integer timestep expressed in months to years."""
    if not isinstance(months, (int, float)):
        raise TypeError("Months must be a numeric value.")
    if not math.isfinite(months):
        raise ValueError("Months must be a finite number.")
    if months < 0:
        raise ValueError("Months must be non-negative.")
    return float(months) / MONTHS_PER_YEAR
