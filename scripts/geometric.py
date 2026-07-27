"""Ten real-world "how many trials until the first success?" questions modeled with the Geometric distribution.

Each function wraps `scipy.stats.geom` around one concrete question.
`p` is the per-trial success probability; `k` counts trials starting
at 1 (scipy's convention: the first trial is k=1, not k=0).
"""

from scipy.stats import geom


def trials_until_first_claim(p: float, k: int) -> float:
    """On exactly which policy year does a policyholder's first claim occur?

    Probability the first claim happens on exactly trial `k`, given
    per-year claim probability `p`.

    Example
    -------
    A policyholder has a 10% claim probability each year. Chance the
    first claim happens exactly in year 5:

    >>> round(trials_until_first_claim(p=0.1, k=5), 4)
    0.0656
    """
    return float(geom.pmf(k, p))


def trials_until_first_claim_within(p: float, k: int) -> float:
    """What's the chance a policyholder's first claim happens within k years?

    Probability the first claim happens on or before trial `k`, given
    per-year claim probability `p`.

    Example
    -------
    A policyholder has a 10% claim probability each year. Chance the
    first claim happens within 5 years:

    >>> round(trials_until_first_claim_within(p=0.1, k=5), 4)
    0.4095
    """
    return float(geom.cdf(k, p))


def expected_years_until_first_claim(p: float) -> float:
    """On average, how many years until a policyholder's first claim?

    Expected number of trials until the first claim, given per-year
    claim probability `p`.

    Example
    -------
    A policyholder has a 10% claim probability each year:

    >>> round(expected_years_until_first_claim(p=0.1), 4)
    10.0
    """
    return float(geom(p).mean())


def calls_until_first_sale(p: float, k: int) -> float:
    """On exactly which sales call does the first sale happen?

    Probability the first sale happens on exactly call `k`, given
    per-call conversion probability `p`.

    Example
    -------
    A sales rep converts 20% of calls. Chance the first sale happens on
    exactly the 3rd call:

    >>> round(calls_until_first_sale(p=0.2, k=3), 4)
    0.128
    """
    return float(geom.pmf(k, p))


def calls_until_first_sale_within(p: float, k: int) -> float:
    """What's the chance the first sale happens within the first k calls?

    Probability the first sale happens on or before call `k`, given
    per-call conversion probability `p`.

    Example
    -------
    A sales rep converts 20% of calls. Chance the first sale happens
    within 3 calls:

    >>> round(calls_until_first_sale_within(p=0.2, k=3), 4)
    0.488
    """
    return float(geom.cdf(k, p))


def attempts_until_first_approval(p: float, k: int) -> float:
    """On exactly which resubmission does an application first get approved?

    Probability the first approval happens on exactly attempt `k`,
    given per-attempt approval probability `p`.

    Example
    -------
    An application has a 60% approval chance per attempt. Chance it
    first gets approved on exactly the 2nd attempt:

    >>> round(attempts_until_first_approval(p=0.6, k=2), 4)
    0.24
    """
    return float(geom.pmf(k, p))


def inspections_until_first_defect(p: float, k: int) -> float:
    """On exactly which inspection does the first defect turn up?

    Probability the first defect is found on exactly inspection `k`,
    given per-inspection defect probability `p`.

    Example
    -------
    A production line has a 5% chance of a defect per inspection.
    Chance the first defect turns up on exactly the 10th inspection:

    >>> round(inspections_until_first_defect(p=0.05, k=10), 4)
    0.0315
    """
    return float(geom.pmf(k, p))


def expected_inspections_until_defect(p: float) -> float:
    """On average, how many inspections until the first defect is found?

    Expected number of trials until the first defect, given
    per-inspection defect probability `p`.

    Example
    -------
    A production line has a 5% chance of a defect per inspection:

    >>> round(expected_inspections_until_defect(p=0.05), 4)
    20.0
    """
    return float(geom(p).mean())


def years_until_first_lapse_within(p: float, k: int) -> float:
    """What's the chance a policy's first lapse happens within k years?

    Probability the first lapse happens on or before year `k`, given
    per-year lapse probability `p`.

    Example
    -------
    A policy has a 15% lapse probability each year. Chance it first
    lapses within 3 years:

    >>> round(years_until_first_lapse_within(p=0.15, k=3), 4)
    0.3859
    """
    return float(geom.cdf(k, p))


def expected_calls_until_first_sale(p: float) -> float:
    """On average, how many calls until the first sale?

    Expected number of trials until the first sale, given per-call
    conversion probability `p`.

    Example
    -------
    A sales rep converts 20% of calls:

    >>> round(expected_calls_until_first_sale(p=0.2), 4)
    5.0
    """
    return float(geom(p).mean())


if __name__ == "__main__":
    import doctest

    doctest.testmod()
