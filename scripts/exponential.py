"""Ten real-world "how long until...?" questions modeled with the Exponential distribution.

Each function wraps `scipy.stats.expon` around one concrete question.
`rate` is the expected number of events per unit time (same lambda you'd
plug into a matching Poisson model); scipy's `expon` takes `scale = 1/rate`.
"""

from scipy.stats import expon


def time_to_server_failure(rate: float, t: float) -> float:
    """How long until a critical server fails, given it fails `rate` times/year?

    Probability the server fails within `t` years.

    Example
    -------
    A server fails on average 0.5 times/year. Chance it fails within the
    first 6 months (0.5 years):

    >>> round(time_to_server_failure(rate=0.5, t=0.5), 4)
    0.2212
    """
    return float(expon.cdf(t, scale=1 / rate))


def time_to_next_claim(rate: float, t: float) -> float:
    """How long until the next claim is filed on a policy?

    Probability the next claim arrives within `t` years, given `rate`
    claims/year.

    Example
    -------
    A policy averages 4 claims/year. Chance the next claim arrives
    within 1 month (1/12 year):

    >>> round(time_to_next_claim(rate=4, t=1 / 12), 4)
    0.2835
    """
    return float(expon.cdf(t, scale=1 / rate))


def survival_probability(rate: float, t: float) -> float:
    """How long will a policyholder survive after diagnosis (constant-hazard model)?

    Probability of surviving *beyond* `t` years, given a constant hazard
    (failure) `rate`/year. A simplification -- real mortality hazard
    increases with age, unlike the flat rate assumed here.

    Example
    -------
    A constant hazard rate of 0.1/year. Chance of surviving beyond 5 years:

    >>> round(survival_probability(rate=0.1, t=5), 4)
    0.6065
    """
    return float(expon.sf(t, scale=1 / rate))


def time_to_next_call(rate: float, t: float) -> float:
    """How long until the next customer arrives at a call center?

    Probability the next call arrives within `t` hours, given `rate`
    calls/hour.

    Example
    -------
    A call center averages 12 calls/hour. Chance the next call arrives
    within 5 minutes (5/60 hour):

    >>> round(time_to_next_call(rate=12, t=5 / 60), 4)
    0.6321
    """
    return float(expon.cdf(t, scale=1 / rate))


def gap_between_earthquakes(rate: float, t: float) -> float:
    """What's the time gap between consecutive earthquakes in a region?

    Probability the gap until the next earthquake exceeds `t` years,
    given `rate` earthquakes/year.

    Example
    -------
    A region averages 0.4 earthquakes/year. Chance the gap to the next
    one exceeds 3 years:

    >>> round(gap_between_earthquakes(rate=0.4, t=3), 4)
    0.3012
    """
    return float(expon.sf(t, scale=1 / rate))


def time_to_part_wearout(rate: float, t: float) -> float:
    """How long until a machine part wears out (constant failure-rate assumption)?

    Probability the part fails within `t` years, given a constant
    failure `rate`/year.

    Example
    -------
    A part fails on average 0.2 times/year. Chance it fails within 2 years:

    >>> round(time_to_part_wearout(rate=0.2, t=2), 4)
    0.3297
    """
    return float(expon.cdf(t, scale=1 / rate))


def time_to_next_fraud_detection(rate: float, t: float) -> float:
    """How long until the next fraudulent claim is detected?

    Probability the next fraud detection happens within `t` weeks,
    given `rate` detections/week.

    Example
    -------
    A fraud team averages 2 detections/week. Chance the next detection
    happens within half a week:

    >>> round(time_to_next_fraud_detection(rate=2, t=0.5), 4)
    0.6321
    """
    return float(expon.cdf(t, scale=1 / rate))


def expected_gap_between_large_losses(rate: float) -> float:
    """What's the expected wait time between large reinsurance losses?

    Mean time between events, given `rate` large losses/year.

    Example
    -------
    A reinsurer averages 1.5 large losses/year. Expected wait between
    losses:

    >>> round(expected_gap_between_large_losses(rate=1.5), 4)
    0.6667
    """
    return float(expon(scale=1 / rate).mean())


def time_to_bulb_failure(mean_lifetime: float, t: float) -> float:
    """How long until a lightbulb burns out, given a known average lifetime?

    Probability the bulb fails within `t` hours, given a `mean_lifetime`
    in hours (equivalent to rate = 1 / mean_lifetime).

    Example
    -------
    A bulb has a 1,000-hour average lifetime. Chance it fails within
    500 hours:

    >>> round(time_to_bulb_failure(mean_lifetime=1000, t=500), 4)
    0.3935
    """
    return float(expon.cdf(t, scale=mean_lifetime))


def time_to_next_catastrophic_claim(rate: float, t: float) -> float:
    """How long until the next large (catastrophic) claim event hits the portfolio?

    Probability the next catastrophic claim takes *longer than* `t`
    years to occur, given `rate` catastrophic claims/year.

    Example
    -------
    A portfolio averages 1.5 catastrophic claims/year. Chance none
    occurs in the next 2 years:

    >>> round(time_to_next_catastrophic_claim(rate=1.5, t=2), 4)
    0.0498
    """
    return float(expon.sf(t, scale=1 / rate))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
