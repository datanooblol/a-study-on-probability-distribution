"""Ten real-world "equally likely anywhere in a range" questions modeled with the Uniform distribution.

Each function wraps `scipy.stats.uniform` around one concrete question.
`a` and `b` are the lower and upper bounds of the range; scipy's
`uniform` is parameterized by `loc=a, scale=b-a`, not `a, b` directly.
"""

from scipy.stats import uniform


def _dist(a: float, b: float):
    return uniform(loc=a, scale=b - a)


def probability_arrival_time_before(a: float, b: float, x: float) -> float:
    """What's the chance a customer arrives before minute x of a service window?

    Probability a uniformly distributed arrival time falls at or before
    `x`, given the window [`a`, `b`].

    Example
    -------
    Arrivals are uniform across a 30-minute window. Chance a customer
    arrives before minute 10:

    >>> round(probability_arrival_time_before(a=0, b=30, x=10), 4)
    0.3333
    """
    return float(_dist(a, b).cdf(x))


def probability_arrival_time_after(a: float, b: float, x: float) -> float:
    """What's the chance a customer arrives after minute x of a service window?

    Probability a uniformly distributed arrival time falls after `x`,
    given the window [`a`, `b`].

    Example
    -------
    Arrivals are uniform across a 30-minute window. Chance a customer
    arrives after minute 25:

    >>> round(probability_arrival_time_after(a=0, b=30, x=25), 4)
    0.1667
    """
    return float(_dist(a, b).sf(x))


def probability_wait_time_between(a: float, b: float, low: float, high: float) -> float:
    """What's the chance a wait time falls between two values?

    Probability a uniformly distributed wait time falls in
    [`low`, `high`], given the range [`a`, `b`].

    Example
    -------
    Wait times are uniform across 0-60 minutes. Chance the wait falls
    between 20 and 40 minutes:

    >>> round(probability_wait_time_between(a=0, b=60, low=20, high=40), 4)
    0.3333
    """
    return float(_dist(a, b).cdf(high) - _dist(a, b).cdf(low))


def expected_wait_time(a: float, b: float) -> float:
    """What's the average wait time in a uniform service window?

    Expected value of a uniform distribution, given range [`a`, `b`] --
    always the midpoint.

    Example
    -------
    Wait times are uniform across a 30-minute window:

    >>> round(expected_wait_time(a=0, b=30), 4)
    15.0
    """
    return float(_dist(a, b).mean())


def median_processing_time(a: float, b: float) -> float:
    """What's the median processing time in a uniform range?

    The 50th-percentile value of a uniform distribution, given range
    [`a`, `b`] -- same as the mean for a uniform distribution, but
    expressed via the inverse CDF.

    Example
    -------
    Processing time is uniform between 2 and 10 minutes:

    >>> round(median_processing_time(a=2, b=10), 4)
    6.0
    """
    return float(_dist(a, b).ppf(0.5))


def percentile_value(a: float, b: float, percentile: float) -> float:
    """What value marks a given percentile of a uniform range?

    The value below which `percentile` of the distribution falls, given
    range [`a`, `b`]. The inverse of the CDF.

    Example
    -------
    A value is uniform between 0 and 100. The 90th percentile value:

    >>> round(percentile_value(a=0, b=100, percentile=0.9), 4)
    90.0
    """
    return float(_dist(a, b).ppf(percentile))


def probability_random_number_in_range(a: float, b: float, low: float, high: float) -> float:
    """What's the chance a randomly generated number falls in a sub-range?

    Probability a uniformly distributed random number falls in
    [`low`, `high`], given the generator's range [`a`, `b`].

    Example
    -------
    A random number generator draws uniformly between 1 and 10. Chance
    the draw falls between 3 and 7:

    >>> round(probability_random_number_in_range(a=1, b=10, low=3, high=7), 4)
    0.4444
    """
    return float(_dist(a, b).cdf(high) - _dist(a, b).cdf(low))


def variance_of_uniform_process(a: float, b: float) -> float:
    """How spread out is a uniform process's outcomes?

    Variance of a uniform distribution, given range [`a`, `b`].

    Example
    -------
    A process is uniform between 0 and 10:

    >>> round(variance_of_uniform_process(a=0, b=10), 4)
    8.3333
    """
    return float(_dist(a, b).var())


def probability_delivery_time_exceeds(a: float, b: float, x: float) -> float:
    """What's the chance a delivery takes longer than x days?

    Probability a uniformly distributed delivery time exceeds `x` days,
    given range [`a`, `b`].

    Example
    -------
    Delivery time is uniform between 1 and 5 days. Chance delivery
    takes longer than 4 days:

    >>> round(probability_delivery_time_exceeds(a=1, b=5, x=4), 4)
    0.25
    """
    return float(_dist(a, b).sf(x))


def probability_call_answered_within(a: float, b: float, x: float) -> float:
    """What's the chance a call gets answered within x seconds?

    Probability a uniformly distributed answer time falls at or before
    `x` seconds, given range [`a`, `b`].

    Example
    -------
    Answer time is uniform between 0 and 20 seconds. Chance a call is
    answered within 5 seconds:

    >>> round(probability_call_answered_within(a=0, b=20, x=5), 4)
    0.25
    """
    return float(_dist(a, b).cdf(x))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
