"""Ten real-world "how many events?" questions modeled with the Poisson distribution.

Each function wraps `scipy.stats.poisson` around one concrete question. `rate`
is always the expected number of events (lambda) over the same window the
question asks about.
"""

from scipy.stats import poisson


def claims_this_month(rate: float, k: int) -> float:
    """How many claims will a block of policies generate this month?

    Probability of exactly `k` claims this month, given an average of
    `rate` claims/month.

    Example
    -------
    A block of 10,000 policies averages 8 claims/month. Chance of exactly
    8 claims this month:

    >>> round(claims_this_month(rate=8, k=8), 4)
    0.1396
    """
    return float(poisson.pmf(k, rate))


def deaths_in_portfolio(rate: float, k: int) -> float:
    """How many deaths will occur in a pension annuitant portfolio this year?

    Probability of at most `k` deaths this year, given an expected
    `rate` deaths/year.

    Example
    -------
    A portfolio expects 5 deaths/year. Chance of at most 3 deaths this year:

    >>> round(deaths_in_portfolio(rate=5, k=3), 4)
    0.265
    """
    return float(poisson.cdf(k, rate))


def support_calls_this_hour(rate: float, k: int) -> float:
    """How many customer support calls arrive in a 1-hour window?

    Probability of exactly `k` calls in the hour, given an average of
    `rate` calls/hour.

    Example
    -------
    A call center averages 12 calls/hour. Chance of exactly 15 calls in
    the next hour:

    >>> round(support_calls_this_hour(rate=12, k=15), 4)
    0.0724
    """
    return float(poisson.pmf(k, rate))


def server_crashes_this_quarter(rate: float, k: int) -> float:
    """How many server crashes happen in a given quarter?

    Probability of at most `k` crashes this quarter, given an average of
    `rate` crashes/quarter.

    Example
    -------
    A critical server averages 2 crashes/quarter. Chance of at most 1
    crash this quarter:

    >>> round(server_crashes_this_quarter(rate=2, k=1), 4)
    0.406
    """
    return float(poisson.cdf(k, rate))


def typos_in_document(rate: float, k: int) -> float:
    """How many typos are in a 10-page document?

    Probability of exactly `k` typos, given an average of `rate` typos
    per document of that length.

    Example
    -------
    A 10-page document averages 3 typos. Chance the document is typo-free:

    >>> round(typos_in_document(rate=3, k=0), 4)
    0.0498
    """
    return float(poisson.pmf(k, rate))


def earthquakes_this_decade(rate: float, k: int) -> float:
    """How many earthquakes above magnitude 5 hit a region in a decade?

    Probability of at least `k` earthquakes this decade, given an
    average of `rate` earthquakes/decade.

    Example
    -------
    A region averages 4 magnitude-5+ earthquakes/decade. Chance of at
    least 6 this decade:

    >>> round(earthquakes_this_decade(rate=4, k=6), 4)
    0.2149
    """
    return float(poisson.sf(k - 1, rate))


def branch_visitors_this_hour(rate: float, k: int) -> float:
    """How many customers walk into a branch between 9am-10am?

    Probability of exactly `k` visitors in the hour, given an average of
    `rate` visitors/hour.

    Example
    -------
    A branch averages 20 visitors in the 9-10am hour. Chance of exactly
    25 visitors tomorrow:

    >>> round(branch_visitors_this_hour(rate=20, k=25), 4)
    0.0446
    """
    return float(poisson.pmf(k, rate))


def catastrophic_losses_this_year(rate: float, k: int) -> float:
    """How many large (catastrophic) losses does a reinsurer pay out this year?

    Probability of exactly `k` catastrophic losses this year, given an
    average of `rate` such losses/year.

    Example
    -------
    A reinsurer averages 1.5 catastrophic losses/year. Chance of zero
    catastrophic losses this year:

    >>> round(catastrophic_losses_this_year(rate=1.5, k=0), 4)
    0.2231
    """
    return float(poisson.pmf(k, rate))


def defects_in_batch(rate: float, k: int) -> float:
    """How many defective items appear in a batch of 1,000 manufactured parts?

    Probability of at most `k` defects in the batch, given an average of
    `rate` defects/batch.

    Example
    -------
    A batch averages 5 defects. Chance of at most 10 defects in the
    next batch:

    >>> round(defects_in_batch(rate=5, k=10), 4)
    0.9863
    """
    return float(poisson.cdf(k, rate))


def fraud_flags_this_week(rate: float, k: int) -> float:
    """How many fraudulent claims get flagged in a given week?

    Probability of at least `k` flagged claims this week, given an
    average of `rate` flagged claims/week.

    Example
    -------
    A fraud team averages 2 flagged claims/week. Chance of at least 5
    flagged claims this week:

    >>> round(fraud_flags_this_week(rate=2, k=5), 4)
    0.0527
    """
    return float(poisson.sf(k - 1, rate))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
