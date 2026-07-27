"""Ten real-world "bell curve" questions modeled with the Normal distribution.

Each function wraps `scipy.stats.norm` around one concrete question.
`mu` is the mean, `sigma` the standard deviation.
"""

from scipy.stats import norm


def probability_claim_amount_between(mu: float, sigma: float, low: float, high: float) -> float:
    """What's the chance a claim amount falls between two dollar values?

    Probability a normally distributed claim amount falls in
    [`low`, `high`], given mean `mu` and std `sigma`.

    Example
    -------
    Claim amounts average $5,000 with a $1,200 std. Chance a claim
    falls between $4,000 and $6,000:

    >>> round(probability_claim_amount_between(mu=5000, sigma=1200, low=4000, high=6000), 4)
    0.5953
    """
    return float(norm.cdf(high, mu, sigma) - norm.cdf(low, mu, sigma))


def probability_claim_amount_exceeds(mu: float, sigma: float, x: float) -> float:
    """What's the chance a claim amount exceeds x?

    Probability a normally distributed claim amount exceeds `x`, given
    mean `mu` and std `sigma`.

    Example
    -------
    Claim amounts average $5,000 with a $1,200 std. Chance a claim
    exceeds $8,000:

    >>> round(probability_claim_amount_exceeds(mu=5000, sigma=1200, x=8000), 4)
    0.0062
    """
    return float(norm.sf(x, mu, sigma))


def probability_test_score_below(mu: float, sigma: float, x: float) -> float:
    """What's the chance a test score falls below x?

    Probability a normally distributed score falls below `x`, given
    mean `mu` and std `sigma`.

    Example
    -------
    Test scores average 70 with a std of 10. Chance a score falls
    below 60:

    >>> round(probability_test_score_below(mu=70, sigma=10, x=60), 4)
    0.1587
    """
    return float(norm.cdf(x, mu, sigma))


def percentile_rank_of_claim(mu: float, sigma: float, x: float) -> float:
    """What percentile does a given claim amount fall at?

    The fraction of the distribution at or below `x`, given mean `mu`
    and std `sigma` -- i.e. the CDF read as a percentile rank.

    Example
    -------
    Claim amounts average $5,000 with a $1,200 std. Percentile rank of
    a $6,500 claim:

    >>> round(percentile_rank_of_claim(mu=5000, sigma=1200, x=6500), 4)
    0.8944
    """
    return float(norm.cdf(x, mu, sigma))


def value_at_percentile(mu: float, sigma: float, percentile: float) -> float:
    """What claim amount marks a given percentile (e.g. the 90th)?

    The value below which `percentile` of the distribution falls, given
    mean `mu` and std `sigma`. The inverse of the CDF.

    Example
    -------
    Claim amounts average $5,000 with a $1,200 std. The 90th percentile
    claim amount:

    >>> round(value_at_percentile(mu=5000, sigma=1200, percentile=0.9), 4)
    6537.8619
    """
    return float(norm.ppf(percentile, mu, sigma))


def probability_within_k_std(mu: float, sigma: float, k: float) -> float:
    """What's the chance a value falls within k standard deviations of the mean?

    Probability of falling in [mu - k*sigma, mu + k*sigma] -- the
    empirical rule (68/95/99.7) generalized to any `k`.

    Example
    -------
    A distribution with mean 100 and std 15. Chance of falling within
    2 standard deviations (i.e. 70 to 130):

    >>> round(probability_within_k_std(mu=100, sigma=15, k=2), 4)
    0.9545
    """
    return float(norm.cdf(mu + k * sigma, mu, sigma) - norm.cdf(mu - k * sigma, mu, sigma))


def probability_processing_time_exceeds(mu: float, sigma: float, x: float) -> float:
    """What's the chance a claim takes longer than x days to process?

    Probability a normally distributed processing time exceeds `x`
    days, given mean `mu` and std `sigma`.

    Example
    -------
    Claims take 5 days on average to process, with a 1.5-day std.
    Chance processing takes longer than 8 days:

    >>> round(probability_processing_time_exceeds(mu=5, sigma=1.5, x=8), 4)
    0.0228
    """
    return float(norm.sf(x, mu, sigma))


def probability_bmi_in_range(mu: float, sigma: float, low: float, high: float) -> float:
    """What's the chance an applicant's BMI falls in a given underwriting range?

    Probability a normally distributed BMI falls in [`low`, `high`],
    given mean `mu` and std `sigma`.

    Example
    -------
    Applicant BMI averages 25 with a std of 4. Chance BMI falls between
    18.5 and 25 (normal weight range):

    >>> round(probability_bmi_in_range(mu=25, sigma=4, low=18.5, high=25), 4)
    0.4479
    """
    return float(norm.cdf(high, mu, sigma) - norm.cdf(low, mu, sigma))


def probability_portfolio_return_below(mu: float, sigma: float, x: float) -> float:
    """What's the chance an investment portfolio's return falls below x?

    Probability a normally distributed return falls below `x`, given
    mean `mu` and std `sigma`.

    Example
    -------
    A portfolio averages a 7% return with a 15% std. Chance the return
    is below 0% (a loss):

    >>> round(probability_portfolio_return_below(mu=0.07, sigma=0.15, x=0), 4)
    0.3204
    """
    return float(norm.cdf(x, mu, sigma))


def z_score_of_value(mu: float, sigma: float, x: float) -> float:
    """How many standard deviations away from the mean does a value sit?

    The z-score of `x`, given mean `mu` and std `sigma` -- the
    standardized value used to look up probabilities on a standard
    normal table, or to feed into `norm(0, 1)` directly.

    Example
    -------
    A distribution with mean 100 and std 15. Z-score of a value of 130:

    >>> round(z_score_of_value(mu=100, sigma=15, x=130), 4)
    2.0
    """
    return (x - mu) / sigma


if __name__ == "__main__":
    import doctest

    doctest.testmod()
