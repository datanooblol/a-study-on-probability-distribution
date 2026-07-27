"""Ten real-world "skewed positive-value" questions modeled with the Log-normal distribution.

Each function wraps `scipy.stats.lognorm` around one concrete question.
`mu` and `sigma` are the mean and standard deviation of the underlying
*normal* distribution of ln(X) -- not of X itself. scipy parameterizes
this as `lognorm(s=sigma, scale=exp(mu))`.
"""

import numpy as np
from scipy.stats import lognorm


def _dist(mu: float, sigma: float):
    return lognorm(s=sigma, scale=np.exp(mu))


def probability_claim_severity_exceeds(mu: float, sigma: float, x: float) -> float:
    """What's the chance a claim's severity (dollar size) exceeds x?

    Probability a log-normally distributed claim severity exceeds `x`,
    given log-mean `mu` and log-std `sigma`.

    Example
    -------
    Claim severity has log-mean 8.5 and log-std 1.0. Chance a claim
    exceeds $10,000:

    >>> round(probability_claim_severity_exceeds(mu=8.5, sigma=1.0, x=10000), 4)
    0.2387
    """
    return float(_dist(mu, sigma).sf(x))


def probability_claim_severity_within(mu: float, sigma: float, x: float) -> float:
    """What's the chance a claim's severity stays within x?

    Probability a log-normally distributed claim severity is at most
    `x`, given log-mean `mu` and log-std `sigma`.

    Example
    -------
    Claim severity has log-mean 8.5 and log-std 1.0. Chance a claim
    stays at or below $5,000:

    >>> round(probability_claim_severity_within(mu=8.5, sigma=1.0, x=5000), 4)
    0.5069
    """
    return float(_dist(mu, sigma).cdf(x))


def median_claim_severity(mu: float, sigma: float) -> float:
    """What's the typical (median) claim severity?

    The median of a log-normal distribution is exp(mu) -- unlike the
    mean, it isn't inflated by the distribution's right skew.

    Example
    -------
    Claim severity has log-mean 8.5 and log-std 1.0:

    >>> round(median_claim_severity(mu=8.5, sigma=1.0), 4)
    4914.7688
    """
    return float(_dist(mu, sigma).median())


def probability_income_exceeds(mu: float, sigma: float, x: float) -> float:
    """What's the chance a policyholder's income exceeds x?

    Probability a log-normally distributed income exceeds `x`, given
    log-mean `mu` and log-std `sigma`.

    Example
    -------
    Income has log-mean 10.5 and log-std 0.6. Chance income exceeds
    $50,000:

    >>> round(probability_income_exceeds(mu=10.5, sigma=0.6, x=50000), 4)
    0.297
    """
    return float(_dist(mu, sigma).sf(x))


def probability_asset_price_below(mu: float, sigma: float, x: float) -> float:
    """What's the chance an asset's price stays below x?

    Probability a log-normally distributed asset price is below `x`,
    given log-mean `mu` and log-std `sigma`.

    Example
    -------
    An asset price has log-mean 4.0 and log-std 0.3. Chance the price
    stays below $60:

    >>> round(probability_asset_price_below(mu=4.0, sigma=0.3, x=60), 4)
    0.6234
    """
    return float(_dist(mu, sigma).cdf(x))


def expected_claim_severity(mu: float, sigma: float) -> float:
    """What's the average claim severity used for pricing?

    The mean of a log-normal distribution, exp(mu + sigma**2 / 2) --
    always larger than the median because of the right skew, which is
    why pricing off the mean (not the median) matters for reserving.

    Example
    -------
    Claim severity has log-mean 8.5 and log-std 1.0:

    >>> round(expected_claim_severity(mu=8.5, sigma=1.0), 4)
    8103.0839
    """
    return float(_dist(mu, sigma).mean())


def probability_hospital_bill_exceeds_threshold(mu: float, sigma: float, x: float) -> float:
    """What's the chance a hospital bill exceeds a given threshold?

    Probability a log-normally distributed hospital bill exceeds `x`,
    given log-mean `mu` and log-std `sigma`.

    Example
    -------
    Hospital bills have log-mean 9.0 and log-std 0.8. Chance a bill
    exceeds $20,000:

    >>> round(probability_hospital_bill_exceeds_threshold(mu=9.0, sigma=0.8, x=20000), 4)
    0.1294
    """
    return float(_dist(mu, sigma).sf(x))


def value_at_risk_claim_severity(mu: float, sigma: float, percentile: float) -> float:
    """What claim severity marks the 95th percentile (a VaR-style threshold)?

    The value below which `percentile` of claim severities fall, given
    log-mean `mu` and log-std `sigma`. The inverse of the CDF.

    Example
    -------
    Claim severity has log-mean 8.5 and log-std 1.0. The 95th
    percentile severity:

    >>> round(value_at_risk_claim_severity(mu=8.5, sigma=1.0, percentile=0.95), 4)
    25459.7392
    """
    return float(_dist(mu, sigma).ppf(percentile))


def probability_repair_time_exceeds(mu: float, sigma: float, x: float) -> float:
    """What's the chance a repair job takes longer than x hours?

    Probability a log-normally distributed repair duration exceeds `x`
    hours, given log-mean `mu` and log-std `sigma`.

    Example
    -------
    Repair time has log-mean 1.5 and log-std 0.5. Chance a repair takes
    longer than 10 hours:

    >>> round(probability_repair_time_exceeds(mu=1.5, sigma=0.5, x=10), 4)
    0.0542
    """
    return float(_dist(mu, sigma).sf(x))


def std_of_claim_severity(mu: float, sigma: float) -> float:
    """How spread out are claim severities in dollar terms?

    Standard deviation of a log-normal distribution, given log-mean
    `mu` and log-std `sigma`. Note this is in dollars, not log-dollars.

    Example
    -------
    Claim severity has log-mean 8.5 and log-std 1.0:

    >>> round(std_of_claim_severity(mu=8.5, sigma=1.0), 4)
    10621.7857
    """
    return float(_dist(mu, sigma).std())


if __name__ == "__main__":
    import doctest

    doctest.testmod()
