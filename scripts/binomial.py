"""Ten real-world "how many successes out of n trials?" questions modeled with the Binomial distribution.

Each function wraps `scipy.stats.binom` around one concrete question.
`n` is the number of independent trials, `p` the per-trial success
probability.
"""

from scipy.stats import binom


def claims_exactly_k_in_block(n: int, p: float, k: int) -> float:
    """Out of n policies, how many file exactly k claims this year?

    Probability of exactly `k` claims out of `n` independent policies,
    each with claim probability `p`.

    Example
    -------
    A block of 50 policies, each with a 10% claim probability. Chance
    exactly 5 file claims this year:

    >>> round(claims_exactly_k_in_block(n=50, p=0.1, k=5), 4)
    0.1849
    """
    return float(binom.pmf(k, n, p))


def claims_at_most_k_in_block(n: int, p: float, k: int) -> float:
    """Out of n policies, what's the chance at most k file claims this year?

    Probability of at most `k` claims out of `n` policies, each with
    claim probability `p`.

    Example
    -------
    A block of 50 policies, each with a 10% claim probability. Chance
    at most 8 file claims:

    >>> round(claims_at_most_k_in_block(n=50, p=0.1, k=8), 4)
    0.9421
    """
    return float(binom.cdf(k, n, p))


def claims_at_least_k_in_block(n: int, p: float, k: int) -> float:
    """Out of n policies, what's the chance at least k file claims this year?

    Probability of at least `k` claims out of `n` policies, each with
    claim probability `p`.

    Example
    -------
    A block of 50 policies, each with a 10% claim probability. Chance
    at least 10 file claims:

    >>> round(claims_at_least_k_in_block(n=50, p=0.1, k=10), 4)
    0.0245
    """
    return float(binom.sf(k - 1, n, p))


def expected_claims_in_block(n: int, p: float) -> float:
    """How many claims should a block of n policies generate on average?

    Expected number of claims out of `n` policies, each with claim
    probability `p`.

    Example
    -------
    A block of 50 policies, each with a 10% claim probability. Expected
    claims this year:

    >>> round(expected_claims_in_block(n=50, p=0.1), 4)
    5.0
    """
    return float(binom(n, p).mean())


def approvals_exactly_k(n: int, p: float, k: int) -> float:
    """Out of n applicants, how many get approved for exactly k of them?

    Probability of exactly `k` approvals out of `n` applicants, each
    with approval probability `p`.

    Example
    -------
    20 applicants, each with a 80% approval chance. Chance exactly 18
    get approved:

    >>> round(approvals_exactly_k(n=20, p=0.8, k=18), 4)
    0.1369
    """
    return float(binom.pmf(k, n, p))


def defective_items_at_most_k(n: int, p: float, k: int) -> float:
    """In a batch of n items, what's the chance at most k are defective?

    Probability of at most `k` defects out of `n` manufactured items,
    each with defect probability `p`.

    Example
    -------
    A batch of 100 items, each with a 2% defect rate. Chance at most 3
    are defective:

    >>> round(defective_items_at_most_k(n=100, p=0.02, k=3), 4)
    0.859
    """
    return float(binom.cdf(k, n, p))


def sales_conversions_exactly_k(n: int, p: float, k: int) -> float:
    """Out of n sales calls, what's the chance exactly k convert?

    Probability of exactly `k` conversions out of `n` calls, each with
    conversion probability `p`.

    Example
    -------
    15 sales calls, each with a 30% conversion rate. Chance exactly 5
    convert:

    >>> round(sales_conversions_exactly_k(n=15, p=0.3, k=5), 4)
    0.2061
    """
    return float(binom.pmf(k, n, p))


def deaths_at_most_k_in_group(n: int, p: float, k: int) -> float:
    """In a cohort of n policyholders, what's the chance at most k die this year?

    Probability of at most `k` deaths out of `n` policyholders, each
    with mortality probability `p`.

    Example
    -------
    A cohort of 200 policyholders, each with a 1% mortality rate. Chance
    at most 3 die this year:

    >>> round(deaths_at_most_k_in_group(n=200, p=0.01, k=3), 4)
    0.858
    """
    return float(binom.cdf(k, n, p))


def fraud_cases_at_least_k(n: int, p: float, k: int) -> float:
    """Out of n claims, what's the chance at least k are fraudulent?

    Probability of at least `k` fraudulent claims out of `n` claims,
    each with fraud probability `p`.

    Example
    -------
    500 claims, each with a 0.5% fraud rate. Chance at least 5 are
    fraudulent:

    >>> round(fraud_cases_at_least_k(n=500, p=0.005, k=5), 4)
    0.1083
    """
    return float(binom.sf(k - 1, n, p))


def survey_responses_exactly_k(n: int, p: float, k: int) -> float:
    """Out of n survey invitations, what's the chance exactly k respond?

    Probability of exactly `k` responses out of `n` invitations, each
    with response probability `p`.

    Example
    -------
    40 survey invitations, each with a 25% response rate. Chance exactly
    10 respond:

    >>> round(survey_responses_exactly_k(n=40, p=0.25, k=10), 4)
    0.1444
    """
    return float(binom.pmf(k, n, p))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
