"""Ten real-world "single yes/no trial" questions modeled with the Bernoulli distribution.

Each function wraps `scipy.stats.bernoulli` around one concrete question.
`p` is always the probability of the "success" (1) outcome for that single
trial.
"""

from scipy.stats import bernoulli


def chance_single_claim(p: float) -> float:
    """Will a given policyholder file a claim this year?

    Probability the single trial comes up "claim" (1), given claim
    probability `p`.

    Example
    -------
    A policyholder has a 12% chance of filing a claim this year:

    >>> round(chance_single_claim(p=0.12), 4)
    0.12
    """
    return float(bernoulli.pmf(1, p))


def chance_no_claim(p: float) -> float:
    """Will a given policyholder NOT file a claim this year?

    Probability the single trial comes up "no claim" (0), given claim
    probability `p`.

    Example
    -------
    A policyholder has a 12% chance of filing a claim this year. Chance
    of no claim:

    >>> round(chance_no_claim(p=0.12), 4)
    0.88
    """
    return float(bernoulli.pmf(0, p))


def chance_applicant_approved(p: float) -> float:
    """Will a given applicant be approved for coverage?

    Probability of approval, given approval probability `p`.

    Example
    -------
    An applicant has a 75% chance of approval:

    >>> round(chance_applicant_approved(p=0.75), 4)
    0.75
    """
    return float(bernoulli.pmf(1, p))


def chance_policy_lapses(p: float) -> float:
    """Will a given policy lapse this year?

    Probability of lapse, given lapse probability `p`.

    Example
    -------
    A policy has an 8% chance of lapsing this year:

    >>> round(chance_policy_lapses(p=0.08), 4)
    0.08
    """
    return float(bernoulli.pmf(1, p))


def chance_policy_renews(lapse_p: float) -> float:
    """Will a given policy renew (not lapse) this year?

    Probability of renewal -- the complement of lapsing, given lapse
    probability `lapse_p`.

    Example
    -------
    A policy has an 8% chance of lapsing. Chance it renews instead:

    >>> round(chance_policy_renews(lapse_p=0.08), 4)
    0.92
    """
    return float(bernoulli.pmf(0, lapse_p))


def expected_claim_indicator(p: float) -> float:
    """What's the expected value of the claim indicator for one policyholder?

    The mean of the 0/1 claim indicator, given claim probability `p` --
    this is just `p` itself, but it's the building block behind
    `expected_claims_in_block` in binomial.py (summing n of these).

    Example
    -------
    >>> round(expected_claim_indicator(p=0.12), 4)
    0.12
    """
    return float(bernoulli(p).mean())


def variance_of_claim_indicator(p: float) -> float:
    """How much does a single policyholder's claim indicator vary?

    Variance of the 0/1 claim indicator, given claim probability `p`.
    Useful as the per-policy risk unit before aggregating across a block.

    Example
    -------
    >>> round(variance_of_claim_indicator(p=0.12), 4)
    0.1056
    """
    return float(bernoulli(p).var())


def std_of_claim_indicator(p: float) -> float:
    """What's the standard deviation of a single policyholder's claim indicator?

    Standard deviation of the 0/1 claim indicator, given claim
    probability `p`.

    Example
    -------
    >>> round(std_of_claim_indicator(p=0.12), 4)
    0.325
    """
    return float(bernoulli(p).std())


def chance_marketing_email_opened(p: float) -> float:
    """Will a given recipient open a marketing email?

    Probability the email gets opened, given open-rate `p`.

    Example
    -------
    An email campaign has a 22% open rate:

    >>> round(chance_marketing_email_opened(p=0.22), 4)
    0.22
    """
    return float(bernoulli.pmf(1, p))


def chance_component_defective(p: float) -> float:
    """Is a given manufactured component defective?

    Probability the component is defective, given defect rate `p`.

    Example
    -------
    A component has a 3% defect rate:

    >>> round(chance_component_defective(p=0.03), 4)
    0.03
    """
    return float(bernoulli.pmf(1, p))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
