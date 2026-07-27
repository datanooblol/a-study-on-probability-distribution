"""Ten real-world "how do n outcomes split across categories?" questions modeled with the Multinomial distribution.

Each function wraps `scipy.stats.multinomial` around one concrete
question. `n` is the total number of trials, `p` a list of
category probabilities (must sum to 1), and `counts` a same-length list
of how many trials landed in each category.
"""

from scipy.stats import multinomial


def probability_claim_cause_split(n: int, p: list, counts: list) -> float:
    """Out of n claims, what's the chance they split this way across causes?

    Probability of the exact `counts` split across claim-cause
    categories (e.g. fire/theft/flood), given `n` total claims and
    per-category probabilities `p`.

    Example
    -------
    10 claims, causes split 50% fire / 30% theft / 20% flood. Chance of
    exactly 5 fire, 3 theft, 2 flood:

    >>> round(probability_claim_cause_split(n=10, p=[.5, .3, .2], counts=[5, 3, 2]), 4)
    0.085
    """
    return float(multinomial.pmf(counts, n, p))


def probability_policy_type_split(n: int, p: list, counts: list) -> float:
    """Out of n new policies sold, what's the chance they split this way by type?

    Probability of the exact `counts` split across policy types (e.g.
    term/whole/universal life), given `n` total sales and per-type
    probabilities `p`.

    Example
    -------
    12 policies sold, types split 40% term / 35% whole / 25% universal.
    Chance of exactly 5 term, 4 whole, 3 universal:

    >>> round(probability_policy_type_split(n=12, p=[.4, .35, .25], counts=[5, 4, 3]), 4)
    0.0666
    """
    return float(multinomial.pmf(counts, n, p))


def probability_grade_distribution(n: int, p: list, counts: list) -> float:
    """Out of n applicants, what's the chance they split this way by risk grade?

    Probability of the exact `counts` split across risk grades (e.g.
    preferred/standard/substandard), given `n` applicants and
    per-grade probabilities `p`.

    Example
    -------
    8 applicants, grades split 60% preferred / 30% standard / 10%
    substandard. Chance of exactly 5 preferred, 2 standard, 1
    substandard:

    >>> round(probability_grade_distribution(n=8, p=[.6, .3, .1], counts=[5, 2, 1]), 4)
    0.1176
    """
    return float(multinomial.pmf(counts, n, p))


def probability_channel_split(n: int, p: list, counts: list) -> float:
    """Out of n sales, what's the chance they split this way by channel?

    Probability of the exact `counts` split across sales channels (e.g.
    agent/online/broker), given `n` total sales and per-channel
    probabilities `p`.

    Example
    -------
    9 sales, channels split 50% agent / 30% online / 20% broker. Chance
    of exactly 4 agent, 3 online, 2 broker:

    >>> round(probability_channel_split(n=9, p=[.5, .3, .2], counts=[4, 3, 2]), 4)
    0.085
    """
    return float(multinomial.pmf(counts, n, p))


def probability_claim_outcome_split(n: int, p: list, counts: list) -> float:
    """Out of n claims processed, what's the chance they split this way by outcome?

    Probability of the exact `counts` split across claim outcomes (e.g.
    approved/denied/pending), given `n` claims and per-outcome
    probabilities `p`.

    Example
    -------
    10 claims, outcomes split 70% approved / 20% denied / 10% pending.
    Chance of exactly 7 approved, 2 denied, 1 pending:

    >>> round(probability_claim_outcome_split(n=10, p=[.7, .2, .1], counts=[7, 2, 1]), 4)
    0.1186
    """
    return float(multinomial.pmf(counts, n, p))


def probability_customer_segment_split(n: int, p: list, counts: list) -> float:
    """Out of n new customers, what's the chance they split this way by segment?

    Probability of the exact `counts` split across customer segments,
    given `n` new customers and per-segment probabilities `p`.

    Example
    -------
    6 new customers, segments split roughly evenly (34%/33%/33%).
    Chance of exactly 2 in each segment:

    >>> round(probability_customer_segment_split(n=6, p=[.34, .33, .33], counts=[2, 2, 2]), 4)
    0.1234
    """
    return float(multinomial.pmf(counts, n, p))


def probability_underwriting_decision_split(n: int, p: list, counts: list) -> float:
    """Out of n applications, what's the chance they split this way by decision?

    Probability of the exact `counts` split across underwriting
    decisions (e.g. accept/rate-up/decline), given `n` applications and
    per-decision probabilities `p`.

    Example
    -------
    15 applications, decisions split 60% accept / 25% rate-up / 15%
    decline. Chance of exactly 9 accept, 4 rate-up, 2 decline:

    >>> round(probability_underwriting_decision_split(n=15, p=[.6, .25, .15], counts=[9, 4, 2]), 4)
    0.0665
    """
    return float(multinomial.pmf(counts, n, p))


def probability_dice_roll_outcome(n: int, p: list, counts: list) -> float:
    """Out of n rolls of a fair die, what's the chance each face comes up this many times?

    Probability of the exact `counts` split across the 6 faces, given
    `n` rolls and per-face probabilities `p` (the classic textbook
    multinomial example).

    Example
    -------
    A fair die rolled 6 times. Chance each face (1-6) comes up exactly
    once:

    >>> round(probability_dice_roll_outcome(n=6, p=[1 / 6] * 6, counts=[1] * 6), 4)
    0.0154
    """
    return float(multinomial.pmf(counts, n, p))


def probability_survey_response_split(n: int, p: list, counts: list) -> float:
    """Out of n survey responses, what's the chance they split this way by sentiment?

    Probability of the exact `counts` split across response categories
    (e.g. satisfied/neutral/dissatisfied), given `n` responses and
    per-category probabilities `p`.

    Example
    -------
    7 survey responses, sentiment split 40% satisfied / 40% neutral /
    20% dissatisfied. Chance of exactly 3 satisfied, 3 neutral, 1
    dissatisfied:

    >>> round(probability_survey_response_split(n=7, p=[.4, .4, .2], counts=[3, 3, 1]), 4)
    0.1147
    """
    return float(multinomial.pmf(counts, n, p))


def probability_lapse_reason_split(n: int, p: list, counts: list) -> float:
    """Out of n lapsed policies, what's the chance they split this way by reason?

    Probability of the exact `counts` split across lapse reasons (e.g.
    nonpayment/replaced/matured/other), given `n` lapses and
    per-reason probabilities `p`.

    Example
    -------
    8 lapses, reasons split 40% nonpayment / 30% replaced / 20% matured
    / 10% other. Chance of exactly 3 nonpayment, 3 replaced, 1 matured,
    1 other:

    >>> round(probability_lapse_reason_split(n=8, p=[.4, .3, .2, .1], counts=[3, 3, 1, 1]), 4)
    0.0387
    """
    return float(multinomial.pmf(counts, n, p))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
