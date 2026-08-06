"""Hand-written metadata for the 25 core ('Tier 1') teaching distributions."""

TIER1_METADATA = {
    # ---------------------------------------------------------------- discrete
    "bernoulli": {
        "display": "Bernoulli",
        "params": [{"name": "p", "default": 0.3, "desc": "probability of success (the golden gumball)"}],
        "formula": r"P(X=k) = p^{k}(1-p)^{1-k}, \quad k \in \{0, 1\}",
        "intuition": (
            "The atom of probability: one single yes/no trial. Everything else in the discrete "
            "family -- Binomial, Geometric, Negative Binomial, Multinomial -- is Bernoulli repeated, "
            "counted, or generalized. If a question has exactly two outcomes and one fixed chance of "
            "'success', it's a Bernoulli trial."
        ),
        "kid_story": (
            "There's a magic gumball machine. Every time you put in a coin, there's a 30% chance you "
            "get a shiny golden gumball, and a 70% chance you get a plain one."
        ),
        "kid_question": "If you put in exactly one coin, what's the chance you get the golden gumball?",
        "use_cases": [
            "Whether a single coin flip lands heads",
            "Whether one visitor clicks an ad",
            "Whether one manufactured part passes inspection",
            "Whether one email gets opened",
        ],
        "code": "dist = stats.bernoulli(p=0.3)\nprint('P(golden gumball):', round(dist.pmf(1), 4))\nprint('P(plain gumball):', round(dist.pmf(0), 4))",
    },
    "binom": {
        "display": "Binomial",
        "params": [
            {"name": "n", "default": 10, "desc": "number of independent trials"},
            {"name": "p", "default": 0.3, "desc": "probability of success on each trial"},
        ],
        "formula": r"P(X=k) = \binom{n}{k} p^{k}(1-p)^{n-k}, \quad k = 0,1,\ldots,n",
        "intuition": (
            "Binomial is just Bernoulli repeated n independent times and the successes summed up. "
            "Each trial doesn't care about the others -- same success chance p every time -- and we "
            "only care about the total count, not the order they happened in."
        ),
        "kid_story": (
            "Same magic gumball machine, 30% chance of gold each time. Today you have 10 coins, "
            "so you play 10 times in a row."
        ),
        "kid_question": "Out of your 10 tries, what's the chance you get exactly 3 golden gumballs?",
        "use_cases": [
            "Number of conversions out of n website visitors (A/B testing)",
            "Number of defective items in a batch of n",
            "Number of correct guesses out of n multiple-choice questions",
        ],
        "code": "dist = stats.binom(n=10, p=0.3)\nprint('P(exactly 3 gold):', round(dist.pmf(3), 4))\nprint('expected gold gumballs:', dist.mean())",
    },
    "geom": {
        "display": "Geometric",
        "params": [{"name": "p", "default": 0.3, "desc": "probability of success on each attempt"}],
        "formula": r"P(X=k) = (1-p)^{k-1} p, \quad k = 1, 2, 3, \ldots",
        "intuition": (
            "Instead of counting successes in a fixed number of Bernoulli trials (that's Binomial), "
            "Geometric flips the question: keep trying until the *first* success, and count how many "
            "tries that took. It's the 'how long do I have to wait' distribution."
        ),
        "kid_story": (
            "You really want that golden gumball (30% chance each try), so you keep feeding coins into "
            "the machine, one at a time, until you finally get one."
        ),
        "kid_question": "What's the chance your very first golden gumball comes on your 4th try, and how many tries do you expect to need on average?",
        "use_cases": [
            "Number of sales calls until the first sale",
            "Number of job applications until the first offer",
            "Number of spins until a slot machine pays out",
        ],
        "code": "dist = stats.geom(p=0.3)\nprint('P(first gold on try 4):', round(dist.pmf(4), 4))\nprint('expected tries needed:', round(dist.mean(), 4))",
    },
    "nbinom": {
        "display": "Negative Binomial",
        "params": [
            {"name": "n", "default": 5, "desc": "number of successes we're waiting for"},
            {"name": "p", "default": 0.4, "desc": "probability of success on each attempt"},
        ],
        "formula": r"P(X=k) = \binom{k+n-1}{k} p^{n}(1-p)^{k}, \quad k = 0, 1, 2, \ldots",
        "intuition": (
            "Geometric generalized: instead of stopping at the 1st success, keep going until the n-th "
            "success, and count the *failures* along the way. Geometric is the special case n=1. It's "
            "popular in real data because it handles more spread-out ('overdispersed') counts than Poisson."
        ),
        "kid_story": (
            "A kid is fishing at a pond and wants to catch exactly 5 fish before going home. Each cast "
            "has a 40% chance of catching one."
        ),
        "kid_question": "By the time they land their 5th fish, what's the chance they had exactly 3 empty casts along the way?",
        "use_cases": [
            "Insurance claim counts (more variable than Poisson allows)",
            "Number of failed manufacturing attempts before r good units",
            "Modeling overdispersed count data in ecology and biology",
        ],
        "code": "dist = stats.nbinom(n=5, p=0.4)\nprint('P(exactly 3 empty casts before 5th fish):', round(dist.pmf(3), 4))\nprint('expected empty casts:', round(dist.mean(), 4))",
    },
    "poisson": {
        "display": "Poisson",
        "params": [{"name": "mu", "default": 4, "desc": "average number of events in the interval (lambda)"}],
        "formula": r"P(X=k) = \frac{e^{-\mu}\mu^{k}}{k!}, \quad k = 0, 1, 2, \ldots",
        "intuition": (
            "The distribution of rare, independent events counted over a fixed window of time or space, "
            "when you only know the *average* rate. It's the limit of Binomial when n is huge, p is tiny, "
            "and n*p settles on a fixed average -- lots of chances, each individually unlikely."
        ),
        "kid_story": (
            "On a clear night, a kid lying in the backyard sees on average 4 shooting stars per hour."
        ),
        "kid_question": "What's the chance they see exactly 6 shooting stars in the next hour?",
        "use_cases": [
            "Customer arrivals per hour at a store",
            "Number of typos per page in a book",
            "Server requests per second, or website hits per minute",
        ],
        "code": "dist = stats.poisson(mu=4)\nprint('P(exactly 6 shooting stars):', round(dist.pmf(6), 4))\nprint('P(more than 6):', round(dist.sf(6), 4))",
    },
    "hypergeom": {
        "display": "Hypergeometric",
        "params": [
            {"name": "M", "default": 20, "desc": "total population size"},
            {"name": "n", "default": 7, "desc": "number of 'success' items in the population"},
            {"name": "N", "default": 10, "desc": "number of items drawn (sample size)"},
        ],
        "formula": r"P(X=k) = \frac{\binom{n}{k}\binom{M-n}{N-k}}{\binom{M}{N}}",
        "intuition": (
            "Binomial's sibling for sampling *without* replacement. Once you grab a candy from the jar, "
            "it's gone, so each draw changes the odds for the next one -- unlike Binomial, where every "
            "trial has the exact same probability."
        ),
        "kid_story": (
            "A jar has 20 candies, 7 of which are red, the rest are other colors. A kid grabs a big "
            "handful of 10 candies all at once, without looking."
        ),
        "kid_question": "What's the chance exactly 4 of the 10 candies they grabbed are red?",
        "use_cases": [
            "Quality control: sampling defective items from a finite batch without replacement",
            "Card game probabilities (drawing specific cards from a deck)",
            "Ecology: capture-recapture population estimates",
        ],
        "code": "dist = stats.hypergeom(M=20, n=7, N=10)\nprint('P(exactly 4 red candies):', round(dist.pmf(4), 4))\nprint('expected red candies:', round(dist.mean(), 4))",
    },
    "randint": {
        "display": "Discrete Uniform",
        "params": [
            {"name": "low", "default": 1, "desc": "smallest possible value (inclusive)"},
            {"name": "high", "default": 7, "desc": "one past the largest possible value"},
        ],
        "formula": r"P(X=k) = \frac{1}{\text{high} - \text{low}}, \quad k = \text{low}, \ldots, \text{high}-1",
        "intuition": (
            "Every whole number in a range is exactly equally likely -- no favorites. It's the discrete "
            "cousin of the continuous Uniform distribution, and the simplest possible 'fair' randomness."
        ),
        "kid_story": "Rolling one ordinary, perfectly fair six-sided die.",
        "kid_question": "What's the chance of rolling a 4, and what's the chance of rolling a 4 or higher?",
        "use_cases": [
            "Simulating fair dice or lottery draws",
            "Randomly sampling an index/ID from a fixed range",
            "Shuffling and permutation tests in statistics",
        ],
        "code": "dist = stats.randint(low=1, high=7)\nprint('P(roll a 4):', round(dist.pmf(4), 4))\nprint('P(roll 4 or higher):', round(dist.sf(3), 4))",
    },
    # -------------------------------------------------------------- continuous
    "uniform": {
        "display": "Uniform",
        "params": [
            {"name": "loc", "default": 0, "desc": "lower bound of the range"},
            {"name": "scale", "default": 10, "desc": "width of the range (upper bound = loc + scale)"},
        ],
        "formula": r"f(x) = \frac{1}{b-a}, \quad a \le x \le b",
        "intuition": (
            "Any value in a range is equally likely, and there are infinitely many possible values -- so "
            "unlike its discrete cousin, no single value has positive probability; only *intervals* do. "
            "It's also the universal building block: feeding Uniform samples through a distribution's "
            "inverse CDF is literally how computers generate random draws from any other distribution."
        ),
        "kid_story": "A spinner arrow can land anywhere at all between 0 and 10, with no spot favored over any other.",
        "kid_question": "What's the chance the arrow lands somewhere between 3 and 5?",
        "use_cases": [
            "The core random-number generator behind simulations",
            "Modeling an arrival time known only to fall within a window",
            "Bootstrapping and Monte Carlo methods",
        ],
        "code": "dist = stats.uniform(loc=0, scale=10)\nprint('P(3 <= X <= 5):', round(dist.cdf(5) - dist.cdf(3), 4))",
    },
    "norm": {
        "display": "Normal (Gaussian)",
        "params": [
            {"name": "loc", "default": 170, "desc": "mean (center of the bell curve)"},
            {"name": "scale", "default": 10, "desc": "standard deviation (spread)"},
        ],
        "formula": r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}",
        "intuition": (
            "The famous bell curve: values cluster near the mean and get rarer the further out you go, "
            "symmetrically in both directions. It shows up everywhere because of the Central Limit "
            "Theorem -- add up enough small, independent effects (genetics, measurement noise, coin "
            "flips) and the sum starts looking Normal, no matter what the individual pieces looked like."
        ),
        "kid_story": "Heights of all the 10-year-olds in a big school average around 170 cm, with a standard spread of 10 cm -- most kids are close to average, very few are extremely short or extremely tall.",
        "kid_question": "What fraction of kids are taller than 185 cm?",
        "use_cases": [
            "Measurement error in scientific instruments",
            "Standardized test scores (IQ, SAT)",
            "Approximating sums/averages of many small effects (Central Limit Theorem)",
        ],
        "code": "dist = stats.norm(loc=170, scale=10)\nprint('P(height > 185cm):', round(dist.sf(185), 4))",
    },
    "expon": {
        "display": "Exponential",
        "params": [{"name": "scale", "default": 5, "desc": "mean waiting time (1/rate)"}],
        "formula": r"f(x) = \frac{1}{\text{scale}} e^{-x/\text{scale}}, \quad x \ge 0",
        "intuition": (
            "The waiting time between events that happen randomly and independently at a constant "
            "average rate -- the continuous-time twin of the Poisson count. It's 'memoryless': no "
            "matter how long you've already waited, the time left to wait has the exact same "
            "distribution as if you'd just started."
        ),
        "kid_story": "Buses arrive randomly at a stop, on average once every 5 minutes.",
        "kid_question": "If you just arrived at the stop, what's the chance you'll wait more than 8 minutes for the next bus?",
        "use_cases": [
            "Time between customer arrivals or phone calls",
            "Lifetime of an electronic component with a constant failure rate",
            "Time between radioactive decay events",
        ],
        "code": "dist = stats.expon(scale=5)\nprint('P(wait > 8 min):', round(dist.sf(8), 4))\nprint('expected wait:', dist.mean())",
    },
    "gamma": {
        "display": "Gamma",
        "params": [
            {"name": "a", "default": 3, "desc": "shape -- how many exponential waits are being summed"},
            {"name": "scale", "default": 2, "desc": "scale of each underlying wait"},
        ],
        "formula": r"f(x) = \frac{x^{a-1} e^{-x/\text{scale}}}{\Gamma(a)\,\text{scale}^{a}}, \quad x \ge 0",
        "intuition": (
            "Chain several independent Exponential waits back to back and ask for the *total* time -- "
            "that sum follows a Gamma distribution. Shape parameter a is 'how many waits', so Gamma "
            "with a=1 is just Exponential itself."
        ),
        "kid_story": "An ice-cream truck passes by every 2 minutes on average (Exponential). A kid wants to know how long until the 3rd truck shows up.",
        "kid_question": "What's the chance the 3rd truck arrives within 4 minutes, and what's the expected total wait?",
        "use_cases": [
            "Insurance claim severity / total claim amount modeling",
            "Rainfall accumulation over a storm",
            "Modeling total waiting time across multiple sequential events",
        ],
        "code": "dist = stats.gamma(a=3, scale=2)\nprint('P(3rd truck within 4 min):', round(dist.cdf(4), 4))\nprint('expected wait for 3rd truck:', dist.mean())",
    },
    "beta": {
        "display": "Beta",
        "params": [
            {"name": "a", "default": 2, "desc": "pseudo-count of 'successes' (shapes the curve toward 1)"},
            {"name": "b", "default": 5, "desc": "pseudo-count of 'failures' (shapes the curve toward 0)"},
        ],
        "formula": r"f(x) = \frac{x^{a-1}(1-x)^{b-1}}{B(a,b)}, \quad 0 \le x \le 1",
        "intuition": (
            "The distribution *of a probability itself* -- always squeezed between 0 and 1, its shape "
            "controlled by two 'pseudo-counts' that pull the mass toward 1 or toward 0. It's the natural "
            "way to express uncertainty about a rate or proportion, and it's the mathematical partner "
            "('conjugate prior') of the Binomial distribution."
        ),
        "kid_story": "A kid is guessing how full a candy jar looks based on a few early scoops -- somewhere between completely empty (0) and completely full (1), but they're not sure exactly where.",
        "kid_question": "Given what they've seen so far (leaning toward 'not very full'), what's the chance the jar is judged to be more than 50% full?",
        "use_cases": [
            "Bayesian estimate of a conversion rate or click-through rate",
            "Modeling any quantity that's naturally a proportion (0 to 1)",
            "A/B testing: uncertainty about the true success probability",
        ],
        "code": "dist = stats.beta(a=2, b=5)\nprint('P(fullness > 0.5):', round(dist.sf(0.5), 4))\nprint('most likely fullness (mean):', round(dist.mean(), 4))",
    },
    "chi2": {
        "display": "Chi-square",
        "params": [{"name": "df", "default": 4, "desc": "degrees of freedom -- how many squared normals are summed"}],
        "formula": r"f(x) = \frac{x^{df/2 - 1} e^{-x/2}}{2^{df/2}\Gamma(df/2)}, \quad x \ge 0",
        "intuition": (
            "Take several independent standard-Normal 'wobbles', square each one (so they're all "
            "positive), and add them up -- the total follows a Chi-square distribution. It measures "
            "total squared deviation, which is exactly what's needed to test whether observed data "
            "matches an expected pattern."
        ),
        "kid_story": "Four toy robots each wobble left-right by a random, bell-shaped amount. A scientist adds up the squares of all four wobbles to get one 'total clumsiness' score.",
        "kid_question": "What's the chance the total clumsiness score comes out above 9?",
        "use_cases": [
            "Goodness-of-fit tests (does data match an expected distribution?)",
            "Constructing confidence intervals for a variance",
            "The building block behind the t and F distributions",
        ],
        "code": "dist = stats.chi2(df=4)\nprint('P(total clumsiness > 9):', round(dist.sf(9), 4))\nprint('expected total clumsiness:', dist.mean())",
    },
    "t": {
        "display": "Student's t",
        "params": [{"name": "df", "default": 8, "desc": "degrees of freedom (roughly, sample size - 1)"}],
        "formula": r"f(x) = \frac{\Gamma(\frac{df+1}{2})}{\sqrt{df\pi}\,\Gamma(\frac{df}{2})}\left(1+\frac{x^2}{df}\right)^{-\frac{df+1}{2}}",
        "intuition": (
            "Looks like the Normal bell curve but with fatter tails -- it shows up when you estimate an "
            "average from a *small* sample and don't actually know the true spread, only your sample's "
            "own (uncertain) estimate of it. Small samples deserve fatter tails, because your spread "
            "estimate itself might be off; as the sample grows, t quietly turns into Normal."
        ),
        "kid_story": "A kid wants to know the average number of jellybeans in a bag, but only got to open 9 bags so far, and their guess about how 'spread out' bag counts are is itself shaky.",
        "kid_question": "Based on just those 9 bags, how far from their average guess could the true average plausibly be (using the 8-degrees-of-freedom t distribution)?",
        "use_cases": [
            "Confidence intervals and hypothesis tests for a mean with small samples",
            "Any statistical test where the true variance is unknown and estimated from data",
            "Robust regression modeling (fatter tails than Normal)",
        ],
        "code": "dist = stats.t(df=8)\nprint('critical t-value (95% CI, two-sided):', round(dist.ppf(0.975), 4))",
    },
    "f": {
        "display": "F",
        "params": [
            {"name": "dfn", "default": 5, "desc": "numerator degrees of freedom"},
            {"name": "dfd", "default": 10, "desc": "denominator degrees of freedom"},
        ],
        "formula": r"f(x) = \frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}{(d_1 x + d_2)^{d_1+d_2}}}}{x\,B(d_1/2,\,d_2/2)}, \quad x \ge 0",
        "intuition": (
            "The distribution of a *ratio* of two independent Chi-square variables (each divided by its "
            "own degrees of freedom). Because it compares two spread-out-of-squares numbers, it's the "
            "natural tool for asking 'is group A more variable than group B?' or 'does adding these "
            "extra variables actually explain more variance?'"
        ),
        "kid_story": "Two classes take the same quiz. Class A's scores are more spread out than Class B's. A teacher wants a number that captures 'how much more spread out'.",
        "kid_question": "If the two classes were really equally variable, what's the chance the observed spread ratio would be this large just by chance?",
        "use_cases": [
            "ANOVA: comparing variance between multiple groups",
            "Comparing the variances of two samples",
            "The F-test in regression (does a set of predictors explain significant variance?)",
        ],
        "code": "dist = stats.f(dfn=5, dfd=10)\nprint('P(ratio >= 3):', round(dist.sf(3), 4))\nprint('critical value (95%):', round(dist.ppf(0.95), 4))",
    },
    "lognorm": {
        "display": "Log-normal",
        "params": [{"name": "s", "default": 0.5, "desc": "sigma of the underlying normal (shape)"}],
        "formula": r"f(x) = \frac{1}{xs\sqrt{2\pi}} e^{-\frac{(\ln x)^2}{2s^2}}, \quad x > 0",
        "intuition": (
            "If a quantity's *logarithm* is Normally distributed, the quantity itself is Log-normal: "
            "always positive, with a long right tail. This is what happens when many small effects "
            "multiply together rather than add -- exactly the pattern behind house prices, incomes, "
            "and stock prices, where a few outcomes end up enormous."
        ),
        "kid_story": "House sizes in a neighborhood: most houses are modest, a fair number are a bit bigger, and a rare few are giant mansions -- never negative, and skewed toward the big side.",
        "kid_question": "What fraction of houses are bigger than double the 'typical' size?",
        "use_cases": [
            "Household income and wealth distributions",
            "Stock prices and financial returns (multiplicative growth)",
            "City/company sizes, and particle size distributions",
        ],
        "code": "dist = stats.lognorm(s=0.5)\nprint('P(size > 2x typical):', round(dist.sf(2), 4))\nprint('median size:', round(dist.median(), 4))",
    },
    "weibull_min": {
        "display": "Weibull",
        "params": [
            {"name": "c", "default": 1.5, "desc": "shape -- controls whether failure risk rises or falls over time"},
            {"name": "scale", "default": 10, "desc": "characteristic lifetime"},
        ],
        "formula": r"f(x) = \frac{c}{\text{scale}}\left(\frac{x}{\text{scale}}\right)^{c-1} e^{-(x/\text{scale})^{c}}, \quad x \ge 0",
        "intuition": (
            "Exponential's more flexible cousin for 'time until failure' -- it lets the failure risk "
            "change over time instead of staying constant. Shape c < 1 means it's more likely to fail "
            "early (infant mortality); c > 1 means it wears out and gets *more* likely to fail the "
            "longer it's run; c = 1 collapses back to plain Exponential."
        ),
        "kid_story": "A toy robot's battery tends to die a bit more often the longer it's been running, rather than at a totally constant rate.",
        "kid_question": "What's the chance the battery lasts longer than 15 hours?",
        "use_cases": [
            "Reliability engineering: time-to-failure of mechanical and electronic parts",
            "Wind speed modeling",
            "Survival analysis in medicine and manufacturing",
        ],
        "code": "dist = stats.weibull_min(c=1.5, scale=10)\nprint('P(battery lasts > 15h):', round(dist.sf(15), 4))",
    },
    "pareto": {
        "display": "Pareto",
        "params": [{"name": "b", "default": 2.5, "desc": "shape -- how quickly the tail thins out"}],
        "formula": r"f(x) = \frac{b}{x^{b+1}}, \quad x \ge 1",
        "intuition": (
            "The '80/20 rule' distribution -- a small number of cases account for most of the total. "
            "It has a long, heavy tail: values far above the minimum are rare but not negligible, which "
            "is exactly the shape of wealth, city sizes, and file sizes."
        ),
        "kid_story": "In a make-believe kingdom, most families own a modest plot of land, but a handful of families own absolutely enormous estates.",
        "kid_question": "What fraction of families own more than 3 times the smallest possible plot?",
        "use_cases": [
            "Wealth and income distribution (the 80/20 rule)",
            "File size distributions on computer systems",
            "City population sizes and word-frequency (Zipf-like) phenomena",
        ],
        "code": "dist = stats.pareto(b=2.5)\nprint('P(land > 3x minimum):', round(dist.sf(3), 4))",
    },
    "cauchy": {
        "display": "Cauchy",
        "params": [
            {"name": "loc", "default": 0, "desc": "location (peak of the curve)"},
            {"name": "scale", "default": 1, "desc": "spread"},
        ],
        "formula": r"f(x) = \frac{1}{\pi\,\text{scale}\left[1+\left(\frac{x-\text{loc}}{\text{scale}}\right)^2\right]}",
        "intuition": (
            "Looks bell-shaped like Normal but is far wilder: it has no defined mean or variance, "
            "because extreme values are common enough that the averages never settle down. It shows up "
            "naturally as the ratio of two independent Normal variables, and it's the classic "
            "stress-test case in statistics for 'what breaks if the tails are too heavy'."
        ),
        "kid_story": "A laser pointer is mounted on a spinning stand at a fixed height above a long wall, and spun to point in a completely random direction. Wherever the beam hits the wall is the outcome.",
        "kid_question": "What's the chance the laser dot lands more than 5 units from the point directly opposite the laser?",
        "use_cases": [
            "Resonance and spectral line shapes in physics",
            "Stress-testing statistical methods that assume light tails",
            "Ratio of two independent Normal random variables",
        ],
        "code": "dist = stats.cauchy(loc=0, scale=1)\nprint('P(|X| > 5):', round(2 * dist.sf(5), 4))",
    },
    "logistic": {
        "display": "Logistic",
        "params": [
            {"name": "loc", "default": 0, "desc": "location (center)"},
            {"name": "scale", "default": 1, "desc": "spread"},
        ],
        "formula": r"f(x) = \frac{e^{-(x-\text{loc})/\text{scale}}}{\text{scale}\left(1+e^{-(x-\text{loc})/\text{scale}}\right)^2}",
        "intuition": (
            "Very close to Normal in shape but with a simple S-shaped CDF and slightly fatter tails -- "
            "that S-curve is exactly what logistic regression uses to turn any real number into a "
            "probability between 0 and 1, which is where the distribution gets its fame."
        ),
        "kid_story": "A silly video starts with almost no views, then suddenly 'catches on' and views shoot up fast, before eventually leveling off near everyone who was ever going to watch it.",
        "kid_question": "At what 'time' (x value) has the video reached 90% of its eventual total views?",
        "use_cases": [
            "Logistic regression (classification models)",
            "Growth curves that level off (viral spread, adoption curves)",
            "Neural network activation functions (the sigmoid is its CDF)",
        ],
        "code": "dist = stats.logistic(loc=0, scale=1)\nprint('x at 90% of the curve:', round(dist.ppf(0.9), 4))",
    },
    "laplace": {
        "display": "Laplace",
        "params": [
            {"name": "loc", "default": 0, "desc": "location (center)"},
            {"name": "scale", "default": 1, "desc": "spread (b)"},
        ],
        "formula": r"f(x) = \frac{1}{2b} e^{-|x-\text{loc}|/b}",
        "intuition": (
            "Two Exponential distributions glued back-to-back at the center -- a sharp peak with fatter "
            "tails than Normal. It models 'usually pretty close, but with more frequent bigger misses "
            "than a bell curve would predict', which is why it's the statistical backbone of robust "
            "(outlier-tolerant) methods and L1-style penalties."
        ),
        "kid_story": "A pretty good dart player usually lands very close to the bullseye, but every so often badly misjudges and lands quite far off, in either direction.",
        "kid_question": "What's the chance a dart lands more than 2 units from the bullseye?",
        "use_cases": [
            "Modeling errors with more outliers than a Normal distribution allows",
            "Signal and image processing (edge/differences tend to be Laplace-shaped)",
            "Statistical justification for L1 regularization / median-based robust regression",
        ],
        "code": "dist = stats.laplace(loc=0, scale=1)\nprint('P(|miss| > 2):', round(2 * dist.sf(2), 4))",
    },
    "rayleigh": {
        "display": "Rayleigh",
        "params": [{"name": "scale", "default": 2.0, "desc": "the typical magnitude scale, sigma"}],
        "formula": r"f(x) = \frac{x}{\sigma^2} e^{-x^2/(2\sigma^2)}, \quad x \ge 0",
        "intuition": (
            "If you take two independent, equally-spread-out Normal 'wiggles' (say, sideways and "
            "forward wind gusts) and ask how far the combined push lands from zero, the answer follows "
            "a Rayleigh distribution -- it's the distance from the origin when each coordinate is an "
            "independent, zero-mean Normal."
        ),
        "kid_story": "A dart-throwing robot aims at the exact bullseye, but its arm shakes a little left-right and a little up-down, each wobble following its own bell curve. Every dart lands some distance away, never exactly on target and rarely wildly far off.",
        "kid_question": "What's the chance a dart lands within 2 units of the bullseye, and what's the typical (mean) landing distance?",
        "use_cases": [
            "Wind speed modeling in meteorology",
            "Signal magnitude in wireless communication (Rayleigh fading)",
            "Distance from a target when aiming errors are 2D and Normal",
        ],
        "code": "dist = stats.rayleigh(scale=2.0)\nprint('P(distance <= 2):', round(dist.cdf(2), 4))\nprint('mean landing distance:', round(dist.mean(), 4))",
    },
    # ------------------------------------------------------------- multivariate
    "multinomial": {
        "display": "Multinomial",
        "params": [
            {"name": "n", "default": 10, "desc": "number of trials"},
            {"name": "p", "default": [0.2, 0.3, 0.5], "desc": "probability of each of the k outcomes"},
        ],
        "formula": r"P(x_1,\ldots,x_k) = \frac{n!}{x_1!\cdots x_k!}\,p_1^{x_1}\cdots p_k^{x_k}",
        "intuition": (
            "Binomial generalized from two outcomes to k outcomes: spin a weighted, multi-sided spinner "
            "n times and ask how many times each color came up. Binomial is just the k=2 special case."
        ),
        "kid_story": "A spinner has three colors -- red (20% chance), blue (30%), green (50%). A kid spins it 10 times.",
        "kid_question": "What's the chance they get exactly 2 reds, 3 blues, and 5 greens?",
        "use_cases": [
            "Survey responses split across multiple categories",
            "Word counts in text documents (bag-of-words models)",
            "Genetics: counts of different alleles in a sample",
        ],
        "code": "dist = stats.multinomial(n=10, p=[0.2, 0.3, 0.5])\nprint('P(2 red, 3 blue, 5 green):', round(dist.pmf([2, 3, 5]), 4))",
    },
    "multivariate_normal": {
        "display": "Multivariate Normal",
        "params": [
            {"name": "mean", "default": [0, 0], "desc": "center point in each dimension"},
            {"name": "cov", "default": [[1, 0.5], [0.5, 1]], "desc": "covariance matrix (spread and correlation)"},
        ],
        "formula": r"f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^k|\Sigma|}} \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\mu)^T \Sigma^{-1} (\mathbf{x}-\mu)\right)",
        "intuition": (
            "Normal in more than one dimension at once, where the variables can wobble together instead "
            "of independently -- the covariance matrix says both how spread out each one is, and how "
            "much they lean in the same direction as each other."
        ),
        "kid_story": "A dart robot's arm wobbles left-right and up-down like before, but this time a shaky elbow means when it drifts left it also tends to drift up a little -- the two wobbles aren't quite independent.",
        "kid_question": "What's the relative density of darts landing near the point (1, 1) compared to landing exactly on the bullseye (0, 0)?",
        "use_cases": [
            "Correlated returns of multiple assets in a financial portfolio",
            "Sensor noise modeling in robotics and tracking",
            "Gaussian mixture models and Gaussian processes in machine learning",
        ],
        "code": "dist = stats.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]])\nprint('density at (1,1):', round(dist.pdf([1, 1]), 4))\nprint('density at (0,0):', round(dist.pdf([0, 0]), 4))",
    },
    "dirichlet": {
        "display": "Dirichlet",
        "params": [{"name": "alpha", "default": [2, 3, 5], "desc": "concentration for each of the k proportions"}],
        "formula": r"f(x_1,\ldots,x_k) = \frac{1}{B(\alpha)}\prod_{i=1}^{k} x_i^{\alpha_i - 1}, \quad \sum_i x_i = 1",
        "intuition": (
            "Beta generalized from one proportion to a whole set of proportions that must add up to 1 -- "
            "it's the distribution *of the probability vector itself*, and is Multinomial's mathematical "
            "partner the same way Beta is Binomial's."
        ),
        "kid_story": "Three friends are randomly splitting one whole pizza. Based on past pizza nights, one friend usually ends up with the biggest share, another usually gets a medium share, and the third usually gets the smallest -- but it varies night to night.",
        "kid_question": "What's the relative likelihood of an even 1/3-1/3-1/3 split compared to a lopsided 20%-30%-50% split?",
        "use_cases": [
            "Bayesian prior for Multinomial category probabilities",
            "Topic modeling (Latent Dirichlet Allocation) in NLP",
            "Any 'proportions that must sum to 1' modeling problem",
        ],
        "code": "dist = stats.dirichlet(alpha=[2, 3, 5])\nprint('density at even split (1/3,1/3,1/3):', round(dist.pdf([1/3, 1/3, 1/3]), 4))\nprint('density at lopsided split (0.2,0.3,0.5):', round(dist.pdf([0.2, 0.3, 0.5]), 4))",
    },
}
