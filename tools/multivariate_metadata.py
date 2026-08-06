"""Hand-written metadata for the 11 Tier-2 multivariate & matrix distributions.

These sit in the 'extended catalog' tier (shorter treatment than Tier 1) but,
because their scipy call signatures are unusual, are hand-authored rather than
delegated, so the parameters are verified rather than guessed.
"""

MULTIVARIATE_METADATA = {
    "dirichlet_multinomial": {
        "display": "Dirichlet-Multinomial",
        "params": [
            {"name": "alpha", "default": [2, 3, 5], "desc": "concentration for each of the k categories"},
            {"name": "n", "default": 10, "desc": "number of trials"},
        ],
        "formula": r"P(\mathbf{x}) = \binom{n}{x_1,\ldots,x_k} \frac{B(\alpha+\mathbf{x})}{B(\alpha)}",
        "intuition": (
            "Multinomial, but the category probabilities themselves are randomly redrawn from a "
            "Dirichlet distribution each time -- so counts end up more spread out ('overdispersed') "
            "than plain Multinomial would predict."
        ),
        "kid_story": "Same pizza-splitting friends as the Dirichlet example, but now imagine repeating the pizza night 10 times, and each friend's typical share wobbles a bit differently every single night.",
        "kid_question": "Across 10 slices total, what's the chance of the exact same 2-3-5 split as the Dirichlet's average tendency?",
        "use_cases": ["Overdispersed text/topic count modeling in NLP", "Ecological species-count models with extra variability"],
        "code": "dist = stats.dirichlet_multinomial(alpha=[2, 3, 5], n=10)\nprint('P(split = 2,3,5):', round(float(dist.pmf([2, 3, 5])), 4))",
    },
    "multivariate_hypergeom": {
        "display": "Multivariate Hypergeometric",
        "params": [
            {"name": "m", "default": [10, 8, 7], "desc": "population count of each category"},
            {"name": "n", "default": 10, "desc": "sample size drawn without replacement"},
        ],
        "formula": r"P(\mathbf{x}) = \frac{\prod_i \binom{m_i}{x_i}}{\binom{\sum_i m_i}{n}}",
        "intuition": (
            "Hypergeometric extended to more than two categories: a bag has several colors of marbles, "
            "you grab a handful without putting any back, and ask how many of each color you got."
        ),
        "kid_story": "A bag holds 10 red, 8 blue, and 7 green marbles. A kid grabs a handful of 10 marbles all at once, no peeking, no putting back.",
        "kid_question": "What's the chance the handful is exactly 4 red, 3 blue, and 3 green?",
        "use_cases": ["Quality-control sampling across multiple defect categories", "Ecological sampling without replacement across several species"],
        "code": "dist = stats.multivariate_hypergeom(m=[10, 8, 7], n=10)\nprint('P(4 red, 3 blue, 3 green):', round(float(dist.pmf([4, 3, 3])), 4))",
    },
    "multivariate_t": {
        "display": "Multivariate t",
        "params": [
            {"name": "loc", "default": [0, 0], "desc": "center point in each dimension"},
            {"name": "shape", "default": [[1, 0.3], [0.3, 1]], "desc": "spread/correlation matrix"},
            {"name": "df", "default": 5, "desc": "degrees of freedom (lower = fatter tails)"},
        ],
        "formula": r"f(\mathbf{x}) \propto \left(1 + \tfrac{1}{df}(\mathbf{x}-\mu)^T \Sigma^{-1} (\mathbf{x}-\mu)\right)^{-\frac{df+k}{2}}",
        "intuition": (
            "Multivariate Normal's fatter-tailed cousin: the same idea of several correlated wobbling "
            "quantities, but with a higher chance of an occasional extreme joint move in several "
            "dimensions at once -- useful whenever 'everything crashes together' matters."
        ),
        "kid_story": "The dart robot from the Multivariate Normal example, except on rare days its whole arm has a 'bad day' and throws wildly off in both directions together, more often than a calm bell curve would suggest.",
        "kid_question": "How much more/less likely is a joint miss to (1,1) compared to landing exactly on target, versus the calmer Normal version?",
        "use_cases": ["Robust portfolio risk modeling (fat-tailed joint asset moves)", "Robust multivariate regression error modeling"],
        "code": "dist = stats.multivariate_t(loc=[0, 0], shape=[[1, 0.3], [0.3, 1]], df=5)\nprint('density at (1,1):', round(float(dist.pdf([1, 1])), 4))\nprint('density at (0,0):', round(float(dist.pdf([0, 0])), 4))",
    },
    "vonmises_fisher": {
        "display": "Von Mises-Fisher",
        "params": [
            {"name": "mu", "default": [0, 0, 1], "desc": "the preferred direction (unit vector)"},
            {"name": "kappa", "default": 5.0, "desc": "concentration -- how tightly clustered around mu"},
        ],
        "formula": r"f(\mathbf{x}) = C_p(\kappa)\, e^{\kappa\, \mu^T \mathbf{x}}, \quad \|\mathbf{x}\| = 1",
        "intuition": (
            "The Normal distribution's equivalent for directions instead of positions: points are "
            "constrained to the surface of a sphere, clustering around a preferred direction mu, with "
            "kappa controlling how tightly (kappa=0 is uniform all over the sphere)."
        ),
        "kid_story": "A toy compass needle mostly points north, but wobbles a little side to side and up-down instead of pointing exactly true north every time.",
        "kid_question": "How much denser is the cloud of needle directions right at true north compared to a direction 90 degrees away?",
        "use_cases": ["Directional statistics (wind direction, animal migration heading)", "Embeddings constrained to a hypersphere in machine learning"],
        "code": "dist = stats.vonmises_fisher(mu=[0, 0, 1], kappa=5.0)\nprint('density at true north (0,0,1):', round(float(dist.pdf([0, 0, 1])), 4))\nprint('density 90 deg off (1,0,0):', round(float(dist.pdf([1, 0, 0])), 4))",
    },
    "wishart": {
        "display": "Wishart",
        "params": [
            {"name": "df", "default": 5, "desc": "degrees of freedom (number of samples averaged into it)"},
            {"name": "scale", "default": [[1, 0], [0, 1]], "desc": "the underlying scale matrix"},
        ],
        "formula": r"f(\mathbf{X}) \propto |\mathbf{X}|^{\frac{df-p-1}{2}} e^{-\frac{1}{2}\text{tr}(\Sigma^{-1}\mathbf{X})}",
        "intuition": (
            "The distribution of a sample covariance matrix itself: draw df random vectors, form their "
            "covariance, and that covariance matrix is Wishart-distributed. It's the matrix-valued "
            "generalization of the Chi-square distribution (in fact Wishart with a 1x1 scale matrix "
            "*is* a Chi-square)."
        ),
        "kid_story": "Instead of one wobbly dart-throwing robot, imagine averaging together the 'wobble patterns' from 5 different robots -- the result is a whole matrix describing typical combined spread and correlation, and that matrix itself is random.",
        "kid_question": "What does a single random draw of that combined-wobble matrix look like?",
        "use_cases": ["Covariance matrix estimation and hypothesis testing", "Bayesian statistics: the natural prior/likelihood for covariance matrices"],
        "code": "dist = stats.wishart(df=5, scale=np.eye(2))\nsample = dist.rvs(random_state=0)\nprint('one sampled covariance matrix:')\nprint(np.round(sample, 3))",
    },
    "invwishart": {
        "display": "Inverse Wishart",
        "params": [
            {"name": "df", "default": 5, "desc": "degrees of freedom"},
            {"name": "scale", "default": [[1, 0], [0, 1]], "desc": "the underlying scale matrix"},
        ],
        "formula": r"f(\mathbf{X}) \propto |\mathbf{X}|^{-\frac{df+p+1}{2}} e^{-\frac{1}{2}\text{tr}(\Sigma \mathbf{X}^{-1})}",
        "intuition": (
            "The matrix version of 'flip it upside down': if a covariance matrix is Wishart-distributed, "
            "its inverse is Inverse-Wishart. It's the standard Bayesian prior for an unknown covariance "
            "matrix, the same role Inverse-Gamma plays for a single unknown variance."
        ),
        "kid_story": "A scientist doesn't know in advance how two ingredients in a recipe wobble together, so before seeing any data they draw a plausible 'recipe wobble matrix' from a big bag of reasonable guesses.",
        "kid_question": "What does one such randomly-guessed wobble matrix look like, before any real data is seen?",
        "use_cases": ["Bayesian prior for an unknown covariance matrix", "Hierarchical models in Bayesian statistics (e.g. Gaussian mixture priors)"],
        "code": "dist = stats.invwishart(df=5, scale=np.eye(2))\nsample = dist.rvs(random_state=0)\nprint('one sampled covariance matrix:')\nprint(np.round(sample, 3))",
    },
    "matrix_normal": {
        "display": "Matrix Normal",
        "params": [
            {"name": "mean", "default": [[0, 0], [0, 0]], "desc": "mean matrix"},
            {"name": "rowcov", "default": [[1, 0], [0, 1]], "desc": "covariance between rows"},
            {"name": "colcov", "default": [[1, 0], [0, 1]], "desc": "covariance between columns"},
        ],
        "formula": r"f(\mathbf{X}) \propto \exp\!\left(-\tfrac{1}{2}\text{tr}\!\left[\Sigma_{col}^{-1}(\mathbf{X}-M)^T \Sigma_{row}^{-1}(\mathbf{X}-M)\right]\right)",
        "intuition": (
            "Multivariate Normal stretched across a whole grid instead of a single vector: every cell "
            "in a matrix wobbles Normally, with separate covariance structure for how rows relate to "
            "each other and how columns relate to each other."
        ),
        "kid_story": "Instead of one dart-throwing robot, picture a 2x2 grid of them, where robots in the same row tend to wobble together, and robots in the same column also tend to wobble together.",
        "kid_question": "What does one random snapshot of the whole grid's wobble look like?",
        "use_cases": ["Spatiotemporal data (grid of sensors over time)", "Multivariate time series with structured row/column covariance"],
        "code": "dist = stats.matrix_normal(mean=np.zeros((2, 2)), rowcov=np.eye(2), colcov=np.eye(2))\nsample = dist.rvs(random_state=0)\nprint('one sampled 2x2 grid:')\nprint(np.round(sample, 3))",
    },
    "ortho_group": {
        "display": "Orthogonal Group",
        "params": [{"name": "dim", "default": 3, "desc": "dimension of the random orthogonal matrix"}],
        "formula": r"\text{Uniform measure over } O(n) = \{\mathbf{Q} : \mathbf{Q}^T\mathbf{Q} = I\}",
        "intuition": (
            "Not a distribution over numbers but over *rotations and reflections*: every possible way to "
            "rigidly rotate or flip space is equally likely. Multiplying any vector by one of these "
            "matrices spins or mirrors it without stretching or squashing it."
        ),
        "kid_story": "Randomly spinning (or sometimes mirror-flipping) a cube in space to a completely new orientation, with every possible orientation equally likely.",
        "kid_question": "What does one such random orientation matrix look like?",
        "use_cases": ["Random rotations in computer graphics and robotics", "Random orthogonal projections/initializations in machine learning"],
        "code": "dist = stats.ortho_group(dim=3)\nsample = dist.rvs(random_state=0)\nprint('one random orthogonal matrix:')\nprint(np.round(sample, 3))",
    },
    "special_ortho_group": {
        "display": "Special Orthogonal Group",
        "params": [{"name": "dim", "default": 3, "desc": "dimension of the random rotation matrix"}],
        "formula": r"\text{Uniform measure over } SO(n) = \{\mathbf{Q}: \mathbf{Q}^T\mathbf{Q}=I,\ \det(\mathbf{Q})=1\}",
        "intuition": (
            "Just like the Orthogonal Group, but mirror-flips are excluded -- only genuine, "
            "physically-realizable rotations (like actually spinning a globe) are allowed."
        ),
        "kid_story": "Spinning a globe on its stand to a totally random new orientation -- twisting only, never flipped inside-out like a mirror image.",
        "kid_question": "What does one such random pure-rotation matrix look like?",
        "use_cases": ["Random 3D rotations in robotics, animation, and molecular simulation", "Data augmentation via random rotation in 3D machine learning"],
        "code": "dist = stats.special_ortho_group(dim=3)\nsample = dist.rvs(random_state=0)\nprint('one random rotation matrix:')\nprint(np.round(sample, 3))",
    },
    "unitary_group": {
        "display": "Unitary Group",
        "params": [{"name": "dim", "default": 2, "desc": "dimension of the random unitary (complex) matrix"}],
        "formula": r"\text{Uniform measure over } U(n) = \{\mathbf{U}: \mathbf{U}^{*}\mathbf{U} = I\}",
        "intuition": (
            "The complex-number version of the Orthogonal Group -- random 'rotations' in complex vector "
            "space. It's the mathematical structure quantum computing gates are built from: every valid "
            "quantum operation is a unitary matrix."
        ),
        "kid_story": "Imagine a magical spinner whose arrow can point in directions that mix ordinary spinning with a second, invisible 'imaginary' spinning dimension -- and it lands on a totally random valid combination.",
        "kid_question": "What does one such random valid rotation look like?",
        "use_cases": ["Modeling random quantum gates/circuits in quantum computing", "Random matrix theory in physics"],
        "code": "dist = stats.unitary_group(dim=2)\nsample = dist.rvs(random_state=0)\nprint('one random unitary matrix (magnitudes shown):')\nprint(np.round(np.abs(sample), 3))",
    },
    "random_correlation": {
        "display": "Random Correlation Matrix",
        "params": [{"name": "eigs", "default": [0.5, 0.8, 1.2, 1.5], "desc": "desired eigenvalues (must sum to the dimension)"}],
        "formula": r"\text{Correlation matrices } \mathbf{R} \text{ with prescribed eigenvalues } \{\lambda_i\}, \textstyle\sum \lambda_i = n",
        "intuition": (
            "Generates a random, valid correlation matrix (1's on the diagonal, everything between -1 "
            "and 1 off it) that matches a chosen set of eigenvalues -- useful for stress-testing "
            "portfolios or simulations against a specific 'how correlated is everything' profile."
        ),
        "kid_story": "A teacher wants to randomly pair up how strongly every subject's grades relate to every other subject's grades (math with science, science with art, etc.), while keeping the overall 'total relatedness' fixed at a chosen amount.",
        "kid_question": "What does one such random, valid correlation matrix between 4 subjects look like?",
        "use_cases": ["Stress-testing portfolio risk models against specific correlation structures", "Simulating multivariate data with a controlled correlation profile"],
        "code": "sample = stats.random_correlation.rvs(eigs=[0.5, 0.8, 1.2, 1.5], random_state=0)\nprint('one random correlation matrix:')\nprint(np.round(sample, 3))",
    },
}
