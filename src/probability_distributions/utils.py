import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rv_discrete


def test():
    return "aha probability distribution"


def plot_distribution(dist, size=10000, random_state=None):
    """Sample a frozen scipy distribution and overlay it against its theoretical pmf/pdf.
    Auto-detects discrete vs continuous so the same call works for both."""
    samples = dist.rvs(size=size, random_state=random_state)
    is_discrete = isinstance(dist.dist, rv_discrete)

    if is_discrete:
        sns.displot(samples, discrete=True, stat="probability")
        x = np.arange(samples.min(), samples.max() + 1)
        y = dist.pmf(x)
        plt.plot(x, y, "o-", color="red", label="theoretical pmf")
    else:
        sns.displot(samples, kde=True, stat="density")
        x = np.linspace(samples.min(), samples.max(), 300)
        y = dist.pdf(x)
        plt.plot(x, y, color="red", label="theoretical pdf")

    plt.legend()
    return samples