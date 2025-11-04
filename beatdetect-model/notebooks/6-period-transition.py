import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from beatdetect.config_loader import load_config
    from beatdetect.utils.paths import iterate_beat_files
    from scipy.stats import laplace, norm
    from sklearn.mixture import GaussianMixture

    return (
        GaussianMixture,
        go,
        iterate_beat_files,
        laplace,
        load_config,
        mo,
        norm,
        np,
        pl,
        px,
    )


@app.cell
def _(load_config):
    config = load_config()
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We want to model changes in beat period, $P(p_t | p_{t-1})$ for our HMM post processing. We will use a mixture of two [Laplace distribution](https://en.wikipedia.org/wiki/Laplace_distribution).

    Let the log-ratio between consecutive periods be $\delta \coloneqq \ln {\frac{p_t}{p_{t-1}}}$, which we will assume follows the mixture model
    $$f(\delta) = (1-\pi) f_L(\delta; \lambda_1) + \pi f_L(\delta; \lambda_2)$$
    where $f_L$ is the continuous Laplace distribution defined as
    $$f_L(\delta; \lambda) \coloneqq  \frac{\lambda}{2} e^{-\lambda |\delta|}, \quad \lambda \geq 0$$
    $f_L(\delta; \lambda)$ is more concentrated around $0$ the higher $\lambda$ is.

    If $\lambda_1 > \lambda_2$, we can interpret $\pi$ as the probability that a tempo change occurs. When no tempo change happens, the new period ($p_t$) is likely to remain close to the previous one ($p_t-1$), meaning $\delta$ stays near $0$.

    Our goal is to estimate $\pi, \lambda_1, \lambda_2$ so that $P(\delta)$ best fits the data.
    To do this we use the [Expectation–maximization algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm):

    1. Choose initial values $\pi^{(0)}, \lambda_1^{(0)}, \lambda_2^{(0)}$.
    2. E-step (iteration $k$): Using current parameter estimates, compute the responsibility that each observation $\delta_i$ belongs to component 2:
    $$\gamma_i^{(k)} = \frac{\pi^{(k)} f_L(\delta_i; \lambda_2^{(k)})}{\pi^{(k)} f_L(\delta_i; \lambda_2^{(k)}) + (1-\pi^{(k)}) f_L(\delta_i; \lambda_1^{(k)})}$$
    3. M-step (iteration $k$): Update parameters using the responsibilities:
    $$\pi^{(k+1)} = \frac{1}{n}\sum_{i=1}^n \gamma_i^{(k)}, \qquad
    \lambda_1^{(k+1)} = \frac{\sum_{i=1}^n (1-\gamma_i^{(k)})}{\sum_{i=1}^n (1-\gamma_i^{(k)})|\delta_i|}, \qquad
    \lambda_2^{(k+1)} = \frac{\sum_{i=1}^n \gamma_i^{(k)}}{\sum_{i=1}^n \gamma_i^{(k)} |\delta_i|}$$

    Iterate E and M steps until the log-likelihood 
    $$\mathcal{L}^{(k)} = \sum_{i=1}^n \log\left[(1-\pi^{(k)}) f(\delta_i; \lambda_1^{(k)}) + \pi^{(k)} f(\delta_i; \lambda_2^{(k)})\right]$$
    converges, i.e., $|\mathcal{L}^{(k+1)} - \mathcal{L}^{(k)}| < \epsilon$ for some tolerance $\epsilon$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Extracting features from data""")
    return


@app.cell
def _(config, iterate_beat_files, np, pl):
    all_periods = []
    period_ratio = []

    for _beats_file in iterate_beat_files(config):
        _beats = (
            pl.read_csv(
                _beats_file,
                separator="\t",
                has_header=False,
            )
            .get_columns()[0]
            .to_list()
        )

        _beats = np.array(sorted(_beats))
        _period = np.diff(_beats * config.spectrogram.fps)
        _period = _period[_period > 0]
        all_periods.append(_period)
        if len(_period) < 2:
            continue
        period_ratio.append(_period[1:] / _period[:-1])

    all_periods = np.concatenate(all_periods)
    all_tempos = 60 / (all_periods / 50)
    period_ratio = np.concatenate(period_ratio)
    delta = np.log(period_ratio)

    print(f"Loaded {len(all_periods)} songs, {len(all_periods)} beat intervals total.")
    return all_tempos, delta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Explore tempo distribution""")
    return


@app.cell
def _(all_tempos, np, px):
    px.histogram(
        all_tempos[all_tempos < 400].astype(np.float16),
        labels={"value": "Tempo (BPM)"},
        title="Distribution of Tempi Across Dataset",
    )
    return


@app.cell
def _(all_tempos, np):
    def summary(x):
        print(f"Mean: {np.mean(x)}")
        print(f"Median: {np.median(x)}")
        print(f"STD: {np.std(x)}")
        print(f"MAD: {np.median(np.abs(x - np.median(x)))}")
        print(f"IQR: {np.subtract(*np.percentile(x, [75, 25]))}")

    summary(all_tempos)
    return (summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Explore $\delta$ distribution""")
    return


@app.cell
def _(delta, np, px):
    px.histogram(
        np.random.choice(delta[abs(delta) < 0.5].astype(np.float16), 99999),
        labels={"value": "δ Period (ln(period / previous period))"},
        title="Distribution of Tempo Changes Between Consecutive Sections",
        nbins=100,
    )
    return


@app.cell
def _(delta, summary):
    summary(delta)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Initial parameters

    We can make informed guess for initial values $\pi^{(0)}, \lambda_1^{(0)}, \lambda_2^{(0)}$ for the EM algorithm.

    For a Laplace distribution, the mean absolute deviation is: $E[|X|] = \frac{1}{\lambda}$, so $\hat{\lambda}=\frac{1}{mean(|X|)}$.
    """
    )
    return


@app.cell
def _(delta, np):
    near_zero = np.abs(delta) < 1e-3  # meaning places with no change in period
    pi_0 = 1 - np.mean(near_zero)
    lambda1_0 = 1 / (np.mean(np.abs(delta[near_zero])) + 1e-8)
    lambda2_0 = 1 / (np.mean(np.abs(delta[~near_zero])) + 1e-8)
    print(f"π = {pi_0}")
    print(f"λ1 = {lambda1_0}")
    print(f"λ2 = {lambda2_0}")
    return lambda1_0, lambda2_0, pi_0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Baseline models""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Single Laplace
    We try to fit a single laplace model first, meaning $\delta∼Laplace(λ)$. We try to estimate lambda using the fact that $\lambda = \frac{1}{E[|\delta|]}$.
    """
    )
    return


@app.cell
def _(delta, laplace, np):
    lambda_single = 1 / np.mean(np.abs(delta))
    log_likelihood_single = np.sum(
        np.log(laplace.pdf(delta, loc=0, scale=1 / lambda_single))
    )
    avg_log_likelihood_single = log_likelihood_single / len(delta)

    print("Single Laplace:")
    print("lambda =", lambda_single)
    print("Avg log-likelihood =", avg_log_likelihood_single)
    return (lambda_single,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Guassian mixture
    Instead of mixture of 2 Laplace distributions, another option for the baseline is a mixture of 2 Gaussian distribution, meaning
    $$P(\delta) = (1 - \pi) , \mathcal{N}(\delta \mid \mu_1, \sigma_1^2) + \pi \mathcal{N}(\delta \mid \mu_2, \sigma_2^2)$$
    There's already [implentation](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture) for this in scipy, which will tunes all of the parameters for us.
    """
    )
    return


@app.cell
def _(GaussianMixture, delta, np):
    delta_obs_reshaped = delta.reshape(-1, 1)  # sklearn expects 2D
    gmm = GaussianMixture(n_components=2, covariance_type="diag", random_state=0)
    gmm.fit(delta_obs_reshaped)
    # per-sample log-likelihood
    log_likelihood_gmm = gmm.score_samples(delta_obs_reshaped)
    avg_log_likelihood_gmm = np.mean(log_likelihood_gmm)

    print("\n2-component Gaussian Mixture:")
    print("Number of EM iterations:", gmm.n_iter_)
    print("Weights:", gmm.weights_)
    print("Means:", gmm.means_.flatten())
    print("Variances:", gmm.covariances_.flatten())
    print("Avg log-likelihood =", avg_log_likelihood_gmm)
    return (gmm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# EM algorithm for Laplace mixture""")
    return


@app.cell
def _(delta, lambda1_0, lambda2_0, laplace, np, pi_0):
    def em_laplace_mixture(X, pi_0, lambda1_0, lambda2_0, tol=1e-8, max_iter=1000):
        pi = pi_0
        lambda1 = lambda1_0
        lambda2 = lambda2_0

        log_likelihoods = []

        for k in range(max_iter):
            # E-step: responsibilities for component 2
            f1 = laplace.pdf(X, scale=1 / lambda1)
            f2 = laplace.pdf(X, scale=1 / lambda2)
            gamma = (pi * f2) / (pi * f2 + (1 - pi) * f1)

            # M-step: update parameters
            pi_new = np.mean(gamma)
            lambda1_new = np.sum(1 - gamma) / np.sum((1 - gamma) * np.abs(X))
            lambda2_new = np.sum(gamma) / np.sum(gamma * np.abs(X))

            # Compute log-likelihood
            log_likelihood = np.sum(np.log((1 - pi_new) * f1 + pi_new * f2))
            log_likelihoods.append(log_likelihood)

            # Check convergence
            if k > 0 and np.abs(log_likelihood - log_likelihoods[-2]) < tol:
                break

            pi, lambda1, lambda2 = pi_new, lambda1_new, lambda2_new
        print(f"Converged after {k} iterations")

        return (
            pi,
            lambda1,
            lambda2,
            log_likelihood / len(delta),
        )

    pi, lambda1, lambda2, avg_log_likelihood_lmm = em_laplace_mixture(
        delta, pi_0, lambda1_0, lambda2_0
    )
    print(f"π = {pi}")
    print(f"λ1 = {lambda1}")
    print(f"λ2 = {lambda2}")
    print("Avg log-likelihood =", avg_log_likelihood_lmm)
    return lambda1, lambda2, pi


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Comparison

    We compared three models for the log-difference of beat periods, $\delta = \ln(p_t / p_{t-1})$: a single Laplace, a 2-component Laplace mixture, and a 2-component Gaussian mixture. 

    - Single Laplaceachieved: average log-likelihood of 1.645.
    - 2-component Gaussian mixture: average log-likelihood of 2.529.
    - 2-component Laplace mixture, estimated via a custom EM algorithm, achieved the highest average log-likelihood of 4.5187.

    The Laplace mixture has heavier tails allow it to better model both small $\delta$ (no tempo change) and larger $\delta$ (tempo jumps), outperforming the Gaussian mixture and the single Laplace. These results justify using the Laplace mixture to model beat period changes in this dataset.
    """
    )
    return


@app.cell
def _(delta, gmm, go, lambda1, lambda2, lambda_single, laplace, norm, np, pi):
    hist_vals, bin_edges = np.histogram(delta[abs(delta) < 0.5], bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    def bin_aware_pdf(cdf_func, *args, **kwargs):
        cdf_hi = cdf_func(bin_edges[1:], *args, **kwargs)
        cdf_lo = cdf_func(bin_edges[:-1], *args, **kwargs)
        pdf_bin = (cdf_hi - cdf_lo) / bin_widths
        return pdf_bin

    pdf_single = bin_aware_pdf(laplace.cdf, scale=1 / lambda_single)

    # Compute bin-averaged model density (prob per bin / bin width)
    pdf_mixlap = np.zeros(len(bin_edges) - 1)
    for w, lam in ((1 - pi, lambda1), (pi, lambda2)):
        pdf_mixlap += w * bin_aware_pdf(laplace.cdf, scale=1 / lam)

    pdf_gaussmix = np.zeros(len(bin_edges) - 1)
    for w, mu, sigma in zip(
        gmm.weights_.ravel(),
        gmm.means_.ravel(),
        np.sqrt(gmm.covariances_.ravel()),
        strict=False,
    ):
        pdf_gaussmix += w * bin_aware_pdf(norm.cdf, mu, sigma)

    # === Plot ===
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=hist_vals,
            width=bin_widths,
            name="Empirical δ",
            opacity=0.5,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=pdf_single,
            mode="lines",
            name="Single Laplace",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=pdf_mixlap,
            mode="lines",
            name="Laplace Mixture",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=pdf_gaussmix,
            mode="lines",
            name="Gaussian Mixture",
        )
    )

    fig.update_layout(
        title=" δ Distribution and Model Fits",
        yaxis_title="Density",
        bargap=0,
        legend=dict(x=0.02, y=0.98),
    )
    fig
    return


if __name__ == "__main__":
    app.run()
