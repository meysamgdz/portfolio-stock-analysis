import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm, gaussian_kde
import powerlaw
import matplotlib.pyplot as plt
import seaborn as sns

class FatTailStock():
    """
    A class for analyzing fat-tailed distributions in stock returns.

    Attributes:
        data (pd.DataFrame): Stock price data after individual checks and forward-filling.
        lin_returns (pd.DataFrame): Linear returns computed from stock prices.
        tickers (list): List of stock tickers (column names in data).
        alpha (pd.DataFrame): Power-law exponents (alpha) for stock returns.
        kurtosis (pd.Series): Kurtosis values for stock returns.
        lognorm_sigma (pd.DataFrame): Lognormal sigma values for stock returns.
        t_kapa (pd.DataFrame): Taleb's Kappa metric for fat-tailed distributions.

    Methods:
        get_fat_tail_metrics(): Computes alpha, kurtosis, log sigma, and Taleb's Kappa.
        expected_return(weights): Calculates the expected portfolio return.
        value_at_risk(confidence_level): Estimates the Value-at-Risk (VaR) at a given confidence level.
        std(weights): Computes the standard deviation of portfolio returns.
        get_cov_matrix(): Returns the covariance matrix of stock returns.
        get_lin_returns(): Computes the linear returns of stock prices.
        get_alpha(): Estimates the power-law exponent (alpha) for stock returns.
        get_log_sigma(): Computes lognormal sigma for stock returns.
        taleb_kapa(n0, n): Computes Taleb's Kappa metric for stock returns.
        mad(data): Computes the Mean Absolute Deviation (MAD) of a dataset.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the FatTailStock class by performing individual stock checks and computing linear returns.

        Args:
            data (pd.DataFrame): Stock price data with tickers as columns.
        """
        self._data = data
        self.lin_returns = self.get_lin_returns()
        self.log_returns = self.get_log_returns()
        self.tickers = self.data.columns.to_list()
        self.alpha = None
        self.kurtosis = None
        self.lognorm_params = None
        self.t_kapa = None

        self._validate_inputs_fts()

    def _validate_inputs_fts(self) -> None:
        """Validate constructor inputs"""
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if self.data.empty:
            raise ValueError("Data cannot be empty")
        if len(self.data) < 2:
            raise ValueError("At least 2 data points required")
        if not all(self.data.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError("All columns must be numeric")
        if (self.data <= 0).any().any():
            raise ValueError("All prices must be positive")
        if self.data.isna().any().any():
            raise ValueError("Input data contains missing values.")

    # Getter
    @property
    def data(self):
        return self._data

    # Setter
    @data.setter
    def data(self, new_data):
        if not isinstance(new_data, pd.DataFrame):
            raise ValueError("Data must be a Pandas DataFrame!")
        self._data = new_data

    def get_fat_tail_metrics(self):
        """
        Computes fat-tail related metrics: alpha, kurtosis, log sigma, and Taleb's Kappa.
        """
        self.alpha = self.get_alpha()
        self.kurtosis = self.lin_returns.kurtosis().round(3)
        self.lognorm_params = self.get_lognorm_params()
        self.t_kapa = self.taleb_kappa(n0=int(0.10*self.data.shape[0]), n=int(0.6*self.data.shape[0]))


    def value_at_risk(self, confidence_level: float = 0.95):
        """
        Estimates the Value-at-Risk (VaR) for each stock at a given confidence level.

        Args:
            confidence_level (float, optional): Confidence level for VaR calculation. Default is 0.95.

        Returns:
            pd.DataFrame: Value-at-Risk values for each stock.
        """
        var = pd.DataFrame()
        for ticker in self.tickers:
            sorted_lin_returns = sorted(self.lin_returns[ticker])
            var[ticker] = [np.percentile(sorted_lin_returns, 100 * (1 - confidence_level))]
        return var

    def get_cov_matrix(self):
        """
        Computes the covariance matrix of stock returns.

        Returns:
            pd.DataFrame: Covariance matrix of stock returns.
        """
        return self.get_lin_returns().cov()

    def get_lin_returns(self):
        """
        Computes the linear returns of stock prices.

        Returns:
            pd.DataFrame: Linear returns for each stock.
        """
        lin_returns = self.data.pct_change().dropna()
        if lin_returns.std().min() < 1e-10:
            raise ValueError("Constant or near-constant returns detected")
        return lin_returns

    def get_log_returns(self):
        """
        Computes the logarithmic returns of stock prices.

        Returns:
            pd.DataFrame: logarithmic returns for each stock.
        """
        return np.log10(1 + self.get_lin_returns())

    def get_lognorm_params(self):
        """
        Computes lognormal sigma for positive and negative stock returns.

        Returns:
            pd.DataFrame: Lognormal sigma values for stock returns.
        """
        fitted_powerlaw = self.fit_powerlaw()
        lognorm_params = pd.DataFrame([], index=["left", "right"], columns=self.tickers)
        for ticker in self.tickers:
            sigma_left = fitted_powerlaw.loc["left", ticker].lognormal.sigma
            mu_left = fitted_powerlaw.loc["left", ticker].lognormal.mu
            sigma_right = fitted_powerlaw.loc["right", ticker].lognormal.sigma
            mu_right = fitted_powerlaw.loc["right", ticker].lognormal.mu
            lognorm_params[ticker] = [(mu_left.round(2), np.exp(mu_left).round(2), sigma_left.round(2)),
                                      (mu_right.round(2), np.exp(mu_right).round(2), sigma_right.round(2))]
        return lognorm_params

    def fit_powerlaw(self):
        """
        Fits power-law distributions to the positive and negative linear returns for each ticker.

        This method separates the linear returns of each ticker into positive and negative values,
        takes the absolute value of the negative returns (to ensure positive inputs to the power-law fit),
        and fits a power-law distribution to both sides using the `powerlaw.Fit` function.

        Returns
        -------
        pd.DataFrame
            A DataFrame with index `["left", "right"]` and columns as tickers. Each cell contains
            a `powerlaw.Fit` object:
                - "left": fit for the absolute value of negative returns.
                - "right": fit for the positive returns.
        """
        returns = self.lin_returns
        power_law = pd.DataFrame([], index=["left", "right"], columns=self.tickers)
        for ticker in self.tickers:
            returns_neg = list(np.abs(returns[ticker][returns[ticker] < 0]))
            returns_pos = list(returns[ticker][returns[ticker] >= 0])
            # fitted_powerlaw_left = powerlaw.Fit(returns_neg, xmin=np.mean(returns_neg))
            # fitted_powerlaw_right = powerlaw.Fit(returns_pos, xmin=np.mean(returns_pos))
            fitted_powerlaw_left = powerlaw.Fit(returns_neg)
            fitted_powerlaw_right = powerlaw.Fit(returns_pos)
            power_law[ticker] = [fitted_powerlaw_left, fitted_powerlaw_right]
        return power_law

    def compare_powerlaw(self, dist_list: list = ["lognormal"]):
        """
        Compare power-law fits to alternative distributions for positive and negative returns.

        This method uses the power-law fits obtained from `fit_powerlaw()` and compares them
        to alternative distributions (`lognormal` and `exponential`) using likelihood ratio tests.
        It computes the likelihood ratio and p-value for each comparison on both the negative
        ("left") and positive ("right") return tails.

        Returns
        -------
        str
            A formatted string containing the comparison results for each ticker, suitable for
            display in a Streamlit app. Each section includes likelihood ratios and p-values
            for comparisons of power-law vs lognormal and exponential distributions, for both
            left (negative returns) and right (positive returns) fits.
        """
        fitted_powerlaw = self.fit_powerlaw()
        return_str = ""  # Initialize the return string

        for ticker in self.tickers:
            return_str += f"Comparison of fit quality for powerlaw and lognormal distributions:<br>"
            for dist in dist_list:
                # Left fit comparison
                R, p_value = fitted_powerlaw.loc["left", ticker].distribution_compare("power_law", dist)
                return_str += (
                    f"**Left fit**: &nbsp;&nbsp;&nbsp;&nbsp;powerlaw vs {dist}: "
                    f"likelihood_ratio = {R: .2f}, p-value = {p_value: .2f}<br>"
                )
                # Right fit comparison
                R, p_value = fitted_powerlaw.loc["right", ticker].distribution_compare("power_law", dist)
                return_str += (
                    f"**Right fit**: &nbsp;&nbsp;&nbsp;&nbsp;powerlaw vs {dist}: "
                    f"likelihood_ratio = {R: .2f}, p-value = {p_value: .2f}<br>"
                )
            return_str += "<br>"  # Add an extra line break between tickers

        return return_str

    def get_alpha(self):
        """
        Estimates the power-law exponent (alpha) for positive and negative stock returns.

        Returns:
            pd.DataFrame: Alpha values for right-tail (gains) and left-tail (losses).
        """
        fitted_powerlaw = self.fit_powerlaw()
        alpha = pd.DataFrame([], index=["left", "right"], columns=self.tickers)
        for ticker in self.tickers:
            alpha_left = fitted_powerlaw.loc["left", ticker].power_law.alpha
            xmin_left = fitted_powerlaw.loc["left", ticker].power_law.xmin
            alpha_right = fitted_powerlaw.loc["right", ticker].power_law.alpha
            xmin_right = fitted_powerlaw.loc["right", ticker].power_law.xmin
            alpha[ticker] = [(alpha_left.round(2), np.round(xmin_left, 2)),
                             (alpha_right.round(2), np.round(xmin_right, 2))]
        return alpha

    def taleb_kappa(self, n0: int = 10, n: int = 100, num_iterations: int = 1000):
        """
        Computes Taleb's Kappa metric for measuring fat-tailed distributions.

        Args:
            n0 (int, optional): Small sample size. Default is 10.
            n (int, optional): Large sample size. Default is 100.
            num_iterations (int, optional): Number of bootstrap iterations. Default is 1000.

        Returns:
            pd.DataFrame: Mean and standard deviation of Taleb's Kappa for each stock.
        """
        t_kappa = pd.DataFrame([], index=["t_kappa"], columns=self.tickers)
        for ticker in self.tickers:
            s_n, s_n0 = [], []
            for _ in range(num_iterations):
                s_n0.append(np.sum(np.random.choice(self.lin_returns[ticker], size=n0, replace=True)))
                s_n.append(np.sum(np.random.choice(self.lin_returns[ticker], size=n, replace=True)))
            s_n, s_n0 = np.array(s_n), np.array(s_n0)
            mad_n, mad_n0 = self.mad(s_n), self.mad(s_n0)
            t_kappa_value = 2 - (np.log(n / n0)) / (np.log(mad_n / mad_n0))
            t_kappa[ticker] = [t_kappa_value.round(3)]
        return t_kappa

    @staticmethod
    def mad(data: pd.Series | np.ndarray):
        """
        Computes the Mean Absolute Deviation (MAD) of a dataset.

        Args:
            data (pd.Series | np.ndarray): Input data.

        Returns:
            float: Mean absolute deviation.
        """
        return np.mean(np.abs(data - np.mean(data)))

    def plot_distribution_fits(self):
        """
        Plots four subplots showing distribution fits for stock returns:
        1. Histogram of returns with Power-law and Gaussian fits (linear y-axis).
        2. Histogram of returns with Lognormal and Gaussian fits (linear y-axis).
        3. Log-log plot of positive and negative returns (logarithmic y-axis).
        4. Histogram of returns with Kernel Density Estimate (KDE) and Gaussian fits (linear y-axis).

        Each subplot contains:
        - Histogram of returns (probability mass scale).
        - Fitted Gaussian distribution (normalized to probability mass scale).
        - Fitted advanced distribution (Power-law, Lognormal, or KDE, normalized to probability mass scale).
        """
        for ticker in self.tickers:
            lin_returns = self.lin_returns[ticker].dropna()
            plt.style.use('seaborn-whitegrid')  # Use a clean grid style
            bins = 100  # Use the same number of bins for all subplots

            # Create subplots without sharing the y-axis
            fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharey=False)
            fig.suptitle(f'Distribution Fits for {ticker}', fontsize=20)

            # Calculate histogram once (probability density)
            hist, bin_edges = np.histogram(lin_returns, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers
            bin_width = bin_edges[1] - bin_edges[0]  # Calculate bin width

            # Scale histogram by bin width to convert density to probability mass
            hist_scaled = hist * bin_width

            # Figure 1: Power-law (alpha, xmin)
            # Bar plot (probability mass scale)
            ax[0, 0].bar(bin_centers, hist_scaled, width=bin_width, color='lightblue', alpha=0.7)
            ax[0, 0].set_title('Histogram of Returns')
            ax[0, 0].set_ylabel('Probability Mass')  # Set y-axis label
            ax[0, 0].set_yscale('linear')  # Set y-axis scale to linear

            # Gaussian Fit (normalized to probability mass scale)
            x = bin_centers
            mu, std = norm.fit(lin_returns)
            gaussian_pdf = norm.pdf(x, mu, std)
            gaussian_pdf_scaled = gaussian_pdf * bin_width  # Scale by bin width
            ax[0, 0].plot(x, gaussian_pdf_scaled, 'b--', label='Gaussian Fit')
            ax[0, 0].legend(fontsize=16)

            # Power-law Fit (Positive and Negative, normalized to probability mass scale)
            fitted_powerlaw = self.fit_powerlaw()
            xmin_neg = -fitted_powerlaw.loc["left", ticker].power_law.xmin
            x_neg = x[x <= xmin_neg]
            powerlaw_pdf_neg = fitted_powerlaw.loc["left", ticker].power_law.pdf(np.abs(x_neg)) * bin_width  # Scale by bin width
            # Calculate fraction of data above xmin_pos
            frac_neg = np.sum(lin_returns <= xmin_neg) / len(lin_returns)
            # frac_neg = norm.cdf(xmin_neg, mu, std)
            powerlaw_pdf_neg = (powerlaw_pdf_neg / np.sum(
                powerlaw_pdf_neg) * frac_neg)
            ax[0, 0].plot(x_neg, powerlaw_pdf_neg, 'r-', linewidth=3, label='Power-law Fit (neg)')

            xmin_pos = fitted_powerlaw.loc["right", ticker].power_law.xmin
            x_pos = x[x >= xmin_pos]
            powerlaw_pdf_pos = fitted_powerlaw.loc["right", ticker].power_law.pdf(x_pos) * bin_width
            # Calculate fraction of data above xmin_pos
            frac_pos = np.sum(lin_returns >= xmin_pos) / len(lin_returns)
            # frac_pos = norm.cdf(-xmin_pos, mu, std)
            powerlaw_pdf_pos = powerlaw_pdf_pos / np.sum(
                powerlaw_pdf_pos) * frac_pos # Scale by bin width
            ax[0, 0].plot(x_pos, powerlaw_pdf_pos, 'b-', linewidth=3, label='Power-law Fit (pos)')
            ax[0, 0].legend(fontsize=16)

            # Figure 2: Lognormal (normalized to probability mass scale)
            # Bar plot (probability mass scale)
            ax[0, 1].bar(bin_centers, hist_scaled, width=bin_width, color='lightblue', alpha=0.7)
            ax[0, 1].set_title('Lognormal Fit')
            ax[0, 1].set_ylabel('Probability Mass')  # Set y-axis label
            ax[0, 1].set_yscale('linear')  # Set y-axis scale to linear

            # Gaussian Fit (normalized to probability mass scale)
            ax[0, 1].plot(x, gaussian_pdf_scaled, 'b--', label='Gaussian Fit')
            ax[0, 1].legend(fontsize=16)

            x_neg = x[x < 0]
            frac_neg = np.sum(lin_returns < 0) / len(lin_returns)
            s = fitted_powerlaw.loc["left", ticker].lognormal.sigma
            scale = np.exp(fitted_powerlaw.loc["left", ticker].lognormal.mu)
            lognormal_pdf_neg = lognorm.pdf(np.abs(x_neg), s=s, scale=scale) * bin_width  # scale by bin width
            lognormal_pdf_neg = lognormal_pdf_neg / np.sum(lognormal_pdf_neg) * frac_neg
            ax[0, 1].plot(x_neg, lognormal_pdf_neg, 'r-', linewidth=3, label='Lognormal Fit (neg)')
            ax[0, 1].legend()

            x_pos = x[x > 0]
            frac_pos = np.sum(lin_returns > 0) / len(lin_returns)
            s = fitted_powerlaw.loc["right", ticker].lognormal.sigma
            scale = np.exp(fitted_powerlaw.loc["right", ticker].lognormal.mu)
            lognormal_pdf_pos = lognorm.pdf(x_pos, s=s, scale=scale) * bin_width  # Scale by bin width
            lognormal_pdf_pos = lognormal_pdf_pos / np.sum(lognormal_pdf_pos) * frac_pos
            ax[0, 1].plot(x_pos, lognormal_pdf_pos, 'b-', linewidth=3, label='Lognormal Fit (pos)')
            ax[0, 1].legend(fontsize=16)

            # Figure 3: Log-log plot
            # Calculate the probability density function (PDF) using a histogram
            hist_neg, bin_edges_neg = np.histogram(np.abs(lin_returns[lin_returns < -xmin_neg]), bins=bins+10, density=True)
            hist_pos, bin_edges_pos = np.histogram(lin_returns[lin_returns > xmin_pos], bins=bins+10, density=True)
            bin_centers_neg = (bin_edges_neg[:-1] + bin_edges_neg[1:]) / 2
            bin_centers_pos = (bin_edges_pos[:-1] + bin_edges_pos[1:]) / 2
            bin_width_neg = bin_edges_neg[1] - bin_edges_neg[0]
            bin_width_pos = bin_edges_pos[1] - bin_edges_pos[0]

            # Set ax[1, 0] to logarithmic scale
            ax[1, 0].loglog(bin_centers_neg, hist_neg * bin_width_neg, 'ro', label='Left (neg)')
            ax[1, 0].loglog(bin_centers_pos, hist_pos * bin_width_pos, 'bo', label='Right (pos)')
            ax[1, 0].set_xscale('log')  # Ensure x-axis is logarithmic
            ax[1, 0].set_yscale('log')  # Ensure y-axis is logarithmic
            ax[1, 0].set_title('Log-Log Plot of Returns')
            ax[1, 0].set_ylabel('Probability Mass (log scale)')  # Set y-axis label
            ax[1, 0].legend(fontsize=16)

            # Figure 4: Kernel Density Estimate (KDE, normalized to probability mass scale)
            # Bar plot (probability mass scale)
            ax[1, 1].bar(bin_centers, hist_scaled, width=bin_width, color='lightblue', alpha=0.7)
            ax[1, 1].set_title('KDE Fit')
            ax[1, 1].set_ylabel('Probability Mass')  # Set y-axis label
            ax[1, 1].set_yscale('linear')  # Set y-axis scale to linear

            # Gaussian Fit (normalized to probability mass scale)
            ax[1, 1].plot(x, gaussian_pdf_scaled, 'r--', label='Gaussian Fit (Kurtosis=0)')

            # Compute KDE explicitly using scipy.stats.gaussian_kde
            kde = gaussian_kde(lin_returns)
            y_kde = kde(x)  # Compute KDE y-values
            y_kde_normalized = (y_kde / np.sum(y_kde)) # Scale by bin width
            # Plot the normalized KDE
            ax[1, 1].plot(x, y_kde_normalized, color='black', linewidth=3,
                          label=f'KDE Fit (Kurtosis = {float(self.kurtosis[ticker].round(2))})')
            ax[1, 1].legend(fontsize=16)

            plt.tight_layout()
        return fig, ax