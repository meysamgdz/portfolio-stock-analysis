import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from modules.FatTailStock import FatTailStock as fts


class ModernPortfolioTheory(fts):
    def __init__(self, data: pd.DataFrame, asset_type: str = "powerlaw"):
        """
        Initializes the ModernPortfolioTheory class.

        Args:
            data (pd.DataFrame): Stock price data with tickers as columns.
            asset_type (str): Type of asset return distribution.
                              Options: "normal", "powerlaw", "exponential".
        """
        super().__init__(data)
        self.asset_type = asset_type.lower()
        self.n_assets: int = self.data.shape[1]
        self.n_days: int = 252
        self.get_fat_tail_metrics()  # Compute α, lognormal σ, kurtosis etc.
        self.big_loss = 0.20
        self._data_test = None
        self.w_opt = None
        self._results = None
        self._lowest_risk_indx = None

        self._validate_inputs_mpt()

    def _validate_inputs_mpt(self) -> None:
        """Validate constructor inputs"""
        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            raise ValueError("Input data must be a non-empty DataFrame")
        valid_types = ["normal", "powerlaw"]
        if self.asset_type not in valid_types:
            raise ValueError(f"Invalid asset_type: {self.asset_type}")
        if self.data.isna().any().any():
            raise ValueError("Input data contains missing values.")
    # Getters
    @property
    def data_test(self):
        """Gets the variables value."""
        return self._data_test

    # Setters
    @data_test.setter
    def data_test(self, new_data):
        """Sets the variable to its new value.
        Args:
            new_data: Variable's new value.
        """
        if not isinstance(new_data, pd.DataFrame):
            raise ValueError("Data must be a Pandas DataFrame!")
        self._data_test = new_data

    def compute_mpt_params(self, n_portfolios: int = 10000) -> None:
        """
        Runs the appropriate portfolio simulation based on the selected asset type.

        Args:
            n_portfolios (int): Number of portfolios to simulate.
        """
        if self.asset_type == "powerlaw":
            self._results, self.w_opt, self._lowest_risk_indx = self.powerlaw_assets(n_portfolios)
        elif self.asset_type == "normal":
            self._results, self.w_opt, self._lowest_risk_indx = self.norm_assets(n_portfolios)
        else:
            raise ValueError(f"Invalid asset_type: {self.asset_type}")

    def sample_mixed_returns(self):
        """
        Simulates returns with Gaussian body + power-law loss/gain tails.

        Returns:
            np.ndarray: Simulated return time series.
        """
        mu = self.lin_returns.mean().values
        sigma = self.lin_returns.std().values
        alpha_left = np.array([item[0] for _, item in self.alpha.loc["left"].items()])
        x_min_left = np.array([item[1] for _, item in self.alpha.loc["left"].items()])
        alpha_right = np.array([item[0] for _, item in self.alpha.loc["right"].items()])
        x_min_right = np.array([item[1] for _, item in self.alpha.loc["right"].items()])

        # Sample Gaussian body
        returns = np.random.normal(loc=mu[:, np.newaxis], scale=sigma[:, np.newaxis], size=(len(mu), self.n_days))
        x_min_left = np.tile(x_min_left, (self.n_days, 1)).T
        alpha_left = np.tile(alpha_left, (self.n_days, 1)).T
        x_min_right = np.tile(x_min_right, (self.n_days, 1)).T
        alpha_right = np.tile(alpha_right, (self.n_days, 1)).T

        # Sample the power-law as if the whole body was powerlaw (left)
        u = np.random.uniform(0, 1, (len(mu), self.n_days))
        powerlaw_samples = x_min_left * (1 - u) ** (-1 / alpha_left)
        # Replace a fraction with power-law crashes (left)
        returns = np.where(returns < -x_min_left, -powerlaw_samples, returns)

        # Sample the power-law as if the whole body was powerlaw (right)
        # u = np.random.uniform(0, 1, (len(mu), self.n_days))
        # powerlaw_samples = x_min_right * (1 - u) ** (-1 / alpha_right)
        # Replace a fraction with power-law crashes (right)
        # returns = np.where(returns > x_min_right, powerlaw_samples, returns)

        return returns

    def norm_assets(self, n_portfolios: int = 10000, n_paths: int = 1000) -> tuple:
        """
        Simulates portfolios assuming uncorrelated Gaussian returns and variance as risk.

        Args:
            n_portfolios (int): Number of portfolios to simulate.
        """
        results = np.zeros((3, n_portfolios))
        w = np.random.dirichlet(np.ones(self.n_assets), n_portfolios)
        w /= np.tile(w.sum(axis=1), (self.n_assets, 1)).T
        for j in range(n_paths):
            returns = self.sample_mixed_returns()
            cov_mat = np.cov(returns)*self.n_days
            returns = (1 + returns).prod(axis=1) - 1
            return_tot = w @ returns
            vol = np.einsum('ki,ij,kj->k', w, cov_mat, w)**0.5
            sharpe = return_tot / vol

            results[0, :] += vol
            results[1, :] += return_tot
            results[2, :] += sharpe
        results /= n_paths
        lowest_risk_indx = np.argmin(results[0, :])
        return results, w[lowest_risk_indx, :], lowest_risk_indx

    def powerlaw_assets(self, n_portfolios: int = 10000, n_paths: int = 1000) -> tuple:
        """
        Simulates portfolios under power-law-distributed tail losses using log tail risk.

        Args:
            n_portfolios (int): Number of portfolios to simulate.
            big_loss (float): Large loss threshold (e.g. 20%).
        """
        mu = self.lin_returns.mean().values
        sigma = self.lin_returns.std().values
        alpha = np.array([item[0] for _, item in self.alpha.loc["left"].items()])
        x_min = np.array([item[1] for _, item in self.alpha.loc["left"].items()])

        # Changing variable to fit the definition of the book [Theory of finance, jean-philippe-bouchaud]
        # In the book notation: A = x_min, mu = alpha-1.
        mu_eff, indx = alpha.min() - 1, np.argmin(alpha)  # Smallest exponents dominates when x->\inf
        A_pow_mu = x_min**mu_eff

        # Scale the tail for the total (Gaussian + powerlaw) to sum to 1
        tail_tot_prob = norm.cdf(x=-x_min[indx], loc=mu, scale=sigma)
        A_pow_mu *= tail_tot_prob

        # Tail covariance matrix
        tail_cov = self.compute_tail_covariance(mu_eff)

        results = np.zeros((3, n_portfolios))
        w = np.random.dirichlet(np.ones(self.n_assets), n_portfolios)
        w /= np.tile(w.sum(axis=1), (self.n_assets, 1)).T
        for j in range(n_paths):
            returns = self.sample_mixed_returns()
            returns = (1 + returns).prod(axis=1) - 1
            return_tot = w @ returns

            uncorrelated_term = np.sum(w ** mu_eff * A_pow_mu, axis=1)
            correlated_term = np.einsum('ki,ij,kj->k', (w ** mu_eff) ** 0.5, tail_cov, (w ** mu_eff) ** 0.5)
            total_tail_risk = np.log(uncorrelated_term + correlated_term)
            tail_scores = total_tail_risk - mu_eff * np.log(self.big_loss)

            results[0, :] += total_tail_risk
            results[1, :] += return_tot
            results[2, :] += tail_scores
        results /= n_paths
        lowest_risk_indx = np.argmin(results[2, :])

        return results, w[lowest_risk_indx, :], lowest_risk_indx

    def compute_tail_covariance(self, mu_eff: float) -> np.ndarray:
        """
        Estimates the tail covariance matrix based on Bouchaud's formulation (§ 11.1.5).

        Args:
            mu_eff (float): Effective tail exponent (alpha_min - 1)

        Returns:
            np.ndarray: Tail covariance matrix
        """
        # Extreme events: left-tail thresholding
        x_min = np.array([item[1] for _, item in self.alpha.loc["left"].items()])
        extreme_mask = self.lin_returns.lt(-x_min).T  # Shape: (n_assets, n_days)

        # Indicator for tail events (1 if in tail, 0 otherwise)
        tail_events = extreme_mask.astype(float)

        # Weighting by power-law scaling
        A_mu = x_min ** mu_eff
        scale = np.outer(A_mu, A_mu)

        # Compute co-occurrence matrix of extreme events
        co_events = np.cov(tail_events)
        C_t = co_events * scale  # Elementwise scaling

        return C_t

    def compute_test_return(self, data_test: pd.DataFrame) -> float:
        """
        Computes the potential return on the test data using the w computed from the train data.

        Args:
            data_test: The test data

        Return:
            A float denoting the potetial return.
        """
        if not isinstance(data_test, pd.DataFrame):
            raise ValueError("Test data must be a DataFrame.")
        if set(data_test.columns) != set(self.data.columns):
            raise ValueError("Test data columns must match training data.")

        self.data_test = data_test
        returns = self.data_test.pct_change().dropna()
        returns = (1 + returns).prod(axis=0) - 1
        return_tot = self.w_opt @ returns

        return return_tot
    def plot_frontier(self) -> tuple:
        """
        Utility function to plot the efficient frontier and best portfolio.
        """
        print(f"\nBest Portfolio:")
        print(f"Return: {self._results[1, self._lowest_risk_indx]:.2f}")
        if self.asset_type == "powerlaw":
            print(f"Risk = Logarithm of loss probability ≥ {self.big_loss*100} % : {self._results[0, self._lowest_risk_indx]:.2f}")
            xlabel = r"Tail Risk $log(∑ wᵢ^μ Aᵢ^μ)$"
            colorbar = r"Return-risk trade-off (tail risk - $log(l^μ)$)"
        else:
            print(f"Risk = Standard deviation: {self._results[0, self._lowest_risk_indx]:.2f}")
            big_loss_prob = norm.cdf(x=self.big_loss, loc=self._results[1, self._lowest_risk_indx],
                                     scale=self._results[0, self._lowest_risk_indx])
            print(f"Risk = Logarithm of loss probability ≥ {self.big_loss * 100} % : {np.log(big_loss_prob):.2f}")
            xlabel = "Risk (standard deviation)"
            colorbar = "Return-risk trade-off (Sharpe)"
        print(f"Weights: {np.round(self.w_opt, 3)}")

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        sc = ax.scatter(self._results[0, :], self._results[1, :], c=self._results[2, :], cmap="plasma", alpha=0.6)
        plt.colorbar(sc, label=colorbar)
        plt.xlabel(xlabel)
        plt.ylabel("Annual Return")
        plt.title("Efficient Frontier for Different Portfolios")
        plt.grid(True)

        return fig, ax
