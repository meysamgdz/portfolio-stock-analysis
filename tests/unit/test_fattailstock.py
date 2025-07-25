import pytest
import numpy as np
import pandas as pd
from scipy.stats import pareto
from modules.FatTailStock import FatTailStock


class TestFatTailStock:
    @pytest.fixture
    def realistic_data(self):
        """
        Fixture that generates realistic, fat-tailed stock price data for two assets (AAPL, MSFT).
        Returns a price DataFrame with 252 trading days.
        """
        dates = pd.date_range(start='2020-01-01', periods=252)
        returns = pd.DataFrame({
            'AAPL': self._generate_fat_tailed_returns(252, base_vol=0.01),
            'MSFT': self._generate_fat_tailed_returns(252, base_vol=0.008),
        }, index=dates)
        prices = (1 + returns).cumprod() * 100
        return prices

    def _generate_fat_tailed_returns(self, n, base_vol=0.01):
        """
        Helper method to generate returns with fat tails using a normal base and occasional Pareto-distributed tail events.
        """
        returns = np.random.normal(0.0005, base_vol, n)
        tail_events = np.random.rand(n) < 0.05
        returns[tail_events] += pareto.rvs(2.5, size=tail_events.sum()) * base_vol * 5 * np.random.choice([-1, 1], tail_events.sum())
        return returns

    @pytest.fixture
    def edge_cases(self):
        """
        Fixture providing a dictionary of edge case DataFrames:
        - Empty DataFrame
        - Single-row DataFrame
        - Contains negative prices
        - Contains non-numeric data
        - Constant price data
        """
        dates = pd.date_range(start='2020-01-01', periods=5)
        return {
            'empty': pd.DataFrame(),
            'single_row': pd.DataFrame({'A': [100]}, index=[dates[0]]),
            'negative': pd.DataFrame({'A': [100, 95, -90, 105, 110]}, index=dates),
            'non_numeric': pd.DataFrame({'A': ['100', '95', 'A', '105', '110']}, index=dates),
            'constant': pd.DataFrame({'A': [100] * 5}, index=dates),
        }

    @pytest.mark.parametrize("case, expected_exception", [
        ('empty', ValueError),
        ('single_row', ValueError),
        ('negative', ValueError),
        ('non_numeric', TypeError),
        ('constant', ValueError),
    ])
    def test_initialization_failures(self, edge_cases, case, expected_exception):
        """
        Parametrized test to ensure FatTailStock raises ValueError for invalid edge case inputs.
        """
        with pytest.raises(expected_exception):
            FatTailStock(edge_cases[case])

    def test_successful_initialization(self, realistic_data):
        """
        Ensure that the FatTailStock object can be initialized correctly with valid price data.
        """
        fts = FatTailStock(realistic_data)
        assert fts.data.equals(realistic_data)

    def test_property_setters(self, realistic_data):
        """
        Test the `.data` property setter for proper validation and assignment.
        Should accept valid DataFrames and reject invalid types.
        """
        fts = FatTailStock(realistic_data[['AAPL']])
        new_data = realistic_data[['MSFT']]
        fts.data = new_data
        assert fts.data.equals(new_data)

        with pytest.raises(ValueError):
            fts.data = "invalid_data"

    def test_returns_calculations(self, realistic_data):
        """
        Test linear return computation via `.get_lin_returns()`.
        Check shape and ensure no NaN values are present.
        """
        fts = FatTailStock(realistic_data)
        lin_returns = fts.get_lin_returns()
        assert lin_returns.shape == (251, 2)
        assert not lin_returns.isna().any().any()

    def test_fat_tail_metrics(self, realistic_data):
        """
        Test the calculation of fat-tail metrics (alpha and kurtosis).
        Checks type, structure, and basic validity of results.
        """
        fts = FatTailStock(realistic_data)
        fts.get_fat_tail_metrics()
        assert isinstance(fts.alpha, pd.DataFrame)
        assert "left" in fts.alpha.index
        assert "right" in fts.alpha.index
        assert all(isinstance(v, tuple) and len(v) == 2 for v in fts.alpha.iloc[0])
        assert isinstance(fts.kurtosis, pd.Series)

    def test_value_at_risk(self, realistic_data):
        """
        Test computation of Value at Risk (VaR).
        Check the shape of the output and ensure negative values (representing losses).
        """
        fts = FatTailStock(realistic_data)
        var_95 = fts.value_at_risk()
        assert var_95.shape == (1, 2)
        assert (var_95.values < 0).all()

    def test_taleb_kappa(self, realistic_data):
        """
        Test Taleb's fragility (kappa) metric with large sample simulation.
        Validate type, finite values, and result bounds.
        """
        np.random.seed(42) # removing this may lead to failure of the test
        fts = FatTailStock(realistic_data)
        t_kappa = fts.taleb_kappa(n0=100, n=1000, num_iterations=10000)

        assert isinstance(t_kappa, pd.DataFrame)
        assert t_kappa.shape[1] == realistic_data.shape[1]
        assert not t_kappa.isnull().values.any()
        assert np.all(np.isfinite(t_kappa.values))
        assert ((t_kappa.values > -20) & (t_kappa.values < 20)).all()

    def test_compare_distributions(self, realistic_data):
        """
        Test comparison between power-law and alternative distributions.
        Ensure result is a formatted string with expected components.
        """
        fts = FatTailStock(realistic_data)
        comparison = fts.compare_powerlaw(dist_list=["lognormal"])
        assert isinstance(comparison, str)
        assert "likelihood_ratio" in comparison
        assert "p-value" in comparison

    def test_mad(self):
        """
        Test the static method `mad` (median absolute deviation) on a small data array.
        """
        data = np.array([1, 2, 3, 4, 5])
        expected = 1.2
        assert np.isclose(FatTailStock.mad(data), expected, atol=1e-6)

    def test_cov_matrix(self, realistic_data):
        """
        Test covariance matrix calculation.
        Ensure symmetry and correct shape for two-asset input.
        """
        fts = FatTailStock(realistic_data)
        cov = fts.get_cov_matrix()
        assert cov.shape == (2, 2)
        assert np.allclose(cov.values, cov.values.T, atol=1e-6)
