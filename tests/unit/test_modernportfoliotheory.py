import pytest
import numpy as np
import pandas as pd
from modules.ModernPortfolioTheory import ModernPortfolioTheory

class TestModernPortfolioTheory:

    @pytest.fixture
    def sample_data(self):
        """
        Fixture that generates synthetic daily price data for 3 assets over 100 days.
        Simulates normally distributed returns.
        """
        dates = pd.date_range(start='2020-01-01', periods=100)
        prices = pd.DataFrame({
            'AAPL': np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
            'MSFT': np.cumprod(1 + np.random.normal(0.0008, 0.018, 100)),
            'GOOG': np.cumprod(1 + np.random.normal(0.0012, 0.025, 100))
        }, index=dates)
        return prices

    @pytest.fixture
    def test_data(self):
        """
        Fixture that generates a second set of synthetic price data for out-of-sample testing.
        """
        dates = pd.date_range(start='2021-01-01', periods=50)
        prices = pd.DataFrame({
            'AAPL': np.cumprod(1 + np.random.normal(0.001, 0.02, 50)),
            'MSFT': np.cumprod(1 + np.random.normal(0.0008, 0.018, 50)),
            'GOOG': np.cumprod(1 + np.random.normal(0.0012, 0.025, 50))
        }, index=dates)
        return prices

    def test_initialization(self, sample_data):
        """
        Test that initialization of ModernPortfolioTheory succeeds with valid input,
        sets attributes correctly, and fails for unsupported asset types.
        """
        mpt = ModernPortfolioTheory(sample_data)
        assert mpt.data.equals(sample_data)
        assert mpt.asset_type == "powerlaw"
        assert mpt.n_assets == 3
        assert mpt.n_days == 252

        mpt_normal = ModernPortfolioTheory(sample_data, "normal")
        assert mpt_normal.asset_type == "normal"

        with pytest.raises(ValueError):
            ModernPortfolioTheory(sample_data, "invalid_type")

    def test_data_test_property(self, sample_data, test_data):
        """
        Test assignment to `.data_test` property and validate input type.
        """
        mpt = ModernPortfolioTheory(sample_data)
        mpt.data_test = test_data
        assert mpt.data_test.equals(test_data)

        with pytest.raises(ValueError):
            mpt.data_test = "not a dataframe"

    def test_compute_mpt_params(self, sample_data):
        """
        Test the core method for computing modern portfolio theory parameters.
        Validates shapes and types of result attributes.
        """
        mpt_normal = ModernPortfolioTheory(sample_data, "normal")
        mpt_normal.compute_mpt_params(n_portfolios=100)
        assert mpt_normal._results.shape == (3, 100)
        assert mpt_normal.w_opt.shape == (3,)
        assert isinstance(mpt_normal._lowest_risk_indx, (int, np.integer))

    def test_sample_mixed_returns(self, sample_data):
        """
        Test the method for generating a matrix of mixed (simulated) returns for optimization.
        """
        mpt = ModernPortfolioTheory(sample_data)
        mpt.compute_mpt_params(n_portfolios=10)
        returns = mpt.sample_mixed_returns()
        assert isinstance(returns, np.ndarray)
        assert returns.shape == (3, 252)

    def test_norm_assets(self, sample_data):
        """
        Test the method for normal distribution-based asset simulation and optimization.
        Validate result shapes and that weights sum to 1.
        """
        mpt = ModernPortfolioTheory(sample_data, "normal")
        results, w_opt, idx = mpt.norm_assets(n_portfolios=50, n_paths=5)
        assert results.shape == (3, 50)
        assert np.allclose(w_opt.sum(), 1.0)
        assert 0 <= idx < 50

    def test_powerlaw_assets(self, sample_data):
        """
        Test the method for simulating power-law-distributed assets.
        Ensure portfolio optimization still succeeds.
        """
        mpt = ModernPortfolioTheory(sample_data)
        results, w_opt, idx = mpt.powerlaw_assets(n_portfolios=50, n_paths=5)
        assert results.shape == (3, 50)
        assert np.allclose(w_opt.sum(), 1.0)
        assert 0 <= idx < 50

    def test_compute_test_return(self, sample_data, test_data):
        """
        Test the method that computes the return of the optimized portfolio on test data.
        Validates correct shape, types, and error conditions.
        """
        mpt = ModernPortfolioTheory(sample_data)
        mpt.compute_mpt_params(n_portfolios=100)
        test_data = test_data[mpt.data.columns]  # Ensure aligned columns
        test_return = mpt.compute_test_return(test_data)
        assert isinstance(test_return, float)

        with pytest.raises(ValueError):
            mpt.compute_test_return("invalid_data")

        with pytest.raises(ValueError):
            bad_test_data = test_data.drop(columns=['AAPL'])
            mpt.compute_test_return(bad_test_data)

    def test_plot_frontier(self, sample_data):
        """
        Test the efficient frontier plotting method for both normal and powerlaw types.
        Ensure valid matplotlib figure and axis objects are returned.
        """
        mpt_normal = ModernPortfolioTheory(sample_data, "normal")
        mpt_normal.compute_mpt_params(n_portfolios=100)
        fig, ax = mpt_normal.plot_frontier()
        assert fig is not None and ax is not None

        mpt_powerlaw = ModernPortfolioTheory(sample_data)
        mpt_powerlaw.compute_mpt_params(n_portfolios=100)
        fig, ax = mpt_powerlaw.plot_frontier()
        assert fig is not None and ax is not None

    def test_single_asset(self):
        """
        Validate handling of a single-asset portfolio. The optimal weight should be 1.0.
        """
        dates = pd.date_range(start='2020-01-01', periods=100)
        prices = pd.DataFrame({'AAPL': np.cumprod(1 + np.random.normal(0.001, 0.02, 100))}, index=dates)
        mpt = ModernPortfolioTheory(prices)
        mpt.compute_mpt_params(n_portfolios=10)
        assert mpt.w_opt.shape == (1,)
        assert np.isclose(mpt.w_opt[0], 1.0)

    def test_empty_data(self):
        """
        Ensure that passing an empty DataFrame raises a ValueError during initialization.
        """
        with pytest.raises(ValueError):
            ModernPortfolioTheory(pd.DataFrame())

    def test_constant_price_data(self):
        """
        Ensure that constant price series (zero returns) are rejected.
        """
        dates = pd.date_range(start='2020-01-01', periods=100)
        prices = pd.DataFrame({'AAPL': [100] * 100, 'MSFT': [200] * 100}, index=dates)
        with pytest.raises(ValueError):
            ModernPortfolioTheory(prices)
