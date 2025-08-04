import pytest
import numpy as np
import pandas as pd

from modules.FatTailStock import FatTailStock
from modules.ModernPortfolioTheory import ModernPortfolioTheory
@pytest.fixture
def realistic_data():
    """Simulated fat-tailed price data for 3 assets."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=252)

    def fat_tailed_returns(n, base_vol=0.01):
        returns = np.random.normal(0.0005, base_vol, n)
        tail_events = np.random.rand(n) < 0.05
        from scipy.stats import pareto
        returns[tail_events] += pareto.rvs(2.5, size=tail_events.sum()) * base_vol * 5 * np.random.choice([-1, 1],
                                                                                                          tail_events.sum())
        return returns

    returns = pd.DataFrame({
        'AAPL': fat_tailed_returns(252, 0.012),
        'MSFT': fat_tailed_returns(252, 0.010),
        'GOOG': fat_tailed_returns(252, 0.014),
    }, index=dates)

    prices = (1 + returns).cumprod() * 100
    return prices


def test_end_to_end_fattail_and_mpt(realistic_data):
    """Integration test from fat-tail analysis to portfolio optimization."""
    fts = FatTailStock(realistic_data)
    lin_returns = fts.get_lin_returns()

    assert lin_returns.shape[0] == 251
    assert not lin_returns.isna().any().any()

    fts.get_fat_tail_metrics()
    assert isinstance(fts.alpha, pd.DataFrame)
    assert isinstance(fts.kurtosis, pd.Series)

    mpt = ModernPortfolioTheory(realistic_data)
    mpt.compute_mpt_params(n_portfolios=100)

    assert mpt._results.shape == (3, 100)
    assert mpt.w_opt.shape == (3,)
    assert np.isclose(mpt.w_opt.sum(), 1.0)

    future_returns = pd.DataFrame({
        'AAPL': (1 + np.random.normal(0.001, 0.015, 30)).cumprod() * 120,
        'MSFT': (1 + np.random.normal(0.0008, 0.012, 30)).cumprod() * 110,
        'GOOG': (1 + np.random.normal(0.0012, 0.017, 30)).cumprod() * 130
    }, index=pd.date_range(start='2021-01-01', periods=30))

    test_return = mpt.compute_test_return(future_returns)
    assert isinstance(test_return, float)
    assert -1.0 < test_return < 2.0


def test_powerlaw_optimization(realistic_data):
    """Specifically tests powerlaw asset type integration"""
    fts = FatTailStock(realistic_data)
    fts.get_fat_tail_metrics()

    mpt = ModernPortfolioTheory(realistic_data, asset_type="powerlaw")
    mpt.compute_mpt_params(n_portfolios=3000)

    assert mpt.asset_type == "powerlaw"
    assert mpt.big_loss == 0.20


def test_normal_optimization(realistic_data):
    """Tests normal distribution mode integration"""
    mpt = ModernPortfolioTheory(realistic_data, asset_type="normal")
    mpt.compute_mpt_params(n_portfolios=3000)

    assert mpt.asset_type == "normal"
    fig, ax = mpt.plot_frontier()
    assert "standard deviation" in ax.get_xlabel().lower()


def test_fattail_mpt_pipeline_plot_frontier(realistic_data):
    """End-to-end test including MPT plotting after fat-tail analysis."""
    mpt = ModernPortfolioTheory(realistic_data)
    mpt.compute_mpt_params(n_portfolios=100)
    fig, ax = mpt.plot_frontier()

    assert fig is not None
    assert ax is not None


def test_compare_fat_tail_vs_normal_mpt(realistic_data):
    """Compare powerlaw vs normal asset assumptions."""
    mpt_normal = ModernPortfolioTheory(realistic_data, asset_type="normal")
    mpt_powerlaw = ModernPortfolioTheory(realistic_data, asset_type="powerlaw")

    mpt_normal.compute_mpt_params(n_portfolios=50)
    mpt_powerlaw.compute_mpt_params(n_portfolios=50)

    assert mpt_normal._results.shape == mpt_powerlaw._results.shape
    assert not np.array_equal(mpt_normal._results, mpt_powerlaw._results)
    assert not np.array_equal(mpt_normal.w_opt, mpt_powerlaw.w_opt)


def test_data_pipeline_with_missing_values():
    """Test that NaN values raise error in pipeline."""
    dates = pd.date_range(start="2020-01-01", periods=10)
    bad_data = pd.DataFrame({
        "AAPL": [100, 101, np.nan, 103, 105, 104, np.nan, 107, 108, 110],
        "MSFT": [200, 201, 202, np.nan, 205, 206, 207, 208, np.nan, 210]
    }, index=dates)

    with pytest.raises(ValueError, match="missing values"):
        FatTailStock(bad_data)


def test_weight_calculation_consistency(realistic_data):
    """Verifies weight calculations are consistent between runs"""
    np.random.seed(42)
    results = []
    for _ in range(5):
        mpt = ModernPortfolioTheory(realistic_data)
        mpt.compute_mpt_params(n_portfolios=2000)
        results.append(mpt.w_opt)

    avg_weights = np.mean(results, axis=0)
    for weights in results:
        assert np.allclose(weights, avg_weights, rtol=0.25)
