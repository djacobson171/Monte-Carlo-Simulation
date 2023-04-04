import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime
from pandas_datareader import data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class MonteCarloPortfolioOptimization:
    """
    Monte Carlo simulation tool for portfolio optimization using the Black-Litterman model
    """

    def __init__(self, risk_free_rate=0.02, num_simulations=10000, num_assets=10, market_ticker='^GSPC'):
        self.risk_free_rate = risk_free_rate
        self.num_simulations = num_simulations
        self.num_assets = num_assets
        self.market_ticker = market_ticker
        self.stock_returns = None
        self.cov_matrix = None
        self.market_returns = None
        self.market_capitalization = None
        self.equilibrium_weights = None
        self.bl_cov_matrix = None
        self.bl_expected_returns = None

    def _load_data(self, start_date, end_date):
        """
        Load historical stock prices and calculate daily returns
        """
        tickers = pd.read_csv('tickers.csv', header=None, encoding='utf-8').iloc[:self.num_assets, 0].tolist()
        prices = data.DataReader(tickers, 'yahoo', start_date, end_date)['Adj Close']
        returns = prices.pct_change().dropna()
        self.stock_returns = np.array(returns)

    def _calculate_covariance_matrix(self):
        """
        Calculate the covariance matrix of stock returns
        """
        self.cov_matrix = np.cov(self.stock_returns.T)

    def _calculate_market_parameters(self):
        """
        Calculate the market returns and capitalization
        """
        market_prices = data.DataReader(self.market_ticker, 'yahoo', start_date, end_date)['Adj Close']
        market_returns = market_prices.pct_change().dropna()
        self.market_returns = np.array(market_returns)
        self.market_capitalization = np.sum(market_prices.iloc[-1] * self.equilibrium_weights)

    def _calculate_equilibrium_weights(self):
        """
        Calculate the equilibrium weights using the market capitalization
        """
        self.equilibrium_weights = np.array(pd.read_csv('market_capitalization.csv', header=None).iloc[:self.num_assets, 1]) / self.market_capitalization

    def _run_pca(self):
        """
        Run PCA on the covariance matrix to reduce the number of factors
        """
        pca = PCA(n_components=self.num_assets)
        standardized_returns = StandardScaler().fit_transform(self.stock_returns)
        pca.fit(standardized_returns)
        self.cov_matrix = pca.get_covariance()

    def _calculate_bl_parameters(self):
        """
        Calculate the Black-Litterman parameters
        """
        market_prior = self.market_capitalization * self.equilibrium_weights
        tau = 0.025
        delta = 2.5
        sigma = tau * self.cov_matrix
        omega = np.diag(np.diag(self.equilibrium_weights.dot(sigma).dot(self.equilibrium_weights.T)))
        pi = delta * np.dot(self.cov_matrix, self.equilibrium_weights)
        inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        self.bl_cov_matrix = np.linalg.inv(inv_cov_matrix + np.dot(np.dot(self.equilibrium_weights.T, np.linalg.inv(omega)), self.equilibrium_weights))
        self.bl_expected_returns = np.dot(self.bl_cov_matrix, np.dot(inv_cov_matrix, pi) + np.dot(np.dot(self.equilibrium_weights.T, np.linalg.inv(omega)), self.market_returns - self.risk_free_rate))

    def _simulate_portfolio_returns(self):
        """
        Simulate portfolio returns using Monte Carlo simulation
        """
        self.portfolio_returns = np.zeros((self.num_simulations, self.num_assets))
        for i in range(self.num_simulations):
            random_returns = np.random.multivariate_normal(self.bl_expected_returns, self.bl_cov_matrix)
            self.portfolio_returns[i] = random_returns

    def _optimize_portfolio(self):
        """
        Optimize the portfolio using the mean-variance optimization model
        """
        def objective_function(weights):
            """
            Objective function for mean-variance optimization
            """
            returns = np.dot(self.portfolio_returns, weights)
            variance = np.dot(np.dot(weights.T, self.bl_cov_matrix), weights)
            return -returns / np.sqrt(variance)

        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for i in range(self.num_assets))
        initial_guess = self.equilibrium_weights
        result = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        self.optimized_weights = result.x

    def run(self, start_date, end_date):
        """
        Run the Monte Carlo simulation tool
        """
        # Load data and calculate parameters
        self._load_data(start_date, end_date)
        self._calculate_covariance_matrix()
        self._calculate_market_parameters()
        self._calculate_equilibrium_weights()

        # Run PCA to reduce the number of factors
        self._run_pca()

        # Calculate Black-Litterman parameters
        self._calculate_bl_parameters()

        # Simulate portfolio returns
        self._simulate_portfolio_returns()

        # Optimize the portfolio
        self._optimize_portfolio()

        # Print results
        print("Optimized weights:", self.optimized_weights)
        print("Expected return:", np.dot(self.optimized_weights, self.bl_expected_returns))
        print("Volatility:", np.sqrt(np.dot(np.dot(self.optimized_weights.T, self.bl_cov_matrix), self.optimized_weights)))    

mc = MonteCarloPortfolioOptimization()
mc.run(start_date='2020-01-01', end_date='2020-12-31')

