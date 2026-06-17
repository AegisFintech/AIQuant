from setuptools import setup, find_packages

setup(
    name="aiquant",
    version="0.2.0",
    description="High-Frequency Statistical Arbitrage Framework for Crypto",
    author="AegisFintech",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "backtrader>=1.9.78.123",
        "ccxt>=4.0.0",
        "hyperliquid-python>=0.1.1",
        "eth_account>=0.8.0",
        "statsmodels>=0.14.0",
        "matplotlib>=3.7.0",
        "joblib>=1.2.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "pyarrow>=14.0.0",
        "fastparquet>=2023.10.0"
    ],
    python_requires=">=3.9",
)
