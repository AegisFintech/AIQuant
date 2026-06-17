#!/usr/env/bin python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiquant.data.fetcher import DataFetcher
from aiquant.features.technical import generate_all_features
from aiquant.strategies.mean_reversion import MeanReversionStrategy
from aiquant.backtest.engine import VectorBTBacktester

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting AIQuant Pipeline")
    
    # 1. Fetch Data
    fetcher = DataFetcher()
    # Fetch a smaller subset for quick testing
    df = fetcher.fetch_ohlcv(symbol='BTC/USDT', timeframe='1h', since_date='2023-01-01', limit=1000)
    logger.info(f"Data fetched: {df.shape}")
    
    # 2. Feature Engineering
    logger.info("Generating features...")
    df_features = generate_all_features(df)
    logger.info(f"Features generated: {df_features.shape}")
    
    # 3. Strategy Signals
    logger.info("Generating signals...")
    strategy = MeanReversionStrategy()
    signals = strategy.generate_signals(df_features)
    logger.info(f"Signals generated. Longs: {(signals == 1).sum()}, Shorts: {(signals == -1).sum()}")
    
    # 4. Backtest
    logger.info("Running backtest...")
    backtester = VectorBTBacktester()
    portfolio = backtester.run(df_features['close'], signals)
    
    # 5. Results
    stats = backtester.get_stats(portfolio)
    logger.info("\n=== Backtest Statistics ===")
    print(stats)
    
    # Save plot
    plot_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backtest_results.html')
    backtester.plot_results(portfolio, save_path=plot_path)
    logger.info(f"Plot saved to {plot_path}")
    
if __name__ == "__main__":
    main()
