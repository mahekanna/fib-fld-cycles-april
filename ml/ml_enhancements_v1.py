import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Import needed core components
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.market_regime_detector import MarketRegimeDetector
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Check for TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr

# Try to import optional deep learning libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, save_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


class FeatureEngineer:
    """
    Feature engineering for market data to prepare for ML models.
    Creates technical indicators and transforms data into ML-ready format.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the feature engineer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default parameters
        self.ta_periods = self.config.get('ta_periods', [5, 10, 21, 55, 89])
        self.sequence_length = self.config.get('sequence_length', 20)
        self.include_cycle_features = self.config.get('include_cycle_features', True)
        self.include_volume_features = self.config.get('include_volume_features', True)
        self.include_regime_features = self.config.get('include_regime_features', True)
        
        # Initialize scalers
        self.price_scaler = None
        self.feature_scaler = None
        self.target_scaler = None
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and features from raw price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical features
        """
        # Create a copy to avoid modifying original
        data = df.copy()
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            self.logger.error("Missing required price columns")
            return df
        
        # Add volume features if available and requested
        if 'volume' in data.columns and self.include_volume_features:
            self._add_volume_features(data)
        
        # Add basic price features
        self._add_price_features(data)
        
        # Add moving averages
        self._add_moving_averages(data)
        
        # Add momentum indicators
        self._add_momentum_indicators(data)
        
        # Add volatility indicators
        self._add_volatility_indicators(data)
        
        # Add cycle-based features if requested
        if self.include_cycle_features:
            self._add_cycle_features(data)
        
        # Add market regime features if requested
        if self.include_regime_features:
            self._add_regime_features(data)
        
        # Remove rows with NaN values
        data = data.dropna()
        
        return data
    
    def prepare_sequence_data(self, 
                            features: pd.DataFrame,
                            target_column: str,
                            sequence_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequence data for time series models like LSTM.
        
        Args:
            features: DataFrame with feature columns
            target_column: Column name for the target variable
            sequence_length: Length of sequences (window size)
            
        Returns:
            Tuple of (X_sequences, y_targets) as numpy arrays
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        # Extract target
        if target_column in features.columns:
            y = features[target_column].values
            X = features.drop(columns=[target_column])
        else:
            self.logger.warning(f"Target column {target_column} not found, using all columns as features")
            X = features
            y = np.zeros(len(features))
        
        feature_columns = X.columns
        X = X.values
        
        # Standardize features if not already done
        if self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            X = self.feature_scaler.fit_transform(X)
        else:
            X = self.feature_scaler.transform(X)
        
        # Prepare sequences
        X_sequences = []
        y_targets = []
        
        for i in range(len(X) - sequence_length):
            X_sequences.append(X[i:i+sequence_length])
            y_targets.append(y[i+sequence_length])
        
        return np.array(X_sequences), np.array(y_targets), feature_columns
    
    def prepare_classification_data(self, 
                                  features: pd.DataFrame,
                                  lookahead: int = 5,
                                  threshold_pct: float = 1.0) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for classification models by creating target labels.
        
        Args:
            features: DataFrame with feature columns
            lookahead: Number of bars to look ahead for price change
            threshold_pct: Threshold percentage for significant price change
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Create a copy to avoid modifying original
        data = features.copy()
        
        # Create future return column
        data['future_return'] = data['close'].pct_change(lookahead).shift(-lookahead) * 100
        
        # Create target label based on threshold
        data['target'] = 0  # neutral by default
        data.loc[data['future_return'] >= threshold_pct, 'target'] = 1  # bullish
        data.loc[data['future_return'] <= -threshold_pct, 'target'] = -1  # bearish
        
        # Drop NaN values
        data = data.dropna()
        
        # Extract target column
        y = data['target']
        
        # Remove target and future columns from features
        X = data.drop(columns=['target', 'future_return'])
        
        return X, y
    
    def prepare_regression_data(self, 
                              features: pd.DataFrame,
                              lookahead: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for regression models by creating target values.
        
        Args:
            features: DataFrame with feature columns
            lookahead: Number of bars to look ahead for price prediction
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Create a copy to avoid modifying original
        data = features.copy()
        
        # Create future price column
        if 'close' in data.columns:
            future_price = data['close'].shift(-lookahead)
            
            # Create target as percentage change
            data['target'] = (future_price - data['close']) / data['close'] * 100
        else:
            self.logger.error("Close price column not found")
            return data, pd.Series()
        
        # Drop NaN values
        data = data.dropna()
        
        # Extract target column
        y = data['target']
        
        # Remove target column from features
        X = data.drop(columns=['target'])
        
        return X, y
    
    def normalize_features(self, 
                         features: pd.DataFrame, 
                         fit: bool = True) -> pd.DataFrame:
        """
        Normalize features using StandardScaler.
        
        Args:
            features: DataFrame with feature columns
            fit: Whether to fit the scaler or use existing one
            
        Returns:
            DataFrame with normalized features
        """
        # Create a copy to avoid modifying original
        data = features.copy()
        
        # Initialize scaler if needed
        if fit or self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            scaled_values = self.feature_scaler.fit_transform(data)
        else:
            scaled_values = self.feature_scaler.transform(data)
        
        # Convert back to DataFrame
        scaled_df = pd.DataFrame(scaled_values, index=data.index, columns=data.columns)
        
        return scaled_df
    
    def save_scalers(self, directory: str) -> None:
        """
        Save fitted scalers to disk.
        
        Args:
            directory: Directory to save scalers
        """
        os.makedirs(directory, exist_ok=True)
        
        if self.feature_scaler is not None:
            joblib.dump(self.feature_scaler, os.path.join(directory, 'feature_scaler.joblib'))
        
        if self.price_scaler is not None:
            joblib.dump(self.price_scaler, os.path.join(directory, 'price_scaler.joblib'))
        
        if self.target_scaler is not None:
            joblib.dump(self.target_scaler, os.path.join(directory, 'target_scaler.joblib'))
    
    def load_scalers(self, directory: str) -> None:
        """
        Load fitted scalers from disk.
        
        Args:
            directory: Directory to load scalers from
        """
        feature_scaler_path = os.path.join(directory, 'feature_scaler.joblib')
        price_scaler_path = os.path.join(directory, 'price_scaler.joblib')
        target_scaler_path = os.path.join(directory, 'target_scaler.joblib')
        
        if os.path.exists(feature_scaler_path):
            self.feature_scaler = joblib.load(feature_scaler_path)
        
        if os.path.exists(price_scaler_path):
            self.price_scaler = joblib.load(price_scaler_path)
        
        if os.path.exists(target_scaler_path):
            self.target_scaler = joblib.load(target_scaler_path)
    
    def reduce_dimensions(self, 
                        features: pd.DataFrame, 
                        n_components: int = 0.95) -> pd.DataFrame:
        """
        Reduce feature dimensionality using PCA.
        
        Args:
            features: DataFrame with feature columns
            n_components: Number of components or variance ratio to keep
            
        Returns:
            DataFrame with reduced features
        """
        # Initialize PCA
        pca = PCA(n_components=n_components)
        
        # Fit and transform
        reduced_values = pca.fit_transform(features)
        
        # Create column names for components
        if isinstance(n_components, float):
            n_cols = reduced_values.shape[1]
        else:
            n_cols = n_components
            
        columns = [f'PC{i+1}' for i in range(n_cols)]
        
        # Convert to DataFrame
        reduced_df = pd.DataFrame(reduced_values, index=features.index, columns=columns)
        
        # Log explained variance
        explained_var = pca.explained_variance_ratio_.sum() * 100
        self.logger.info(f"PCA reduced dimensions from {features.shape[1]} to {reduced_df.shape[1]} "
                       f"({explained_var:.2f}% variance explained)")
        
        return reduced_df
    
    def _add_price_features(self, df: pd.DataFrame) -> None:
        """
        Add basic price features to DataFrame.
        
        Args:
            df: DataFrame to modify
        """
        # Price differences
        df['close_diff'] = df['close'].diff()
        df['open_diff'] = df['open'].diff()
        df['high_diff'] = df['high'].diff()
        df['low_diff'] = df['low'].diff()
        
        # Percentage returns
        df['close_ret'] = df['close'].pct_change() * 100
        df['open_ret'] = df['open'].pct_change() * 100
        df['high_ret'] = df['high'].pct_change() * 100
        df['low_ret'] = df['low'].pct_change() * 100
        
        # Log returns
        df['close_log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price ratios
        df['close_to_open'] = df['close'] / df['open']
        df['high_to_low'] = df['high'] / df['low']
        
        # High-Low range
        df['hl_range'] = df['high'] - df['low']
        df['hl_range_pct'] = df['hl_range'] / df['close'] * 100
        
        # Body size (abs of open-close)
        df['body_size'] = np.abs(df['close'] - df['open'])
        df['body_size_pct'] = df['body_size'] / df['close'] * 100
        
        # Upper and lower shadows
        df['upper_shadow'] = df['high'] - np.maximum(df['close'], df['open'])
        df['lower_shadow'] = np.minimum(df['close'], df['open']) - df['low']
        
        # Candlestick features
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_doji'] = (np.abs(df['close'] - df['open']) < (df['high'] - df['low']) * 0.1).astype(int)
    
    def _add_moving_averages(self, df: pd.DataFrame) -> None:
        """
        Add moving averages and derived features to DataFrame.
        
        Args:
            df: DataFrame to modify
        """
        # Simple Moving Averages
        for period in self.ta_periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            # Percentage difference from price to SMA
            df[f'close_to_sma_{period}'] = (df['close'] / df[f'sma_{period}'] - 1) * 100
            
            # SMA slopes
            df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff(3) / 3
        
        # Exponential Moving Averages
        for period in self.ta_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
            # Percentage difference from price to EMA
            df[f'close_to_ema_{period}'] = (df['close'] / df[f'ema_{period}'] - 1) * 100
            
            # EMA slopes
            df[f'ema_{period}_slope'] = df[f'ema_{period}'].diff(3) / 3
        
        # Moving Average Crossovers
        for i, period1 in enumerate(self.ta_periods[:-1]):
            for period2 in self.ta_periods[i+1:]:
                # SMA crossover
                df[f'sma_{period1}_{period2}_ratio'] = df[f'sma_{period1}'] / df[f'sma_{period2}']
                
                # EMA crossover
                df[f'ema_{period1}_{period2}_ratio'] = df[f'ema_{period1}'] / df[f'ema_{period2}']
                
                # Crossover indicators
                df[f'sma_{period1}_{period2}_cross'] = np.where(
                    df[f'sma_{period1}'] > df[f'sma_{period2}'], 1, -1
                )
                
                df[f'ema_{period1}_{period2}_cross'] = np.where(
                    df[f'ema_{period1}'] > df[f'ema_{period2}'], 1, -1
                )
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> None:
        """
        Add momentum indicators to DataFrame.
        
        Args:
            df: DataFrame to modify
        """
        # RSI (Relative Strength Index)
        for period in self.ta_periods:
            delta = df['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        for period in self.ta_periods:
            lowest_low = df['low'].rolling(window=period).min()
            highest_high = df['high'].rolling(window=period).max()
            
            df[f'stoch_k_{period}'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
        
        # MACD (Moving Average Convergence Divergence)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Rate of Change
        for period in self.ta_periods:
            df[f'roc_{period}'] = (df['close'] / df['close'].shift(period) - 1) * 100
        
        # Commodity Channel Index
        for period in self.ta_periods:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - np.mean(x)))
            )
            df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> None:
        """
        Add volatility indicators to DataFrame.
        
        Args:
            df: DataFrame to modify
        """
        # Bollinger Bands
        for period in self.ta_periods:
            df[f'bb_middle_{period}'] = df['close'].rolling(window=period).mean()
            std_dev = df['close'].rolling(window=period).std()
            
            df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + 2 * std_dev
            df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - 2 * std_dev
            
            # Bollinger Band width and percentage
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / df[f'bb_middle_{period}']
            df[f'bb_pct_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # Average True Range (ATR)
        for period in self.ta_periods:
            tr1 = df['high'] - df['low']
            tr2 = np.abs(df['high'] - df['close'].shift(1))
            tr3 = np.abs(df['low'] - df['close'].shift(1))
            
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            df[f'atr_{period}'] = tr.rolling(window=period).mean()
            
            # ATR percentage
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close'] * 100
        
        # Keltner Channels
        for period in self.ta_periods:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df[f'kc_middle_{period}'] = typical_price.rolling(window=period).mean()
            atr = df[f'atr_{period}']
            
            df[f'kc_upper_{period}'] = df[f'kc_middle_{period}'] + 2 * atr
            df[f'kc_lower_{period}'] = df[f'kc_middle_{period}'] - 2 * atr
            
            # Squeeze (Bollinger inside Keltner)
            df[f'squeeze_{period}'] = (
                (df[f'bb_lower_{period}'] > df[f'kc_lower_{period}']) & 
                (df[f'bb_upper_{period}'] < df[f'kc_upper_{period}'])
            ).astype(int)
    
    def _add_volume_features(self, df: pd.DataFrame) -> None:
        """
        Add volume-based features to DataFrame.
        
        Args:
            df: DataFrame to modify
        """
        # Volume moving averages
        for period in self.ta_periods:
            df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
            
            # Volume ratio to moving average
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # Volume rate of change
        for period in self.ta_periods:
            df[f'volume_roc_{period}'] = (df['volume'] / df['volume'].shift(period) - 1) * 100
        
        # On-Balance Volume (OBV)
        df['obv'] = np.where(
            df['close'] > df['close'].shift(1),
            df['volume'],
            np.where(
                df['close'] < df['close'].shift(1),
                -df['volume'],
                0
            )
        ).cumsum()
        
        # Volume-weighted price
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Accumulation/Distribution Line
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)
        df['adl'] = (clv * df['volume']).cumsum()
        
        # Money Flow Index
        for period in self.ta_periods:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            raw_money_flow = typical_price * df['volume']
            
            pos_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
            neg_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
            
            pos_mf = pd.Series(pos_flow).rolling(window=period).sum()
            neg_mf = pd.Series(neg_flow).rolling(window=period).sum()
            
            mf_ratio = pos_mf / neg_mf
            df[f'mfi_{period}'] = 100 - (100 / (1 + mf_ratio))
    
    def _add_cycle_features(self, df: pd.DataFrame) -> None:
        """
        Add cycle-based features to DataFrame.
        
        Args:
            df: DataFrame to modify
        """
        # Check for FLD columns
        fld_columns = [col for col in df.columns if col.startswith('fld_')]
        
        if not fld_columns:
            self.logger.warning("No FLD columns found for cycle features")
            return
        
        # Get cycle lengths from FLD column names
        cycle_lengths = [int(col.split('_')[1]) for col in fld_columns]
        
        # Add FLD cross features
        for col in fld_columns:
            cycle_length = int(col.split('_')[1])
            
            # Price to FLD ratio
            df[f'{col}_ratio'] = df['close'] / df[col] - 1
            
            # FLD crossover indicators
            df[f'{col}_cross_above'] = ((df['close'].shift(1) < df[col].shift(1)) & 
                                     (df['close'] > df[col])).astype(int)
            
            df[f'{col}_cross_below'] = ((df['close'].shift(1) > df[col].shift(1)) & 
                                     (df['close'] < df[col])).astype(int)
            
            # Time since last crossover
            df[f'{col}_days_since_cross'] = 0
            cross_indices = df.index[
                (df[f'{col}_cross_above'] == 1) | (df[f'{col}_cross_below'] == 1)
            ]
            
            last_cross = None
            for i, idx in enumerate(df.index):
                if idx in cross_indices:
                    last_cross = idx
                    df.at[idx, f'{col}_days_since_cross'] = 0
                elif last_cross is not None:
                    df.at[idx, f'{col}_days_since_cross'] = (idx - last_cross).days
        
        # Check for cycle wave columns
        wave_columns = [col for col in df.columns if col.startswith('cycle_wave_')]
        
        if not wave_columns:
            self.logger.warning("No cycle wave columns found for cycle features")
            return
        
        # Add cycle wave features
        for col in wave_columns:
            cycle_length = int(col.split('_')[2])
            
            # Position in cycle (correlation between price and cycle wave)
            df[f'{col}_correlation'] = 0
            
            # Calculate rolling correlation over a window
            window = min(cycle_length * 2, 30)  # Use cycle length as window, but cap it
            for i in range(window, len(df)):
                if i >= window:
                    try:
                        price_window = df['close'].iloc[i-window:i]
                        wave_window = df[col].iloc[i-window:i]
                        
                        # Calculate correlation if there are valid values
                        if not wave_window.isna().all():
                            corr, _ = pearsonr(
                                price_window.fillna(method='ffill'), 
                                wave_window.fillna(method='ffill')
                            )
                            df.at[df.index[i], f'{col}_correlation'] = corr
                    except:
                        pass
    
    def _add_regime_features(self, df: pd.DataFrame) -> None:
        """
        Add market regime features to DataFrame.
        
        Args:
            df: DataFrame to modify
        """
        # ADX (Average Directional Index) for trend strength
        for period in [14, 21]:
            # Calculate +DI and -DI
            plus_dm = np.where(
                (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                np.maximum(df['high'] - df['high'].shift(1), 0),
                0
            )
            minus_dm = np.where(
                (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                np.maximum(df['low'].shift(1) - df['low'], 0),
                0
            )
            
            # True Range
            tr1 = np.abs(df['high'] - df['low'])
            tr2 = np.abs(df['high'] - df['close'].shift(1))
            tr3 = np.abs(df['low'] - df['close'].shift(1))
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            
            # Smoothed TR and DM
            atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
            plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
            
            # ADX calculation
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            df[f'adx_{period}'] = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean()
            df[f'plus_di_{period}'] = plus_di
            df[f'minus_di_{period}'] = minus_di
        
        # Volatility Regime
        for period in [21, 63]:  # 1 month, 3 months
            # Historical volatility (annualized)
            df[f'hist_vol_{period}'] = df['close_log_ret'].rolling(period).std() * np.sqrt(252) * 100
            
            # Calculate if current volatility is high relative to history
            vol_mean = df[f'hist_vol_{period}'].rolling(252).mean()
            vol_std = df[f'hist_vol_{period}'].rolling(252).std()
            
            df[f'vol_zscore_{period}'] = (df[f'hist_vol_{period}'] - vol_mean) / vol_std
            df[f'high_vol_regime_{period}'] = (df[f'vol_zscore_{period}'] > 1).astype(int)
            df[f'low_vol_regime_{period}'] = (df[f'vol_zscore_{period}'] < -1).astype(int)
        
        # Trend Regime
        for period in [21, 63]:
            # Calculate if in a trend (ADX > 25)
            df[f'trending_regime_{period}'] = (df[f'adx_{min(period, 21)}'] > 25).astype(int)
            
            # Trend direction
            df[f'trend_direction_{period}'] = np.where(
                df[f'plus_di_{min(period, 21)}'] > df[f'minus_di_{min(period, 21)}'],
                1,  # Uptrend
                -1  # Downtrend
            )
            
            # Combine trend strength and direction
            df[f'trend_regime_{period}'] = df[f'trending_regime_{period}'] * df[f'trend_direction_{period}']
        
        # Create composite regime indicator
        # 5: Strong Uptrend, 4: Uptrend, 3: Ranging, 2: Downtrend, 1: Strong Downtrend, 0: Volatile
        df['regime_value'] = 3  # Default to ranging
        
        # Strong uptrend
        df.loc[
            (df['adx_14'] > 30) & 
            (df['plus_di_14'] > df['minus_di_14']) & 
            (df['high_vol_regime_21'] == 0),
            'regime_value'
        ] = 5
        
        # Uptrend
        df.loc[
            (df['adx_14'] > 20) & 
            (df['plus_di_14'] > df['minus_di_14']),
            'regime_value'
        ] = 4
        
        # Downtrend
        df.loc[
            (df['adx_14'] > 20) & 
            (df['minus_di_14'] > df['plus_di_14']),
            'regime_value'
        ] = 2
        
        # Strong downtrend
        df.loc[
            (df['adx_14'] > 30) & 
            (df['minus_di_14'] > df['plus_di_14']) & 
            (df['high_vol_regime_21'] == 0),
            'regime_value'
        ] = 1
        
        # Volatile
        df.loc[
            (df['high_vol_regime_21'] == 1),
            'regime_value'
        ] = 0


class CycleMLClassifier:
    """
    Machine learning classifier for predicting market direction based on cycle analysis.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the classifier.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        self.feature_engineer = FeatureEngineer(config)
        self.model = None
        self.model_type = self.config.get('model_type', 'random_forest')
        self.model_params = self.config.get('model_params', {})
        
        # Performance metrics
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.confusion_matrix = None
        
        # Initialize model
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize the machine learning model."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', None),
                min_samples_split=self.model_params.get('min_samples_split', 2),
                min_samples_leaf=self.model_params.get('min_samples_leaf', 1),
                random_state=42
            )
        
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                max_depth=self.model_params.get('max_depth', 3),
                random_state=42
            )
        
        else:
            self.logger.error(f"Unsupported model type: {self.model_type}")
            self.model = RandomForestClassifier(random_state=42)
    
    def train(self, 
             features: pd.DataFrame,
             lookahead: int = 5,
             threshold_pct: float = 1.0,
             test_size: float = 0.2) -> Dict:
        """
        Train the classifier on feature data.
        
        Args:
            features: DataFrame with feature columns
            lookahead: Number of bars to look ahead for price change
            threshold_pct: Threshold percentage for significant price change
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary of training metrics
        """
        # Prepare classification data
        X, y = self.feature_engineer.prepare_classification_data(
            features, lookahead, threshold_pct
        )
        
        # Normalize features
        X_scaled = self.feature_engineer.normalize_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, shuffle=False
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.accuracy = accuracy_score(y_test, y_pred)
        
        # For multi-class, we need to specify average method
        self.precision = precision_score(y_test, y_pred, average='weighted')
        self.recall = recall_score(y_test, y_pred, average='weighted')
        self.f1 = f1_score(y_test, y_pred, average='weighted')
        
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        feature_importances = {}
        if hasattr(self.model, 'feature_importances_'):
            for feature, importance in zip(X.columns, self.model.feature_importances_):
                feature_importances[feature] = importance
        
        # Return metrics
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'feature_importances': feature_importances,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    def cross_validate(self, 
                      features: pd.DataFrame,
                      lookahead: int = 5,
                      threshold_pct: float = 1.0,
                      n_splits: int = 5) -> Dict:
        """
        Perform time-series cross-validation.
        
        Args:
            features: DataFrame with feature columns
            lookahead: Number of bars to look ahead for price change
            threshold_pct: Threshold percentage for significant price change
            n_splits: Number of splits for cross-validation
            
        Returns:
            Dictionary of cross-validation metrics
        """
        # Prepare classification data
        X, y = self.feature_engineer.prepare_classification_data(
            features, lookahead, threshold_pct
        )
        
        # Normalize features
        X_scaled = self.feature_engineer.normalize_features(X)
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_accuracy = []
        cv_precision = []
        cv_recall = []
        cv_f1 = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model on this fold
            self.model.fit(X_train, y_train)
            
            # Evaluate on test fold
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            cv_accuracy.append(accuracy_score(y_test, y_pred))
            cv_precision.append(precision_score(y_test, y_pred, average='weighted'))
            cv_recall.append(recall_score(y_test, y_pred, average='weighted'))
            cv_f1.append(f1_score(y_test, y_pred, average='weighted'))
        
        # Return average metrics
        return {
            'accuracy_mean': np.mean(cv_accuracy),
            'accuracy_std': np.std(cv_accuracy),
            'precision_mean': np.mean(cv_precision),
            'precision_std': np.std(cv_precision),
            'recall_mean': np.mean(cv_recall),
            'recall_std': np.std(cv_recall),
            'f1_mean': np.mean(cv_f1),
            'f1_std': np.std(cv_f1),
            'n_splits': n_splits
        }
    
    def optimize_hyperparameters(self, 
                               features: pd.DataFrame,
                               lookahead: int = 5,
                               threshold_pct: float = 1.0) -> Dict:
        """
        Optimize hyperparameters using grid search.
        
        Args:
            features: DataFrame with feature columns
            lookahead: Number of bars to look ahead for price change
            threshold_pct: Threshold percentage for significant price change
            
        Returns:
            Dictionary of optimization results
        """
        # Prepare classification data
        X, y = self.feature_engineer.prepare_classification_data(
            features, lookahead, threshold_pct
        )
        
        # Normalize features
        X_scaled = self.feature_engineer.normalize_features(X)
        
        # Define parameter grid based on model type
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            base_model = RandomForestClassifier(random_state=42)
            
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
            
            base_model = GradientBoostingClassifier(random_state=42)
            
        else:
            self.logger.error(f"Unsupported model type for optimization: {self.model_type}")
            return {}
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        grid_search.fit(X_scaled, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        # Return optimization results
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'all_scores': {str(p): s for p, s in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'])}
        }
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Array of predicted classes
        """
        if self.model is None:
            self.logger.error("Model not trained yet")
            return np.array([])
        
        # Normalize features
        X_scaled = self.feature_engineer.normalize_features(features, fit=False)
        
        # Make predictions
        return self.model.predict(X_scaled)
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions on new data.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Array of class probabilities
        """
        if self.model is None:
            self.logger.error("Model not trained yet")
            return np.array([])
        
        # Check if model supports predict_proba
        if not hasattr(self.model, 'predict_proba'):
            self.logger.error("Model does not support probability predictions")
            return np.array([])
        
        # Normalize features
        X_scaled = self.feature_engineer.normalize_features(features, fit=False)
        
        # Make probability predictions
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict:
        """
        Get feature importance from the model.
        
        Returns:
            Dictionary mapping features to importance scores
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            self.logger.error("Model not trained or does not support feature importance")
            return {}
        
        # Get feature names from the last training
        if not hasattr(self, 'feature_names'):
            self.logger.warning("Feature names not available")
            return {f"feature_{i}": imp for i, imp in enumerate(self.model.feature_importances_)}
        
        # Return feature importance dictionary
        return {feature: importance for feature, importance in zip(
            self.feature_names, self.model.feature_importances_
        )}
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            self.logger.error("No model to save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            self.logger.error(f"Model file not found: {filepath}")
            return
        
        # Load model
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from {filepath}")


class CycleLSTMPredictor:
    """
    LSTM-based predictor for time series forecasting of market movements.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the LSTM predictor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Check if TensorFlow is available
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow not available. Install with: pip install tensorflow")
            raise ImportError("TensorFlow required for LSTM predictor")
        
        self.feature_engineer = FeatureEngineer(config)
        self.model = None
        
        # Model parameters
        self.sequence_length = self.config.get('sequence_length', 20)
        self.lstm_units = self.config.get('lstm_units', [64, 32])
        self.dropout_rate = self.config.get('dropout_rate', 0.2)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.epochs = self.config.get('epochs', 100)
        self.batch_size = self.config.get('batch_size', 32)
        self.early_stopping = self.config.get('early_stopping', True)
        
        # Performance metrics
        self.train_loss = 0
        self.val_loss = 0
        self.history = None
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, num_features)
        """
        # Create model
        model = Sequential()
        
        # Add LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            
            if i == 0:
                model.add(LSTM(units, return_sequences=return_sequences, input_shape=input_shape))
            else:
                model.add(LSTM(units, return_sequences=return_sequences))
            
            # Add dropout after each LSTM layer
            model.add(Dropout(self.dropout_rate))
        
        # Add output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        # Store model
        self.model = model
        self.logger.info(f"Built LSTM model with {len(self.lstm_units)} layers")
    
    def train(self, 
             features: pd.DataFrame,
             target_column: str = 'future_return',
             validation_split: float = 0.2) -> Dict:
        """
        Train the LSTM model on feature data.
        
        Args:
            features: DataFrame with feature columns
            target_column: Column name for the target variable
            validation_split: Proportion of data to use for validation
            
        Returns:
            Dictionary of training metrics
        """
        # Prepare sequence data
        X, y, feature_names = self.feature_engineer.prepare_sequence_data(
            features, target_column, self.sequence_length
        )
        
        # Check if data is valid
        if len(X) == 0 or len(y) == 0:
            self.logger.error("No valid sequences created")
            return {}
        
        # Store feature names
        self.feature_names = feature_names
        
        # Determine split index
        split_idx = int(len(X) * (1 - validation_split))
        
        # Split into train and validation sets
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model if not already built
        if self.model is None:
            self.build_model((X.shape[1], X.shape[2]))
        
        # Set up callbacks
        callbacks = []
        
        if self.early_stopping:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            callbacks.append(early_stop)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store loss values
        self.train_loss = self.history.history['loss'][-1]
        self.val_loss = self.history.history['val_loss'][-1]
        
        # Evaluate on validation set
        val_pred = self.model.predict(X_val)
        val_mse = np.mean((val_pred.flatten() - y_val) ** 2)
        val_rmse = np.sqrt(val_mse)
        val_mae = np.mean(np.abs(val_pred.flatten() - y_val))
        
        # Return metrics
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'epochs_completed': len(self.history.history['loss'])
        }
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Array of predicted values
        """
        if self.model is None:
            self.logger.error("Model not trained yet")
            return np.array([])
        
        # Prepare sequence data (without target)
        sequence_data = []
        
        # Normalize features
        normalized_features = self.feature_engineer.normalize_features(features, fit=False)
        
        # Create sequences
        for i in range(len(normalized_features) - self.sequence_length + 1):
            sequence_data.append(normalized_features.iloc[i:i+self.sequence_length].values)
        
        # Convert to numpy array
        X = np.array(sequence_data)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Return flattened predictions
        return predictions.flatten()
    
    def predict_trend(self, 
                    features: pd.DataFrame, 
                    threshold: float = 0.0) -> np.ndarray:
        """
        Predict trend direction.
        
        Args:
            features: DataFrame with feature columns
            threshold: Threshold for considering a movement significant
            
        Returns:
            Array of predicted directions (1 for up, -1 for down, 0 for neutral)
        """
        # Get raw predictions
        predictions = self.predict(features)
        
        # Convert to trend directions
        trend = np.zeros_like(predictions)
        trend[predictions > threshold] = 1
        trend[predictions < -threshold] = -1
        
        return trend
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            self.logger.error("No model to save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save model
        save_model(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")
        
        # Save feature scaler
        if self.feature_engineer.feature_scaler is not None:
            scaler_path = os.path.join(os.path.dirname(filepath), 'feature_scaler.joblib')
            joblib.dump(self.feature_engineer.feature_scaler, scaler_path)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            self.logger.error(f"Model file not found: {filepath}")
            return
        
        # Load model
        self.model = load_model(filepath)
        self.logger.info(f"Model loaded from {filepath}")
        
        # Load feature scaler
        scaler_path = os.path.join(os.path.dirname(filepath), 'feature_scaler.joblib')
        if os.path.exists(scaler_path):
            self.feature_engineer.feature_scaler = joblib.load(scaler_path)


class MLSignalEnhancer:
    """
    Enhances FLD cycle signals using machine learning predictions.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the signal enhancer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.classifier_weight = self.config.get('classifier_weight', 0.5)
        
        # Initialize models
        self.classifier = CycleMLClassifier(self.config.get('classifier_config', {}))
        
        # Try to initialize LSTM if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            self.lstm_predictor = CycleLSTMPredictor(self.config.get('lstm_config', {}))
        else:
            self.lstm_predictor = None
            self.logger.warning("TensorFlow not available, LSTM predictor disabled")
        
        # Models trained flag
        self.models_trained = False
    
    def train_models(self, 
                    features: pd.DataFrame,
                    lookahead: int = 5,
                    threshold_pct: float = 1.0) -> Dict:
        """
        Train both classifier and LSTM models.
        
        Args:
            features: DataFrame with feature columns
            lookahead: Number of bars to look ahead for price change
            threshold_pct: Threshold percentage for significant price change
            
        Returns:
            Dictionary of training metrics
        """
        results = {}
        
        # Train classifier
        self.logger.info("Training classifier model...")
        classifier_results = self.classifier.train(features, lookahead, threshold_pct)
        results['classifier'] = classifier_results
        
        # Train LSTM if available
        if self.lstm_predictor is not None:
            try:
                self.logger.info("Training LSTM model...")
                
                # Prepare future return column for LSTM
                lstm_features = features.copy()
                lstm_features['future_return'] = lstm_features['close'].pct_change(lookahead).shift(-lookahead) * 100
                
                lstm_results = self.lstm_predictor.train(lstm_features, 'future_return')
                results['lstm'] = lstm_results
            except Exception as e:
                self.logger.error(f"Error training LSTM model: {e}")
                results['lstm'] = {'error': str(e)}
        
        self.models_trained = True
        return results
    
    def enhance_signal(self, 
                      scan_result: Dict, 
                      current_features: pd.DataFrame) -> Dict:
        """
        Enhance a signal using ML predictions.
        
        Args:
            scan_result: Original scan result from cycle analysis
            current_features: Current market features for ML prediction
            
        Returns:
            Enhanced signal dictionary
        """
        if not self.models_trained:
            self.logger.warning("Models not trained yet, returning original signal")
            return scan_result
        
        # Extract original signal
        original_signal = scan_result.get('signal', {})
        signal_type = original_signal.get('signal', 'neutral')
        signal_strength = original_signal.get('strength', 0.0)
        
        # Make classifier prediction
        try:
            class_proba = self.classifier.predict_proba(current_features)
            
            # Get probability for each class
            if len(class_proba) > 0 and len(class_proba[0]) == 3:
                # Classes are typically [-1, 0, 1] for bearish, neutral, bullish
                bearish_prob = class_proba[0][0]
                neutral_prob = class_proba[0][1]
                bullish_prob = class_proba[0][2]
                
                # Determine ML signal
                if bullish_prob > self.confidence_threshold:
                    ml_signal = 'buy'
                    ml_strength = bullish_prob
                elif bearish_prob > self.confidence_threshold:
                    ml_signal = 'sell'
                    ml_strength = bearish_prob
                else:
                    ml_signal = 'neutral'
                    ml_strength = neutral_prob
            else:
                # Binary classification or unexpected format
                ml_signal = 'neutral'
                ml_strength = 0.5
        except Exception as e:
            self.logger.error(f"Error making classifier prediction: {e}")
            ml_signal = 'neutral'
            ml_strength = 0.5
        
        # Make LSTM prediction if available
        lstm_prediction = 0.0
        if self.lstm_predictor is not None:
            try:
                lstm_prediction = self.lstm_predictor.predict(current_features)[-1]
                
                # Normalize to be in the range [-1, 1]
                lstm_prediction = max(min(lstm_prediction / 5.0, 1.0), -1.0)
            except Exception as e:
                self.logger.error(f"Error making LSTM prediction: {e}")
                lstm_prediction = 0.0
        
        # Combine signals
        combined_signal = signal_type
        
        # Adjust strength based on ML predictions
        adjusted_strength = signal_strength
        
        # Classifier adjustment
        if ('buy' in signal_type and ml_signal == 'buy') or ('sell' in signal_type and ml_signal == 'sell'):
            # ML confirms cycle signal - strengthen
            adjusted_strength = signal_strength * (1.0 + ml_strength * self.classifier_weight)
        elif ('buy' in signal_type and ml_signal == 'sell') or ('sell' in signal_type and ml_signal == 'buy'):
            # ML contradicts cycle signal - weaken
            adjusted_strength = signal_strength * (1.0 - ml_strength * self.classifier_weight)
        
        # LSTM adjustment if available
        if self.lstm_predictor is not None:
            if ('buy' in signal_type and lstm_prediction > 0) or ('sell' in signal_type and lstm_prediction < 0):
                # LSTM confirms cycle signal - strengthen
                adjusted_strength = adjusted_strength * (1.0 + abs(lstm_prediction) * 0.3)
            elif ('buy' in signal_type and lstm_prediction < 0) or ('sell' in signal_type and lstm_prediction > 0):
                # LSTM contradicts cycle signal - weaken
                adjusted_strength = adjusted_strength * (1.0 - abs(lstm_prediction) * 0.3)
        
        # Normalize strength to be in the range [-1, 1]
        adjusted_strength = max(min(adjusted_strength, 1.0), -1.0)
        
        # Update confidence based on adjusted strength
        confidence = original_signal.get('confidence', 'medium')
        if abs(adjusted_strength) > 0.7:
            confidence = 'high'
        elif abs(adjusted_strength) > 0.3:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Create enhanced signal
        enhanced_signal = {
            'signal': signal_type,
            'strength': adjusted_strength,
            'confidence': confidence,
            'alignment': original_signal.get('alignment', 0.0),
            'ml_signal': ml_signal,
            'ml_strength': ml_strength,
            'lstm_prediction': float(lstm_prediction) if self.lstm_predictor is not None else None,
            'enhanced': True
        }
        
        # Create enhanced result
        enhanced_result = scan_result.copy()
        enhanced_result['signal'] = enhanced_signal
        enhanced_result['ml_enhanced'] = True
        
        # Adjust position guidance (optional)
        if 'position_guidance' in enhanced_result:
            position_guidance = enhanced_result['position_guidance'].copy()
            position_guidance['position_size'] = position_guidance.get('position_size', 1.0) * (0.5 + abs(adjusted_strength) * 0.5)
            enhanced_result['position_guidance'] = position_guidance
        
        return enhanced_result
    
    def save_models(self, directory: str) -> None:
        """
        Save trained models to disk.
        
        Args:
            directory: Directory to save models
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save classifier
        self.classifier.save_model(os.path.join(directory, 'classifier_model.joblib'))
        
        # Save LSTM if available
        if self.lstm_predictor is not None:
            self.lstm_predictor.save_model(os.path.join(directory, 'lstm_model'))
        
        # Save feature engineer scalers
        self.classifier.feature_engineer.save_scalers(directory)
    
    def load_models(self, directory: str) -> None:
        """
        Load trained models from disk.
        
        Args:
            directory: Directory to load models from
        """
        # Load classifier
        self.classifier.load_model(os.path.join(directory, 'classifier_model.joblib'))
        
        # Load LSTM if available
        if self.lstm_predictor is not None:
            try:
                self.lstm_predictor.load_model(os.path.join(directory, 'lstm_model'))
            except Exception as e:
                self.logger.error(f"Error loading LSTM model: {e}")
        
        # Load feature engineer scalers
        self.classifier.feature_engineer.load_scalers(directory)
        
        self.models_trained = True


class FeatureSelectionAnalyzer:
    """
    Feature selection and importance analysis for trading models.
    Provides tools to identify the most predictive features and reduce dimensionality.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the feature analyzer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default parameters
        self.top_features_count = self.config.get('top_features_count', 20)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.85)
        
        # Initialized models for feature importance
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Feature importance and selection results
        self.feature_importances = {}
        self.selected_features = []
        self.feature_correlations = None
    
    def analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Analyze feature importance using Random Forest.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            Dictionary of feature importance scores
        """
        # Train a Random Forest model
        self.rf_model.fit(X, y)
        
        # Get feature importances
        importances = self.rf_model.feature_importances_
        
        # Create dictionary mapping features to importance
        self.feature_importances = {feature: importance for feature, importance in zip(X.columns, importances)}
        
        # Sort by importance
        self.feature_importances = dict(sorted(
            self.feature_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return self.feature_importances
    
    def analyze_feature_correlations(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze correlations between features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Correlation matrix
        """
        # Calculate correlation matrix
        self.feature_correlations = X.corr().abs()
        
        return self.feature_correlations
    
    def select_features(self, 
                       X: pd.DataFrame, 
                       y: pd.Series, 
                       method: str = 'importance') -> List[str]:
        """
        Select the most important features using the specified method.
        
        Args:
            X: Feature DataFrame
            y: Target series
            method: Selection method ('importance', 'correlation', 'combined')
            
        Returns:
            List of selected feature names
        """
        if method == 'importance':
            # Select features based on importance
            self.analyze_feature_importance(X, y)
            
            # Get top N features
            self.selected_features = list(self.feature_importances.keys())[:self.top_features_count]
            
        elif method == 'correlation':
            # Select features based on correlation
            self.analyze_feature_correlations(X)
            
            # Find highly correlated features
            selected = set()
            excluded = set()
            
            # Start with all features
            remaining_features = list(X.columns)
            
            # Iterate through features in order of variance
            variances = X.var().sort_values(ascending=False)
            
            for feature in variances.index:
                if feature in excluded:
                    continue
                    
                selected.add(feature)
                
                # Find correlated features to exclude
                for other_feature in remaining_features:
                    if other_feature != feature and other_feature not in excluded:
                        if self.feature_correlations.loc[feature, other_feature] > self.correlation_threshold:
                            excluded.add(other_feature)
            
            self.selected_features = list(selected)
            
        elif method == 'combined':
            # Combination of importance and correlation
            self.analyze_feature_importance(X, y)
            self.analyze_feature_correlations(X)
            
            # Start with top features by importance
            candidates = list(self.feature_importances.keys())[:min(self.top_features_count * 2, len(X.columns))]
            
            selected = set()
            excluded = set()
            
            # Process features in order of importance
            for feature in candidates:
                if feature in excluded:
                    continue
                    
                selected.add(feature)
                
                # Find correlated features to exclude
                for other_feature in candidates:
                    if other_feature != feature and other_feature not in excluded:
                        if self.feature_correlations.loc[feature, other_feature] > self.correlation_threshold:
                            excluded.add(other_feature)
            
            self.selected_features = list(selected)[:self.top_features_count]
            
        else:
            self.logger.error(f"Unsupported feature selection method: {method}")
            self.selected_features = list(X.columns)[:self.top_features_count]
        
        self.logger.info(f"Selected {len(self.selected_features)} features using {method} method")
        return self.selected_features
    
    def create_reduced_features(self, 
                              X: pd.DataFrame, 
                              X_new: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create a reduced feature set based on selected features.
        
        Args:
            X: Original feature DataFrame used for selection
            X_new: Optional new data to apply selection to (defaults to X)
            
        Returns:
            DataFrame with reduced feature set
        """
        if not self.selected_features:
            self.logger.warning("No features selected yet, using all features")
            return X_new if X_new is not None else X
        
        # Default to X if X_new not provided
        data = X_new if X_new is not None else X
        
        # Select features that exist in the data
        valid_features = [f for f in self.selected_features if f in data.columns]
        
        if len(valid_features) < len(self.selected_features):
            self.logger.warning(f"Some selected features not found in data "
                              f"({len(self.selected_features) - len(valid_features)} missing)")
        
        return data[valid_features]
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None) -> None:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            save_path: Optional path to save the plot
        """
        if not self.feature_importances:
            self.logger.error("No feature importance data available")
            return
        
        # Get top N features
        top_features = dict(list(self.feature_importances.items())[:top_n])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.barh(list(top_features.keys())[::-1], list(top_features.values())[::-1])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_correlation_heatmap(self, 
                               features: Optional[List[str]] = None, 
                               save_path: Optional[str] = None) -> None:
        """
        Plot correlation heatmap.
        
        Args:
            features: Optional list of features to include (defaults to selected features)
            save_path: Optional path to save the plot
        """
        if self.feature_correlations is None:
            self.logger.error("No correlation data available")
            return
        
        # Default to selected features
        if features is None:
            features = self.selected_features if self.selected_features else list(self.feature_correlations.columns)
        
        # Limit to reasonable number
        if len(features) > 30:
            self.logger.warning(f"Too many features ({len(features)}), limiting to top 30")
            features = features[:30]
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns_heatmap = None
        
        try:
            import seaborn as sns
            sns_heatmap = sns.heatmap(
                self.feature_correlations.loc[features, features],
                annot=False,
                cmap='coolwarm',
                vmin=0,
                vmax=1
            )
        except ImportError:
            # Fallback to matplotlib
            plt.imshow(
                self.feature_correlations.loc[features, features],
                cmap='coolwarm',
                interpolation='nearest',
                vmin=0,
                vmax=1
            )
            plt.colorbar()
        
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class MarketRegimeClassifier:
    """
    Wrapper around MarketRegimeDetector to provide a standardized
    ML-compatible interface.
    """
    
    def __init__(self, config=None):
        """
        Initialize the market regime classifier.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Use the MarketRegimeDetector from core
        self.detector = MarketRegimeDetector(
            trend_period=self.config.get('trend_period', 50),
            adx_threshold=self.config.get('adx_threshold', 25),
            volatility_threshold=self.config.get('volatility_threshold', 1.5)
        )
        
        # Define regime types
        self.regimes = {
            0: 'Volatile',
            1: 'Strong Downtrend',
            2: 'Downtrend',
            3: 'Ranging',
            4: 'Uptrend',
            5: 'Strong Uptrend'
        }
    
    def classify_regime(self, price_data):
        """
        Classify the market regime using the detector.
        
        Args:
            price_data: DataFrame with OHLC data
            
        Returns:
            Market regime information
        """
        # Use the detector to get regime information
        try:
            regime_info = self.detector.detect_regime(price_data)
            return regime_info
        except Exception as e:
            self.logger.error(f"Error detecting regime: {e}")
            return {"regime": None, "regime_value": 3}  # Default to ranging
    
    def get_trading_parameters(self, price_data):
        """
        Get optimized trading parameters based on the detected regime.
        
        Args:
            price_data: DataFrame with OHLC data
            
        Returns:
            Dictionary of trading parameters
        """
        try:
            # Detect the regime
            regime_info = self.classify_regime(price_data)
            
            # Get trading rules for this regime
            if regime_info and regime_info.get("regime"):
                trading_rules = self.detector.get_regime_trading_rules(regime_info['regime'])
                return trading_rules
            else:
                # Default trading rules
                return {
                    'use_fld_signals': True,
                    'min_strength': 0.3,
                    'min_alignment': 0.6,
                    'position_size': 1.0,
                    'trailing_stop': False,
                    'risk_reward_target': 2.0,
                    'description': "Default rules - regime detection failed"
                }
        except Exception as e:
            self.logger.error(f"Error getting trading parameters: {e}")
            # Return default parameters on error
            return {
                'use_fld_signals': True,
                'min_strength': 0.3,
                'min_alignment': 0.6,
                'position_size': 1.0,
                'trailing_stop': False,
                'risk_reward_target': 2.0,
                'description': "Default rules - error occurred"
            }
    
        
        # Calculate quantiles for regime thresholds
        adx_quantiles = data['adx_14'].quantile([0.2, 0.5, 0.8]).to_dict()
        vol_quantiles = data['volatility_21'].quantile([0.2, 0.5, 0.8]).to_dict()
        
        # Set thresholds
        self.regime_thresholds = {
            'adx_weak': adx_quantiles[0.2],
            'adx_medium': adx_quantiles[0.5],
            'adx_strong': adx_quantiles[0.8],
            'volatility_low': vol_quantiles[0.2],
            'volatility_medium': vol_quantiles[0.5],
            'volatility_high': vol_quantiles[0.8]
        }
        
        return self.regime_thresholds
    
    def classify_regime_rules(self, data: pd.DataFrame) -> np.ndarray:
        """
        Classify market regimes using rule-based approach.
        
        Args:
            data: DataFrame with price data and indicators
            
        Returns:
            Array of regime classifications
        """
        # Detect thresholds if not already set
        if not self.regime_thresholds:
            self.detect_regime_thresholds(data)
        
        # Prepare regime series
        regimes = pd.Series(index=data.index, dtype=int)
        regimes.fillna(3, inplace=True)  # Default to ranging
        
        # Volatile regime
        volatile_mask = data['volatility_21'] > self.regime_thresholds['volatility_high']
        regimes[volatile_mask] = 0  # Volatile
        
        # Trending regimes (non-volatile periods)
        trend_mask = ~volatile_mask & (data['adx_14'] > self.regime_thresholds['adx_medium'])
        
        # Up vs down trends
        uptrend_mask = trend_mask & (data['plus_di_14'] > data['minus_di_14'])
        downtrend_mask = trend_mask & (data['plus_di_14'] <= data['minus_di_14'])
        
        # Strong vs normal trends
        strong_trend_mask = data['adx_14'] > self.regime_thresholds['adx_strong']
        
        # Set regime values
        regimes[uptrend_mask & ~strong_trend_mask] = 4  # Uptrend
        regimes[uptrend_mask & strong_trend_mask] = 5  # Strong Uptrend
        regimes[downtrend_mask & ~strong_trend_mask] = 2  # Downtrend
        regimes[downtrend_mask & strong_trend_mask] = 1  # Strong Downtrend
        
        # Apply smoothing to avoid rapid regime changes
        if self.smooth_window > 1:
            regimes = regimes.rolling(window=self.smooth_window, center=True).median().fillna(method='ffill').fillna(method='bfill')
        
        return regimes.astype(int).values
    
    def train_unsupervised(self, data: pd.DataFrame) -> Dict:
        """
        Train unsupervised model to detect market regimes.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Dictionary of training results
        """
        if self.unsupervised_model is None:
            self.logger.error("Unsupervised model not available")
            return {}
        
        # Extract features for clustering
        features = []
        
        # Returns and volatility features
        if 'close' in data.columns:
            # Daily returns
            daily_returns = data['close'].pct_change() * 100
            features.append(daily_returns)
            
            # Rolling volatility
            volatility = daily_returns.rolling(window=21).std()
            features.append(volatility)
            
            # Price distance from moving averages
            sma50 = data['close'].rolling(window=50).mean()
            sma200 = data['close'].rolling(window=200).mean()
            
            dist_sma50 = (data['close'] / sma50 - 1) * 100
            dist_sma200 = (data['close'] / sma200 - 1) * 100
            
            features.append(dist_sma50)
            features.append(dist_sma200)
        
        # Trend strength features
        if all(col in data.columns for col in ['adx_14', 'plus_di_14', 'minus_di_14']):
            features.append(data['adx_14'])
            
            # Trend direction
            trend_direction = data['plus_di_14'] - data['minus_di_14']
            features.append(trend_direction)
        
        # Combine features
        feature_df = pd.concat(features, axis=1)
        feature_df.columns = [f'feature_{i}' for i in range(len(features))]
        
        # Remove NaN values
        feature_df = feature_df.dropna()
        
        # Normalize features
        feature_values = feature_df.values
        
        # Standardize
        feature_mean = np.nanmean(feature_values, axis=0)
        feature_std = np.nanstd(feature_values, axis=0)
        normalized_features = (feature_values - feature_mean) / feature_std
        
        # Replace any remaining NaNs
        normalized_features = np.nan_to_num(normalized_features)
        
        # Fit clustering model
        self.unsupervised_model.fit(normalized_features)
        
        # Get cluster assignments
        clusters = self.unsupervised_model.predict(normalized_features)
        
        # Map clusters to regimes
        # This requires some logic to assign meaningful labels to clusters
        cluster_stats = {}
        for cluster_id in range(self.n_regimes):
            mask = clusters == cluster_id
            cluster_data = feature_df.iloc[mask]
            
            # Calculate statistics for this cluster
            stats = {
                'count': len(cluster_data),
                'mean': cluster_data.mean().to_dict(),
                'std': cluster_data.std().to_dict()
            }
            
            cluster_stats[cluster_id] = stats
        
        # Assign regimes to clusters based on trend direction and volatility
        regime_mapping = self._map_clusters_to_regimes(cluster_stats)
        
        # Apply regime mapping to clusters
        regimes = np.zeros_like(clusters)
        for cluster_id, regime_id in regime_mapping.items():
            regimes[clusters == cluster_id] = regime_id
        
        # Store regimes in original data index
        regime_series = pd.Series(index=feature_df.index, data=regimes)
        
        # Extend to full data with forward/backward fill
        full_regime_series = pd.Series(index=data.index, dtype=int)
        full_regime_series.loc[regime_series.index] = regime_series
        full_regime_series = full_regime_series.fillna(method='ffill').fillna(method='bfill').fillna(3)
        
        # Apply smoothing
        if self.smooth_window > 1:
            full_regime_series = full_regime_series.rolling(window=self.smooth_window, center=True).median().fillna(method='ffill').fillna(method='bfill')
        
        return {
            'regimes': full_regime_series.astype(int).values,
            'cluster_stats': cluster_stats,
            'regime_mapping': regime_mapping
        }
    
    def _map_clusters_to_regimes(self, cluster_stats: Dict) -> Dict:
        """
        Map clusters to regimes based on statistical properties.
        
        Args:
            cluster_stats: Dictionary of cluster statistics
            
        Returns:
            Dictionary mapping cluster IDs to regime IDs
        """
        # Extract relevant statistics
        cluster_properties = {}
        for cluster_id, stats in cluster_stats.items():
            # Consider trend direction and volatility
            trend_direction = stats['mean'].get('feature_5', 0)  # Assuming feature_5 is trend direction
            volatility = stats['mean'].get('feature_1', 0)       # Assuming feature_1 is volatility
            
            cluster_properties[cluster_id] = {
                'trend_direction': trend_direction,
                'volatility': volatility
            }
        
        # Sort clusters by trend direction
        sorted_by_trend = sorted(
            cluster_properties.items(),
            key=lambda x: x[1]['trend_direction']
        )
        
        # Sort clusters by volatility
        sorted_by_volatility = sorted(
            cluster_properties.items(),
            key=lambda x: x[1]['volatility'],
            reverse=True
        )
        
        # Map clusters to regimes
        regime_mapping = {}
        
        # Most volatile cluster is Volatile regime
        if sorted_by_volatility:
            most_volatile_cluster = sorted_by_volatility[0][0]
            regime_mapping[most_volatile_cluster] = 0  # Volatile
        
        # Assign remaining clusters based on trend direction
        remaining_clusters = [c for c in cluster_properties.keys() if c not in regime_mapping]
        n_remaining = len(remaining_clusters)
        
        if n_remaining >= 4:
            # We have enough clusters for all trend regimes
            sorted_remaining = sorted(
                remaining_clusters,
                key=lambda c: cluster_properties[c]['trend_direction']
            )
            
            # Strong Downtrend
            regime_mapping[sorted_remaining[0]] = 1
            
            # Downtrend
            regime_mapping[sorted_remaining[1]] = 2
            
            # Ranging - middle cluster(s)
            middle_idx = n_remaining // 2
            regime_mapping[sorted_remaining[middle_idx]] = 3
            
            if n_remaining > 4:
                # If more than 4 remaining, assign next one to Ranging as well
                regime_mapping[sorted_remaining[middle_idx + 1]] = 3
            
            # Uptrend
            regime_mapping[sorted_remaining[-2]] = 4
            
            # Strong Uptrend
            regime_mapping[sorted_remaining[-1]] = 5
            
        else:
            # Not enough clusters, do a simpler mapping
            if n_remaining >= 1:
                # Most bearish
                regime_mapping[remaining_clusters[0]] = 2  # Downtrend
            
            if n_remaining >= 2:
                # Middle
                regime_mapping[remaining_clusters[1]] = 3  # Ranging
            
            if n_remaining >= 3:
                # Most bullish
                regime_mapping[remaining_clusters[2]] = 4  # Uptrend
        
        # Make sure all clusters are mapped
        for cluster_id in cluster_properties.keys():
            if cluster_id not in regime_mapping:
                regime_mapping[cluster_id] = 3  # Default to Ranging
        
        return regime_mapping
    
    def detect_regime(self, data: pd.DataFrame, method: str = 'rules') -> Dict:
        """
        Detect current market regime.
        
        Args:
            data: DataFrame with price data and indicators
            method: Detection method ('rules', 'unsupervised')
            
        Returns:
            Dictionary with regime information
        """
        if method == 'rules':
            # Use rule-based classification
            regimes = self.classify_regime_rules(data)
        elif method == 'unsupervised':
            # Use unsupervised clustering
            result = self.train_unsupervised(data)
            regimes = result['regimes']
        else:
            self.logger.error(f"Unsupported regime detection method: {method}")
            regimes = np.full(len(data), 3)  # Default to Ranging
        
        # Get current regime (last value)
        current_regime_id = int(regimes[-1])
        self.current_regime = current_regime_id
        
        # Calculate regime duration
        self.regime_history = list(regimes[-self.lookback_window:])
        regime_duration = 0
        for i in range(len(self.regime_history)-1, -1, -1):
            if self.regime_history[i] == current_regime_id:
                regime_duration += 1
            else:
                break
        
        # Calculate regime probabilities
        unique_regimes, regime_counts = np.unique(regimes[-self.lookback_window:], return_counts=True)
        self.regime_probabilities = {
            self.regimes[int(regime_id)]: count / self.lookback_window
            for regime_id, count in zip(unique_regimes, regime_counts)
        }
        
        # Fill in missing regimes
        for regime_name in self.regimes.values():
            if regime_name not in self.regime_probabilities:
                self.regime_probabilities[regime_name] = 0.0
        
        return {
            'current_regime_id': current_regime_id,
            'current_regime': self.regimes[current_regime_id],
            'regime_duration': regime_duration,
            'regime_probabilities': self.regime_probabilities,
            'regime_history': self.regime_history
        }
    
    def get_trading_parameters(self, regime_id: Optional[int] = None) -> Dict:
        """
        Get optimal trading parameters for the current regime.
        
        Args:
            regime_id: Optional regime ID (uses current regime if None)
            
        Returns:
            Dictionary of trading parameters
        """
        if regime_id is None:
            regime_id = self.current_regime
        
        if regime_id is None:
            self.logger.warning("No current regime detected, using default parameters")
            regime_id = 3  # Default to Ranging
        
        # Define parameters for each regime
        regime_params = {
            0: {  # Volatile
                'description': 'Volatile market - reduce exposure, high standards for entry',
                'min_strength': 0.6,          # Very high threshold
                'min_alignment': 0.9,         # Very high alignment
                'position_size': 0.5,         # Half-sized positions
                'trailing_stop': True,        # Use trailing stops
                'risk_reward_target': 1.2,    # Take profits quickly
                'cycle_weight': 0.4,         # Lower weight on cycle signals
                'ml_weight': 0.6,            # Higher weight on ML signals
                'volatility_filter': True     # Apply volatility filter
            },
            1: {  # Strong Downtrend
                'description': 'Strong downtrend - primarily bearish signals, defensive',
                'min_strength': 0.5,          # High threshold
                'min_alignment': 0.8,         # High alignment requirement
                'position_size': 0.6,         # Smaller positions
                'trailing_stop': False,       # Fixed stops
                'risk_reward_target': 1.5,    # Modest profit targets
                'cycle_weight': 0.6,         # Balanced weight
                'ml_weight': 0.4,            # Balanced weight
                'volatility_filter': False    # No volatility filter
            },
            2: {  # Downtrend
                'description': 'Downtrend - favor bearish signals, be defensive',
                'min_strength': 0.4,          # Higher threshold
                'min_alignment': 0.7,         # Moderate alignment
                'position_size': 0.8,         # Reduced position size
                'trailing_stop': False,       # Fixed stops
                'risk_reward_target': 1.8,    # Standard profit targets
                'cycle_weight': 0.7,         # Higher weight on cycle signals
                'ml_weight': 0.3,            # Lower weight on ML signals
                'volatility_filter': False    # No volatility filter
            },
            3: {  # Ranging
                'description': 'Ranging market - be selective, take smaller positions',
                'min_strength': 0.4,          # Moderate threshold
                'min_alignment': 0.7,         # Moderate alignment
                'position_size': 0.7,         # Reduced position size
                'trailing_stop': False,       # Fixed stops
                'risk_reward_target': 1.5,    # Take profits earlier
                'cycle_weight': 0.5,         # Balanced weight
                'ml_weight': 0.5,            # Balanced weight
                'volatility_filter': True     # Apply volatility filter
            },
            4: {  # Uptrend
                'description': 'Uptrend - favor bullish signals',
                'min_strength': 0.3,          # Standard threshold
                'min_alignment': 0.6,         # Standard alignment
                'position_size': 1.0,         # Full position size
                'trailing_stop': True,        # Use trailing stops
                'risk_reward_target': 2.0,    # Standard profit targets
                'cycle_weight': 0.7,         # Higher weight on cycle signals
                'ml_weight': 0.3,            # Lower weight on ML signals
                'volatility_filter': False    # No volatility filter
            },
            5: {  # Strong Uptrend
                'description': 'Strong uptrend - focus on bullish signals, larger positions',
                'min_strength': 0.2,          # Low threshold to enter more trades
                'min_alignment': 0.5,         # Lower alignment requirement
                'position_size': 1.2,         # Increased position size
                'trailing_stop': True,        # Use trailing stops
                'risk_reward_target': 3.0,    # Aim for larger gains
                'cycle_weight': 0.8,         # Higher weight on cycle signals
                'ml_weight': 0.2,            # Lower weight on ML signals
                'volatility_filter': False    # No volatility filter
            }
        }
        
        return regime_params.get(regime_id, regime_params[3])  # Default to Ranging
    
    def plot_regime_history(self, 
                          data: pd.DataFrame, 
                          method: str = 'rules',
                          save_path: Optional[str] = None) -> None:
        """
        Plot market regimes over time.
        
        Args:
            data: DataFrame with price data
            method: Detection method ('rules', 'unsupervised')
            save_path: Optional path to save the plot
        """
        # Detect regimes
        if method == 'rules':
            regimes = self.classify_regime_rules(data)
        elif method == 'unsupervised':
            result = self.train_unsupervised(data)
            regimes = result['regimes']
        else:
            self.logger.error(f"Unsupported regime detection method: {method}")
            return
        
        # Create regime series
        regime_series = pd.Series(index=data.index, data=regimes)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Plot price
        ax1.plot(data.index, data['close'], color='blue', alpha=0.6)
        ax1.set_title('Price with Market Regimes')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        
        # Plot regimes
        colors = {
            0: 'purple',    # Volatile
            1: 'red',       # Strong Downtrend
            2: 'orange',    # Downtrend
            3: 'gray',      # Ranging
            4: 'green',     # Uptrend
            5: 'darkgreen'  # Strong Uptrend
        }
        
        # Color the background based on regime
        for regime_id, color in colors.items():
            regime_mask = (regime_series == regime_id)
            
            if not any(regime_mask):
                continue
                
            # Find contiguous segments
            regime_change = regime_mask.astype(int).diff().fillna(0)
            segment_starts = data.index[regime_mask & (regime_change != 0)]
            
            if regime_mask.iloc[0]:
                segment_starts = [data.index[0]] + list(segment_starts)
            
            for start_idx in segment_starts:
                end_mask = (data.index > start_idx) & ~regime_mask
                end_idx = data.index[end_mask].min() if any(end_mask) else data.index[-1]
                
                # Shade the region
                ax1.axvspan(start_idx, end_idx, color=color, alpha=0.2)
        
        # Plot regime line
        ax2.plot(data.index, regime_series, color='black', drawstyle='steps-post')
        ax2.set_yticks(range(6))
        ax2.set_yticklabels([self.regimes[i] for i in range(6)])
        ax2.set_ylabel('Regime')
        ax2.set_ylim(-0.5, 5.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class EnsembleSignalGenerator:
    """
    Generate trading signals using an ensemble of methods.
    Combines traditional cycle analysis with ML predictions and regime-aware adjustments.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the ensemble signal generator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default parameters
        self.cycle_weight = self.config.get('cycle_weight', 0.6)
        self.ml_weight = self.config.get('ml_weight', 0.4)
        self.regime_adaptive = self.config.get('regime_adaptive', True)
        
        # Initialize components
        ml_config = self.config.get('ml_config', {})
        self.ml_enhancer = MLSignalEnhancer(ml_config)
        
        regime_config = self.config.get('regime_config', {})
        self.regime_classifier = MarketRegimeClassifier(regime_config)
        
        # Weights for different components
        self.component_weights = {
            'cycle': self.cycle_weight,
            'ml': self.ml_weight,
            'regime': 0.0  # Will be adjusted dynamically
        }
        
    def train_models(self, 
                    features: pd.DataFrame,
                    lookahead: int = 5,
                    threshold_pct: float = 1.0) -> Dict:
        """
        Train ML models used in the ensemble.
        
        Args:
            features: DataFrame with feature columns
            lookahead: Number of bars to look ahead for price change
            threshold_pct: Threshold percentage for significant price change
            
        Returns:
            Dictionary of training metrics
        """
        return self.ml_enhancer.train_models(features, lookahead, threshold_pct)
    
    def generate_signal(self, 
                       scan_result: Dict, 
                       current_features: pd.DataFrame) -> Dict:
        """
        Generate ensemble trading signal.
        
        Args:
            scan_result: Original scan result from cycle analysis
            current_features: Current market features for ML prediction
            
        Returns:
            Enhanced signal dictionary with ensemble output
        """
        # 1. Get cycle signal
        cycle_signal = scan_result.get('signal', {}).copy()
        
        # 2. Get ML-enhanced signal
        ml_enhanced_result = self.ml_enhancer.enhance_signal(scan_result, current_features)
        ml_signal = ml_enhanced_result.get('signal', {})
        
        # 3. Detect market regime
        regime_result = self.regime_classifier.detect_regime(current_features.iloc[-20:], method='rules')
        regime_id = regime_result['current_regime_id']
        
        # 4. Get regime-specific parameters
        regime_params = self.regime_classifier.get_trading_parameters(regime_id)
        
        # 5. Apply regime adjustments to weights if adaptive
        if self.regime_adaptive:
            self.component_weights['cycle'] = regime_params.get('cycle_weight', self.cycle_weight)
            self.component_weights['ml'] = regime_params.get('ml_weight', self.ml_weight)
        
        # 6. Adjust signal thresholds based on regime
        min_strength = regime_params.get('min_strength', 0.3)
        min_alignment = regime_params.get('min_alignment', 0.6)
        
        # 7. Combine signals
        # Extract strengths
        cycle_strength = cycle_signal.get('strength', 0.0)
        ml_strength = ml_signal.get('strength', 0.0)
        
        # Apply weights
        weighted_strength = (
            cycle_strength * self.component_weights['cycle'] +
            ml_strength * self.component_weights['ml']
        )
        
        # Normalize to range [-1, 1]
        ensemble_strength = max(min(weighted_strength, 1.0), -1.0)
        
        # Determine signal type based on strength
        if ensemble_strength > min_strength:
            ensemble_signal_type = "strong_buy" if ensemble_strength > 0.7 else "buy"
        elif ensemble_strength < -min_strength:
            ensemble_signal_type = "strong_sell" if ensemble_strength < -0.7 else "sell"
        else:
            ensemble_signal_type = "neutral"
        
        # Determine confidence based on agreement and strength
        if abs(ensemble_strength) > 0.7:
            confidence = "high"
        elif abs(ensemble_strength) > 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Check alignment
        alignment = cycle_signal.get('alignment', 0.5)
        if alignment < min_alignment and confidence != "low":
            confidence = "medium"  # Downgrade if alignment is low
        
        # 8. Create ensemble signal
        ensemble_signal = {
            'signal': ensemble_signal_type,
            'strength': ensemble_strength,
            'confidence': confidence,
            'alignment': alignment,
            'cycle_contribution': cycle_strength * self.component_weights['cycle'],
            'ml_contribution': ml_strength * self.component_weights['ml'],
            'regime': self.regime_classifier.regimes[regime_id],
            'regime_id': regime_id,
            'regime_duration': regime_result['regime_duration'],
            'ensemble_method': "weighted_average",
            'component_weights': self.component_weights.copy()
        }
        
        # 9. Apply position sizing based on regime
        position_size = regime_params.get('position_size', 1.0)
        
        # 10. Create position guidance
        if ensemble_signal_type in ["buy", "strong_buy", "sell", "strong_sell"]:
            # Base on original guidance but with adjusted size and targets
            original_guidance = scan_result.get('position_guidance', {}).copy()
            
            position_guidance = {
                'entry_price': original_guidance.get('entry_price', scan_result.get('price', 0)),
                'stop_loss': original_guidance.get('stop_loss'),
                'risk': original_guidance.get('risk', 0),
                'position_size': position_size * (0.5 + abs(ensemble_strength) * 0.5),
                'trailing_stop': regime_params.get('trailing_stop', False)
            }
            
            # Adjust target based on regime's risk-reward target
            risk_reward_target = regime_params.get('risk_reward_target', 2.0)
            if 'risk' in position_guidance and position_guidance['risk'] > 0:
                if 'buy' in ensemble_signal_type:
                    position_guidance['target_price'] = position_guidance['entry_price'] + position_guidance['risk'] * risk_reward_target
                else:
                    position_guidance['target_price'] = position_guidance['entry_price'] - position_guidance['risk'] * risk_reward_target
            
            # Recalculate risk-reward ratio
            if 'stop_loss' in position_guidance and position_guidance['stop_loss'] and 'target_price' in position_guidance:
                entry = position_guidance['entry_price']
                stop = position_guidance['stop_loss']
                target = position_guidance['target_price']
                
                risk = abs(entry - stop)
                reward = abs(target - entry)
                
                if risk > 0:
                    position_guidance['risk_reward_ratio'] = reward / risk
                    position_guidance['reward'] = reward
        else:
            # No position for neutral signals
            position_guidance = {
                'position_size': 0.0,
                'message': "No position for neutral signals"
            }
        
        # 11. Create ensemble result
        ensemble_result = scan_result.copy()
        ensemble_result['signal'] = ensemble_signal
        ensemble_result['position_guidance'] = position_guidance
        ensemble_result['regime'] = {
            'id': regime_id,
            'name': self.regime_classifier.regimes[regime_id],
            'duration': regime_result['regime_duration'],
            'probabilities': regime_result['regime_probabilities'],
            'parameters': regime_params
        }
        ensemble_result['ensemble'] = True
        
        return ensemble_result
    
    def save_models(self, directory: str) -> None:
        """
        Save trained models to disk.
        
        Args:
            directory: Directory to save models
        """
        # Create subdirectories
        ml_dir = os.path.join(directory, 'ml')
        os.makedirs(ml_dir, exist_ok=True)
        
        # Save ML models
        self.ml_enhancer.save_models(ml_dir)
        
        # Save configuration
        config_path = os.path.join(directory, 'ensemble_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'component_weights': self.component_weights,
                'regime_adaptive': self.regime_adaptive
            }, f, indent=2)
    
    def load_models(self, directory: str) -> None:
        """
        Load trained models from disk.
        
        Args:
            directory: Directory to load models from
        """
        # Load ML models
        ml_dir = os.path.join(directory, 'ml')
        if os.path.exists(ml_dir):
            self.ml_enhancer.load_models(ml_dir)
        
        # Load configuration
        config_path = os.path.join(directory, 'ensemble_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
                if 'component_weights' in config:
                    self.component_weights = config['component_weights']
                
                if 'regime_adaptive' in config:
                    self.regime_adaptive = config['regime_adaptive']


class WalkForwardOptimizer:
    """
    Perform walk-forward optimization to test and optimize trading strategies.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the walk-forward optimizer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default parameters
        self.window_size = self.config.get('window_size', 252)  # 1 year of trading days
        self.step_size = self.config.get('step_size', 63)       # ~3 months step
        self.min_train_size = self.config.get('min_train_size', 252)  # Minimum training size
        self.test_size = self.config.get('test_size', 63)       # ~3 months test size
        
        # Results storage
        self.results = []
        self.optimal_params = {}
        self.param_stability = {}
        
    def optimize(self, 
                data: pd.DataFrame, 
                optimization_func: callable,
                param_grid: Dict,
                eval_func: callable) -> Dict:
        """
        Perform walk-forward optimization.
        
        Args:
            data: DataFrame with historical data
            optimization_func: Function to optimize parameters on training window
            param_grid: Dictionary of parameter grids to search
            eval_func: Function to evaluate performance on test window
            
        Returns:
            Dictionary of optimization results
        """
        self.results = []
        windows = []
        
        # Calculate number of windows
        data_size = len(data)
        
        if data_size < self.min_train_size + self.test_size:
            self.logger.error(f"Insufficient data for walk-forward optimization. Need at least "
                           f"{self.min_train_size + self.test_size} data points, but got {data_size}.")
            return {}
        
        # Create windows
        start_idx = 0
        while start_idx + self.min_train_size + self.test_size <= data_size:
            train_end = start_idx + self.window_size
            test_end = min(train_end + self.test_size, data_size)
            
            if train_end > data_size:
                train_end = data_size - self.test_size
            
            windows.append({
                'train_start': start_idx,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end
            })
            
            start_idx += self.step_size
        
        # Process each window
        all_window_results = []
        
        for i, window in enumerate(windows):
            self.logger.info(f"Processing window {i+1}/{len(windows)}")
            
            # Extract training and test data
            train_data = data.iloc[window['train_start']:window['train_end']]
            test_data = data.iloc[window['test_start']:window['test_end']]
            
            # Optimize parameters on training data
            opt_result = optimization_func(train_data, param_grid)
            best_params = opt_result.get('best_params', {})
            
            # Evaluate on test data
            eval_result = eval_func(test_data, best_params)
            
            # Store results
            window_result = {
                'window': i + 1,
                'train_start': data.index[window['train_start']],
                'train_end': data.index[window['train_end'] - 1],
                'test_start': data.index[window['test_start']],
                'test_end': data.index[window['test_end'] - 1] if window['test_end'] < len(data) else data.index[-1],
                'train_size': len(train_data),
                'test_size': len(test_data),
                'best_params': best_params,
                'train_performance': opt_result.get('best_score', 0),
                'test_performance': eval_result.get('performance', 0),
                'test_trades': eval_result.get('trades', []),
                'test_metrics': eval_result.get('metrics', {})
            }
            
            all_window_results.append(window_result)
        
        # Analyze parameter stability
        self._analyze_parameter_stability(all_window_results)
        
        # Determine optimal parameters
        self._determine_optimal_parameters(all_window_results)
        
        # Store results
        self.results = all_window_results
        
        # Return summary
        return {
            'windows': len(all_window_results),
            'window_results': all_window_results,
            'optimal_params': self.optimal_params,
            'param_stability': self.param_stability,
            'overall_performance': np.mean([r['test_performance'] for r in all_window_results])
        }
    
    def _analyze_parameter_stability(self, window_results: List[Dict]) -> None:
        """
        Analyze the stability of parameters across windows.
        
        Args:
            window_results: List of window results
        """
        if not window_results:
            return
        
        # Get all parameter names
        param_names = set()
        for result in window_results:
            param_names.update(result.get('best_params', {}).keys())
        
        # Analyze each parameter
        stability_data = {}
        
        for param_name in param_names:
            param_values = []
            
            for result in window_results:
                if param_name in result.get('best_params', {}):
                    param_values.append(result['best_params'][param_name])
            
            # Calculate stability metrics
            if param_values:
                # For numerical parameters
                try:
                    values_numeric = [float(v) for v in param_values]
                    stability_data[param_name] = {
                        'mean': np.mean(values_numeric),
                        'std': np.std(values_numeric),
                        'min': np.min(values_numeric),
                        'max': np.max(values_numeric),
                        'stability': 1 - (np.std(values_numeric) / np.mean(values_numeric)) if np.mean(values_numeric) != 0 else 0,
                        'values': values_numeric
                    }
                except (ValueError, TypeError):
                    # For categorical parameters
                    value_counts = {}
                    for v in param_values:
                        value_counts[v] = value_counts.get(v, 0) + 1
                    
                    most_common = max(value_counts.items(), key=lambda x: x[1])
                    stability_data[param_name] = {
                        'most_common': most_common[0],
                        'most_common_pct': most_common[1] / len(param_values) * 100,
                        'unique_values': len(value_counts),
                        'value_counts': value_counts,
                        'values': param_values
                    }
        
        self.param_stability = stability_data
    
    def _determine_optimal_parameters(self, window_results: List[Dict]) -> None:
        """
        Determine the optimal parameters from walk-forward results.
        
        Args:
            window_results: List of window results
        """
        if not window_results:
            return
        
        # Get all parameter names
        param_names = set()
        for result in window_results:
            param_names.update(result.get('best_params', {}).keys())
        
        # Determine optimal value for each parameter
        optimal_params = {}
        
        for param_name in param_names:
            # Get values from stability analysis
            if param_name in self.param_stability:
                stability_info = self.param_stability[param_name]
                
                # Different approach based on parameter type
                if 'mean' in stability_info:
                    # Numerical parameter - use mean if stable enough, otherwise median
                    if stability_info.get('stability', 0) > 0.7:
                        optimal_params[param_name] = stability_info['mean']
                    else:
                        optimal_params[param_name] = np.median(stability_info['values'])
                else:
                    # Categorical parameter - use most common value
                    optimal_params[param_name] = stability_info['most_common']
        
        self.optimal_params = optimal_params
    
    def plot_performance(self, save_path: Optional[str] = None) -> None:
        """
        Plot performance across walk-forward windows.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.results:
            self.logger.error("No results to plot")
            return
        
        # Extract data for plotting
        windows = [r['window'] for r in self.results]
        train_perf = [r['train_performance'] for r in self.results]
        test_perf = [r['test_performance'] for r in self.results]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(windows, train_perf, 'b-', label='Training Performance')
        plt.plot(windows, test_perf, 'r-', label='Test Performance')
        
        # Add horizontal line for average test performance
        avg_test = np.mean(test_perf)
        plt.axhline(y=avg_test, color='r', linestyle='--', alpha=0.5, 
                  label=f'Avg Test: {avg_test:.4f}')
        
        plt.title('Walk-Forward Optimization Performance')
        plt.xlabel('Window')
        plt.ylabel('Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_parameter_stability(self, 
                               params: Optional[List[str]] = None, 
                               save_path: Optional[str] = None) -> None:
        """
        Plot parameter stability across windows.
        
        Args:
            params: Optional list of parameters to plot (defaults to all)
            save_path: Optional path to save the plot
        """
        if not self.param_stability:
            self.logger.error("No parameter stability data to plot")
            return
        
        # Default to all parameters if none specified
        if params is None:
            # Limit to numerical parameters
            params = [p for p, info in self.param_stability.items() 
                    if 'mean' in info]
        else:
            # Filter to only numerical parameters from those specified
            params = [p for p in params if p in self.param_stability 
                    and 'mean' in self.param_stability[p]]
        
        if not params:
            self.logger.error("No numerical parameters to plot")
            return
        
        # Create plot with subplots for each parameter
        fig, axes = plt.subplots(len(params), 1, figsize=(12, 4 * len(params)), sharex=True)
        
        # Handle single parameter case
        if len(params) == 1:
            axes = [axes]
        
        for i, param in enumerate(params):
            stability_info = self.param_stability[param]
            values = stability_info['values']
            windows = range(1, len(values) + 1)
            
            # Plot parameter values
            axes[i].plot(windows, values, 'o-', color='blue')
            
            # Add horizontal line for mean
            axes[i].axhline(y=stability_info['mean'], color='r', linestyle='--', alpha=0.5,
                         label=f'Mean: {stability_info["mean"]:.4f}')
            
            # Add horizontal lines for standard deviation
            axes[i].axhline(y=stability_info['mean'] + stability_info['std'], 
                         color='r', linestyle=':', alpha=0.3)
            axes[i].axhline(y=stability_info['mean'] - stability_info['std'], 
                         color='r', linestyle=':', alpha=0.3)
            
            # Add optimal parameter value
            if param in self.optimal_params:
                axes[i].axhline(y=self.optimal_params[param], color='g', linestyle='-', alpha=0.5,
                             label=f'Optimal: {self.optimal_params[param]:.4f}')
            
            axes[i].set_title(f'Parameter: {param}')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Set common x-axis label
        axes[-1].set_xlabel('Window')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class AnomalyDetector:
    """
    Detect market anomalies and potential regime changes.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the anomaly detector.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default parameters
        self.lookback_period = self.config.get('lookback_period', 252)  # 1 year of trading days
        self.short_window = self.config.get('short_window', 20)         # ~1 month
        self.volatility_threshold = self.config.get('volatility_threshold', 2.0)  # Z-score threshold
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.volume_threshold = self.config.get('volume_threshold', 3.0)
        
        # Historical data for reference
        self.historical_data = None
        self.historical_stats = {}
        
        # Anomaly tracking
        self.detected_anomalies = []
    
    def fit_historical_data(self, data: pd.DataFrame) -> Dict:
        """
        Fit historical data to establish baseline statistics.
        
        Args:
            data: DataFrame with price and volume data
            
        Returns:
            Dictionary of baseline statistics
        """
        # Store historical data
        self.historical_data = data.copy()
        
        # Calculate baseline statistics
        stats = {}
        
        # Return calculation statistics
        if 'close' in data.columns:
            returns = data['close'].pct_change() * 100
            stats['returns_mean'] = returns.mean()
            stats['returns_std'] = returns.std()
            stats['returns_skew'] = returns.skew()
            stats['returns_kurt'] = returns.kurt()
        
        # Volatility statistics
        if 'close' in data.columns:
            daily_volatility = data['close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
            stats['volatility_mean'] = daily_volatility.mean()
            stats['volatility_std'] = daily_volatility.std()
            stats['volatility_max'] = daily_volatility.max()
        
        # Volume statistics
        if 'volume' in data.columns:
            vol_ratio = data['volume'] / data['volume'].rolling(window=20).mean()
            stats['volume_ratio_mean'] = vol_ratio.mean()
            stats['volume_ratio_std'] = vol_ratio.std()
            stats['volume_ratio_max'] = vol_ratio.max()
        
        # Store statistics
        self.historical_stats = stats
        
        return stats
    
    def detect_anomalies(self, new_data: pd.DataFrame) -> List[Dict]:
        """
        Detect anomalies in new data compared to historical baseline.
        
        Args:
            new_data: New data to check for anomalies
            
        Returns:
            List of detected anomalies
        """
        if self.historical_data is None:
            self.logger.error("Historical data not fitted, call fit_historical_data first")
            return []
        
        # Combine with recent historical data for context
        if len(self.historical_data) > self.lookback_period:
            combined_data = pd.concat([self.historical_data.iloc[-self.lookback_period:], new_data])
        else:
            combined_data = pd.concat([self.historical_data, new_data])
        
        # Remove duplicates if any
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        
        # List to store detected anomalies
        anomalies = []
        
        # Check return anomalies
        if 'close' in combined_data.columns:
            returns = combined_data['close'].pct_change() * 100
            returns_mean = self.historical_stats.get('returns_mean', 0)
            returns_std = self.historical_stats.get('returns_std', 1)
            
            # Calculate z-scores
            returns_zscore = (returns - returns_mean) / returns_std
            
            # Find extreme returns
            extreme_returns = returns.iloc[-len(new_data):]
            extreme_returns_mask = abs(returns_zscore.iloc[-len(new_data):]) > 3.0
            
            if extreme_returns_mask.any():
                for idx in extreme_returns_mask[extreme_returns_mask].index:
                    anomalies.append({
                        'type': 'extreme_return',
                        'date': idx,
                        'value': float(extreme_returns.loc[idx]),
                        'zscore': float(returns_zscore.loc[idx]),
                        'description': f"Extreme price return of {extreme_returns.loc[idx]:.2f}% (z-score: {returns_zscore.loc[idx]:.2f})"
                    })
        
        # Check volatility anomalies
        if 'close' in combined_data.columns:
            # Calculate rolling volatility
            volatility = combined_data['close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
            vol_mean = self.historical_stats.get('volatility_mean', 0)
            vol_std = self.historical_stats.get('volatility_std', 1)
            
            # Calculate z-scores
            vol_zscore = (volatility - vol_mean) / vol_std
            
            # Find volatility spikes
            high_vol = volatility.iloc[-len(new_data):]
            high_vol_mask = vol_zscore.iloc[-len(new_data):] > self.volatility_threshold
            
            if high_vol_mask.any():
                for idx in high_vol_mask[high_vol_mask].index:
                    anomalies.append({
                        'type': 'volatility_spike',
                        'date': idx,
                        'value': float(high_vol.loc[idx]),
                        'zscore': float(vol_zscore.loc[idx]),
                        'description': f"Volatility spike to {high_vol.loc[idx]:.2f}% (z-score: {vol_zscore.loc[idx]:.2f})"
                    })
        
        # Check volume anomalies
        if 'volume' in combined_data.columns:
            # Calculate volume ratio
            vol_ratio = combined_data['volume'] / combined_data['volume'].rolling(window=20).mean()
            vol_ratio_mean = self.historical_stats.get('volume_ratio_mean', 1)
            vol_ratio_std = self.historical_stats.get('volume_ratio_std', 0.2)
            
            # Calculate z-scores
            vol_ratio_zscore = (vol_ratio - vol_ratio_mean) / vol_ratio_std
            
            # Find volume spikes
            high_vol_ratio = vol_ratio.iloc[-len(new_data):]
            high_vol_ratio_mask = vol_ratio_zscore.iloc[-len(new_data):] > self.volume_threshold
            
            if high_vol_ratio_mask.any():
                for idx in high_vol_ratio_mask[high_vol_ratio_mask].index:
                    anomalies.append({
                        'type': 'volume_spike',
                        'date': idx,
                        'value': float(high_vol_ratio.loc[idx]),
                        'zscore': float(vol_ratio_zscore.loc[idx]),
                        'description': f"Volume spike to {high_vol_ratio.loc[idx]:.2f}x average (z-score: {vol_ratio_zscore.loc[idx]:.2f})"
                    })
        
        # Check for correlation breakdowns between cycles
        cycle_cols = [col for col in combined_data.columns if col.startswith('cycle_wave_')]
        if len(cycle_cols) >= 2:
            # Calculate pairwise correlations in recent window
            recent_window = min(self.short_window, len(new_data))
            recent_data = combined_data.iloc[-recent_window:]
            
            for i, col1 in enumerate(cycle_cols[:-1]):
                for col2 in cycle_cols[i+1:]:
                    if col1 in recent_data.columns and col2 in recent_data.columns:
                        try:
                            # Calculate recent correlation
                            corr, p_value = pearsonr(
                                recent_data[col1].fillna(method='ffill'), 
                                recent_data[col2].fillna(method='ffill')
                            )
                            
                            # Calculate historical correlation (reference)
                            hist_window = min(self.lookback_period, len(combined_data) - recent_window)
                            hist_data = combined_data.iloc[-hist_window-recent_window:-recent_window]
                            
                            hist_corr, hist_p = pearsonr(
                                hist_data[col1].fillna(method='ffill'), 
                                hist_data[col2].fillna(method='ffill')
                            )
                            
                            # Check for correlation breakdown
                            if abs(hist_corr) > self.correlation_threshold and abs(corr - hist_corr) > 0.5:
                                anomalies.append({
                                    'type': 'correlation_breakdown',
                                    'date': recent_data.index[-1],
                                    'value': float(corr),
                                    'reference': float(hist_corr),
                                    'description': f"Correlation breakdown between {col1} and {col2}: {hist_corr:.2f} to {corr:.2f}"
                                })
                        except:
                            pass
        
        # Store detected anomalies
        self.detected_anomalies.extend(anomalies)
        
        return anomalies
    
    def get_regime_change_probability(self, 
                                     new_data: pd.DataFrame, 
                                     lookback_days: int = 5) -> float:
        """
        Estimate the probability of a regime change based on recent anomalies.
        
        Args:
            new_data: New market data
            lookback_days: Number of days to consider for recent anomalies
            
        Returns:
            Probability of regime change (0.0 to 1.0)
        """
        # Detect anomalies if not already done
        recent_anomalies = self.detect_anomalies(new_data)
        
        # Filter to anomalies in lookback window
        start_date = new_data.index[-1] - pd.Timedelta(days=lookback_days)
        recent_anomalies = [a for a in self.detected_anomalies 
                          if a['date'] >= start_date]
        
        if not recent_anomalies:
            return 0.0
        
        # Count anomalies by type
        anomaly_counts = {}
        for anomaly in recent_anomalies:
            anomaly_type = anomaly['type']
            anomaly_counts[anomaly_type] = anomaly_counts.get(anomaly_type, 0) + 1
        
        # Calculate base probability - more anomalies increase probability
        base_prob = min(len(recent_anomalies) / 10, 0.7)
        
        # Adjust based on anomaly types
        type_adjustments = {
            'extreme_return': 0.15,
            'volatility_spike': 0.2,
            'volume_spike': 0.1,
            'correlation_breakdown': 0.25
        }
        
        # Apply adjustments
        for anomaly_type, count in anomaly_counts.items():
            adjustment = type_adjustments.get(anomaly_type, 0.0)
            base_prob += adjustment * min(count, 3) / 3  # Cap the effect of multiple anomalies
        
        # Ensure probability is in [0, 1] range
        return min(max(base_prob, 0.0), 1.0)
    
    def get_anomaly_alert(self, 
                        new_data: pd.DataFrame, 
                        min_prob: float = 0.5) -> Optional[Dict]:
        """
        Generate an anomaly alert if probability exceeds threshold.
        
        Args:
            new_data: New market data
            min_prob: Minimum probability for alert
            
        Returns:
            Alert dictionary or None
        """
        # Get regime change probability
        probability = self.get_regime_change_probability(new_data)
        
        if probability < min_prob:
            return None
        
        # Count recent anomalies by type
        start_date = new_data.index[-1] - pd.Timedelta(days=5)
        recent_anomalies = [a for a in self.detected_anomalies 
                          if a['date'] >= start_date]
        
        anomaly_counts = {}
        for anomaly in recent_anomalies:
            anomaly_type = anomaly['type']
            anomaly_counts[anomaly_type] = anomaly_counts.get(anomaly_type, 0) + 1
        
        # Generate alert message
        alert_level = "High" if probability > 0.7 else "Medium" if probability > 0.5 else "Low"
        
        return {
            'date': new_data.index[-1],
            'probability': probability,
            'alert_level': alert_level,
            'anomaly_counts': anomaly_counts,
            'recent_anomalies': recent_anomalies,
            'message': f"{alert_level} probability ({probability:.1%}) of market regime change detected based on {len(recent_anomalies)} recent anomalies"
        }
    
    def plot_anomalies(self, 
                      data: pd.DataFrame, 
                      start_date: Optional[datetime] = None,
                      save_path: Optional[str] = None) -> None:
        """
        Plot detected anomalies on price chart.
        
        Args:
            data: Price data
            start_date: Optional start date for plot
            save_path: Optional path to save the plot
        """
        if not self.detected_anomalies:
            self.logger.warning("No anomalies to plot")
            return
        
        # Filter data by start date if provided
        if start_date is not None:
            plot_data = data[data.index >= start_date].copy()
        else:
            # Default to last 6 months
            default_start = data.index[-1] - pd.Timedelta(days=180)
            plot_data = data[data.index >= default_start].copy()
        
        # Filter anomalies to plot window
        plot_anomalies = [a for a in self.detected_anomalies 
                        if a['date'] in plot_data.index]
        
        if not plot_anomalies:
            self.logger.warning("No anomalies in selected date range")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, 
                              gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot price
        axes[0].plot(plot_data.index, plot_data['close'], color='blue', alpha=0.6)
        axes[0].set_title('Price Chart with Anomalies')
        axes[0].set_ylabel('Price')
        axes[0].grid(True, alpha=0.3)
        
        # Plot anomalies on price chart
        for anomaly in plot_anomalies:
            if anomaly['date'] in plot_data.index:
                # Different markers for different anomaly types
                marker_map = {
                    'extreme_return': 'X',
                    'volatility_spike': '^',
                    'volume_spike': 'o',
                    'correlation_breakdown': 's'
                }
                
                color_map = {
                    'extreme_return': 'red',
                    'volatility_spike': 'purple',
                    'volume_spike': 'green',
                    'correlation_breakdown': 'orange'
                }
                
                marker = marker_map.get(anomaly['type'], 'o')
                color = color_map.get(anomaly['type'], 'black')
                
                # Plot marker
                idx = plot_data.index.get_loc(anomaly['date'])
                axes[0].scatter(
                    anomaly['date'], 
                    plot_data['close'].iloc[idx], 
                    marker=marker, 
                    color=color, 
                    s=100, 
                    alpha=0.7,
                    label=anomaly['type']
                )
        
        # Remove duplicate labels
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys(), loc='best')
        
        # Plot volatility
        if 'close' in plot_data.columns:
            volatility = plot_data['close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
            axes[1].plot(plot_data.index, volatility, color='purple', alpha=0.7)
            axes[1].set_ylabel('Volatility (%)')
            axes[1].grid(True, alpha=0.3)
            
            # Highlight volatility anomalies
            for anomaly in plot_anomalies:
                if anomaly['type'] == 'volatility_spike' and anomaly['date'] in plot_data.index:
                    idx = plot_data.index.get_loc(anomaly['date'])
                    axes[1].scatter(
                        anomaly['date'], 
                        volatility.iloc[idx], 
                        marker='^', 
                        color='red', 
                        s=100, 
                        alpha=0.7
                    )
        
        # Plot volume or volume ratio
        if 'volume' in plot_data.columns:
            vol_ratio = plot_data['volume'] / plot_data['volume'].rolling(window=20).mean()
            axes[2].plot(plot_data.index, vol_ratio, color='green', alpha=0.7)
            axes[2].set_ylabel('Volume Ratio')
            axes[2].grid(True, alpha=0.3)
            
            # Highlight volume anomalies
            for anomaly in plot_anomalies:
                if anomaly['type'] == 'volume_spike' and anomaly['date'] in plot_data.index:
                    idx = plot_data.index.get_loc(anomaly['date'])
                    axes[2].scatter(
                        anomaly['date'], 
                        vol_ratio.iloc[idx], 
                        marker='o', 
                        color='red', 
                        s=100, 
                        alpha=0.7
                    )
        
        # Format x-axis
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

