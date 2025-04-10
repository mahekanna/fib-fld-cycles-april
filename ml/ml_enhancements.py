import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
