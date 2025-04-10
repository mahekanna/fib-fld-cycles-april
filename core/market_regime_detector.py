import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum


class MarketRegime(Enum):
    """Enumeration of market regime types."""
    STRONG_UPTREND = 5
    UPTREND = 4
    RANGING = 3
    DOWNTREND = 2
    STRONG_DOWNTREND = 1
    VOLATILE = 0
    UNKNOWN = -1


class MarketRegimeDetector:
    """
    Market regime detection using multiple technical indicators.
    Identifies the current market condition to adjust trading strategy.
    """
    
    def __init__(self, 
                 trend_period: int = 50,
                 adx_threshold: int = 25,
                 volatility_threshold: float = 1.5,
                 correlation_period: int = 20,
                 regime_lookback: int = 5):
        """
        Initialize the MarketRegimeDetector.
        
        Args:
            trend_period: Period for trend indicators
            adx_threshold: ADX threshold for trend strength
            volatility_threshold: Volatility threshold multiplier
            correlation_period: Period for correlation calculation
            regime_lookback: Lookback period for regime smoothing
        """
        self.trend_period = trend_period
        self.adx_threshold = adx_threshold
        self.volatility_threshold = volatility_threshold
        self.correlation_period = correlation_period
        self.regime_lookback = regime_lookback
    
    def detect_regime(self, price_data: pd.DataFrame) -> Dict:
        """
        Detect the current market regime using multiple indicators.
        
        Args:
            price_data: DataFrame containing OHLC price data
            
        Returns:
            Dictionary containing regime information
        """
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in price_data.columns for col in required_columns):
            raise ValueError("Price data must contain open, high, low, and close columns")
        
        # Create a copy to avoid modifying original data
        df = price_data.copy()
        
        # Calculate indicators
        self._calculate_trend_indicators(df)
        self._calculate_volatility_indicators(df)
        self._calculate_oscillator_indicators(df)
        
        # Determine regime based on indicators
        df['regime_value'] = self._combine_indicators(df)
        
        # Map numerical value to regime enum
        df['regime'] = df['regime_value'].apply(self._map_regime_value)
        
        # Smooth regime to avoid frequent changes
        df['smoothed_regime_value'] = df['regime_value'].rolling(window=self.regime_lookback).mean()
        df['smoothed_regime'] = df['smoothed_regime_value'].apply(self._map_regime_value)
        
        # Get current regime
        current_regime = df['smoothed_regime'].iloc[-1]
        
        # Calculate regime duration
        regime_duration = self._calculate_regime_duration(df)
        
        # Package results
        result = {
            'regime': current_regime,
            'regime_value': df['smoothed_regime_value'].iloc[-1],
            'duration': regime_duration,
            'adx': df['adx'].iloc[-1],
            'volatility': df['atr_pct'].iloc[-1],
            'trend_strength': df['trend_strength'].iloc[-1],
            'oscillator_value': df['oscillator_value'].iloc[-1],
            'is_trending': df['adx'].iloc[-1] > self.adx_threshold,
            'is_volatile': df['atr_pct'].iloc[-1] > df['atr_pct_mean'].iloc[-1] * self.volatility_threshold,
            'regime_history': df['smoothed_regime'].dropna().tolist()[-self.regime_lookback:],
            'indicators': {
                'adx': df['adx'].iloc[-1],
                'dmp': df['dmp'].iloc[-1],
                'dmn': df['dmn'].iloc[-1],
                'atr': df['atr'].iloc[-1],
                'atr_pct': df['atr_pct'].iloc[-1],
                'rsi': df['rsi'].iloc[-1],
                'cci': df['cci'].iloc[-1],
                'macd': df['macd'].iloc[-1],
                'macd_signal': df['macd_signal'].iloc[-1],
                'macd_hist': df['macd_hist'].iloc[-1]
            }
        }
        
        return result
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> None:
        """
        Calculate trend indicators and add to DataFrame.
        
        Args:
            df: DataFrame to update
        """
        # Calculate ADX (Average Directional Index)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=self.trend_period)
        
        # Directional Movement Index
        df['dmp'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=self.trend_period)
        df['dmn'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=self.trend_period)
        
        # Determine trend direction
        df['trend_direction'] = np.where(df['dmp'] > df['dmn'], 1, -1)
        
        # Calculate MA trends
        df['ma_short'] = talib.SMA(df['close'], timeperiod=self.trend_period // 2)
        df['ma_long'] = talib.SMA(df['close'], timeperiod=self.trend_period)
        
        # Price relative to MAs
        df['price_to_ma_short'] = df['close'] / df['ma_short'] - 1
        df['price_to_ma_long'] = df['close'] / df['ma_long'] - 1
        
        # Trend strength combines ADX and direction
        df['trend_strength'] = df['adx'] * df['trend_direction'] / 100
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> None:
        """
        Calculate volatility indicators and add to DataFrame.
        
        Args:
            df: DataFrame to update
        """
        # Calculate ATR (Average True Range)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.trend_period)
        
        # Normalize ATR as percentage of price
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # Calculate rolling average of ATR percentage
        df['atr_pct_mean'] = df['atr_pct'].rolling(window=self.trend_period).mean()
        
        # Volatility ratio (current volatility to average)
        df['volatility_ratio'] = df['atr_pct'] / df['atr_pct_mean']
        
        # Historical volatility (standard deviation of log returns)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['hist_vol'] = df['log_return'].rolling(window=self.trend_period).std() * np.sqrt(252) * 100
    
    def _calculate_oscillator_indicators(self, df: pd.DataFrame) -> None:
        """
        Calculate oscillator indicators and add to DataFrame.
        
        Args:
            df: DataFrame to update
        """
        # Calculate RSI (Relative Strength Index)
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # Calculate CCI (Commodity Channel Index)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        
        # Calculate MACD (Moving Average Convergence Divergence)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Normalize oscillator values
        df['rsi_norm'] = (df['rsi'] - 50) / 50  # Range from -1 to 1
        df['cci_norm'] = df['cci'] / 200  # Normalized CCI
        df['macd_norm'] = df['macd_hist'] / df['close'] * 100  # MACD histogram as percentage of price
        
        # Combine oscillators into single value
        df['oscillator_value'] = (df['rsi_norm'] + df['cci_norm'] + df['macd_norm']) / 3
    
    def _combine_indicators(self, df: pd.DataFrame) -> pd.Series:
        """
        Combine multiple indicators into a single regime value.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Series containing regime values
        """
        # Start with trend strength (-1 to 1 scale)
        regime_value = df['trend_strength'].copy()
        
        # Amplify strong trends
        strong_trend_mask = df['adx'] > self.adx_threshold * 1.5
        regime_value[strong_trend_mask] = regime_value[strong_trend_mask] * 1.5
        
        # Reduce regime strength during high volatility
        high_vol_mask = df['volatility_ratio'] > self.volatility_threshold
        regime_value[high_vol_mask] = regime_value[high_vol_mask] * 0.5
        
        # Add oscillator component (with lower weight)
        regime_value = regime_value + df['oscillator_value'] * 0.3
        
        # Bound to -1 to 1 range
        regime_value = np.clip(regime_value, -1, 1)
        
        # Scale to 0-5 range for better mapping to enum
        scaled_regime_value = (regime_value + 1) * 2.5
        
        return scaled_regime_value
    
    def _map_regime_value(self, value: float) -> MarketRegime:
        """
        Map numerical regime value to MarketRegime enum.
        
        Args:
            value: Numerical regime value
            
        Returns:
            MarketRegime enum value
        """
        if pd.isna(value):
            return MarketRegime.UNKNOWN
        
        if value >= 4.5:
            return MarketRegime.STRONG_UPTREND
        elif value >= 3.5:
            return MarketRegime.UPTREND
        elif value > 2.5:
            return MarketRegime.RANGING
        elif value > 1.0:
            return MarketRegime.DOWNTREND
        elif value >= 0:
            return MarketRegime.STRONG_DOWNTREND
        else:
            return MarketRegime.VOLATILE
    
    def _calculate_regime_duration(self, df: pd.DataFrame) -> int:
        """
        Calculate how long the current regime has been in effect.
        
        Args:
            df: DataFrame with regime data
            
        Returns:
            Number of periods in current regime
        """
        if 'smoothed_regime' not in df.columns or df['smoothed_regime'].isna().all():
            return 0
        
        # Get current regime
        current_regime = df['smoothed_regime'].iloc[-1]
        
        # Find last regime change
        regime_series = df['smoothed_regime'].dropna()
        
        if len(regime_series) <= 1:
            return 1
        
        # Count from the end backwards until regime changes
        duration = 1
        for i in range(len(regime_series) - 2, -1, -1):
            if regime_series.iloc[i] == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def get_regime_trading_rules(self, regime: MarketRegime) -> Dict:
        """
        Get trading rules and parameters appropriate for the current regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Dictionary of trading parameters
        """
        # Default/base parameters
        base_rules = {
            'use_fld_signals': True,
            'min_strength': 0.3,
            'min_alignment': 0.6,
            'position_size': 1.0,
            'trailing_stop': False,
            'risk_reward_target': 2.0,
            'additional_filter': None,
            'description': "Default rules"
        }
        
        # Adjust based on regime
        if regime == MarketRegime.STRONG_UPTREND:
            return {
                **base_rules,
                'min_strength': 0.2,  # Lower threshold to enter more trades
                'min_alignment': 0.5,  # Lower alignment requirement
                'position_size': 1.2,  # Increase position size
                'trailing_stop': True,  # Enable trailing stops to maximize gains
                'risk_reward_target': 3.0,  # Aim for larger gains
                'description': "Strong uptrend - focus on bullish signals, larger positions"
            }
        
        elif regime == MarketRegime.UPTREND:
            return {
                **base_rules,
                'min_strength': 0.3,
                'trailing_stop': True,
                'description': "Uptrend - favor bullish signals"
            }
        
        elif regime == MarketRegime.RANGING:
            return {
                **base_rules,
                'min_strength': 0.4,  # Require stronger signals
                'min_alignment': 0.7,  # Require higher alignment
                'position_size': 0.7,  # Reduce position size
                'risk_reward_target': 1.5,  # Take profits earlier
                'description': "Ranging market - be selective, take smaller positions"
            }
        
        elif regime == MarketRegime.DOWNTREND:
            return {
                **base_rules,
                'min_strength': 0.4,  # Higher threshold
                'position_size': 0.8,  # Reduced position size
                'description': "Downtrend - favor bearish signals, be defensive"
            }
        
        elif regime == MarketRegime.STRONG_DOWNTREND:
            return {
                **base_rules,
                'min_strength': 0.5,  # Much higher threshold for bullish trades
                'min_alignment': 0.8,  # Higher alignment requirement
                'position_size': 0.6,  # Smaller positions
                'risk_reward_target': 1.5,  # Take profits quicker
                'description': "Strong downtrend - primarily bearish signals, defensive"
            }
        
        elif regime == MarketRegime.VOLATILE:
            return {
                **base_rules,
                'min_strength': 0.6,  # Very high threshold
                'min_alignment': 0.9,  # Very high alignment
                'position_size': 0.5,  # Half-sized positions
                'trailing_stop': True,  # Use trailing stops
                'risk_reward_target': 1.2,  # Take profits quickly
                'description': "Volatile market - reduce exposure, high standards for entry"
            }
        
        else:  # UNKNOWN or any other value
            return base_rules


class VolumeProfileAnalyzer:
    """
    Volume profile analysis to identify key support/resistance levels.
    """
    
    def __init__(self, 
                 num_bins: int = 20,
                 lookback_period: int = 100,
                 value_area_pct: float = 0.7):
        """
        Initialize the VolumeProfileAnalyzer.
        
        Args:
            num_bins: Number of price bins for volume profile
            lookback_period: Number of bars to analyze
            value_area_pct: Percentage of volume to include in value area
        """
        self.num_bins = num_bins
        self.lookback_period = lookback_period
        self.value_area_pct = value_area_pct
    
    def analyze_volume_profile(self, price_data: pd.DataFrame) -> Dict:
        """
        Create and analyze a volume profile.
        
        Args:
            price_data: DataFrame containing OHLC and volume data
            
        Returns:
            Dictionary with volume profile analysis
        """
        # Check for required columns
        if 'volume' not in price_data.columns:
            raise ValueError("Price data must contain a volume column")
        
        # Use recent data based on lookback
        df = price_data.tail(self.lookback_period).copy()
        
        # Calculate typical price for each bar
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Determine price range for binning
        price_min = df['low'].min()
        price_max = df['high'].max()
        
        # Create price bins
        bin_edges = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Assign each bar to a bin based on typical price
        df['price_bin'] = pd.cut(df['typical_price'], bins=bin_edges, labels=False)
        
        # Calculate volume per bin
        volume_profile = df.groupby('price_bin')['volume'].sum()
        
        # Find the Point of Control (POC) - price level with highest volume
        poc_bin = volume_profile.idxmax()
        poc_price = bin_centers[poc_bin]
        
        # Calculate value area
        total_volume = volume_profile.sum()
        target_volume = total_volume * self.value_area_pct
        
        # Start from POC and expand outwards until we reach target volume
        current_volume = volume_profile[poc_bin]
        va_low_bin = poc_bin
        va_high_bin = poc_bin
        
        while current_volume < target_volume and (va_low_bin > 0 or va_high_bin < len(volume_profile) - 1):
            # Check volumes of adjacent bins
            vol_below = volume_profile[va_low_bin - 1] if va_low_bin > 0 else 0
            vol_above = volume_profile[va_high_bin + 1] if va_high_bin < len(volume_profile) - 1 else 0
            
            # Add the larger volume bin
            if vol_below > vol_above and va_low_bin > 0:
                va_low_bin -= 1
                current_volume += vol_below
            elif va_high_bin < len(volume_profile) - 1:
                va_high_bin += 1
                current_volume += vol_above
            else:
                break
        
        # Get price levels for value area
        va_low_price = bin_edges[va_low_bin]
        va_high_price = bin_edges[va_high_bin + 1]
        
        # Create volume profile data for visualization
        profile_data = []
        for bin_idx, volume in volume_profile.items():
            bin_low = bin_edges[bin_idx]
            bin_high = bin_edges[bin_idx + 1]
            bin_price = bin_centers[bin_idx]
            
            profile_data.append({
                'bin_idx': bin_idx,
                'price': bin_price,
                'price_low': bin_low,
                'price_high': bin_high,
                'volume': volume,
                'volume_pct': volume / total_volume * 100,
                'is_poc': bin_idx == poc_bin,
                'in_value_area': va_low_bin <= bin_idx <= va_high_bin
            })
        
        # Find developing value areas (clusters of volume)
        value_clusters = self._find_volume_clusters(profile_data)
        
        return {
            'profile_data': profile_data,
            'point_of_control': poc_price,
            'value_area_low': va_low_price,
            'value_area_high': va_high_price,
            'total_volume': total_volume,
            'value_area_volume_pct': current_volume / total_volume * 100,
            'current_price': df['close'].iloc[-1],
            'position_in_profile': self._position_in_profile(df['close'].iloc[-1], profile_data),
            'value_clusters': value_clusters
        }
    
    def _find_volume_clusters(self, profile_data: List[Dict]) -> List[Dict]:
        """
        Find clusters of high volume within the profile.
        
        Args:
            profile_data: List of dictionaries with bin data
            
        Returns:
            List of dictionaries describing volume clusters
        """
        # Sort by volume
        sorted_bins = sorted(profile_data, key=lambda x: x['volume'], reverse=True)
        
        # Take top 30% of bins by volume
        top_volume_bins = sorted_bins[:int(len(sorted_bins) * 0.3)]
        
        # Sort by price for clustering
        top_volume_bins = sorted(top_volume_bins, key=lambda x: x['price'])
        
        # Cluster adjacent high volume bins
        clusters = []
        current_cluster = []
        
        for i, bin_data in enumerate(top_volume_bins):
            if not current_cluster:
                # Start new cluster
                current_cluster = [bin_data]
            elif bin_data['bin_idx'] - current_cluster[-1]['bin_idx'] <= 1:
                # Add to current cluster if adjacent
                current_cluster.append(bin_data)
            else:
                # Finish current cluster and start new one
                if len(current_cluster) >= 2:  # Only save clusters with at least 2 bins
                    clusters.append(self._summarize_cluster(current_cluster))
                current_cluster = [bin_data]
        
        # Add last cluster if needed
        if len(current_cluster) >= 2:
            clusters.append(self._summarize_cluster(current_cluster))
        
        return clusters
    
    def _summarize_cluster(self, cluster: List[Dict]) -> Dict:
        """
        Create a summary of a volume cluster.
        
        Args:
            cluster: List of dictionaries with bin data
            
        Returns:
            Dictionary summarizing the cluster
        """
        total_volume = sum(bin_data['volume'] for bin_data in cluster)
        weighted_price = sum(bin_data['price'] * bin_data['volume'] for bin_data in cluster) / total_volume
        
        return {
            'price_low': cluster[0]['price_low'],
            'price_high': cluster[-1]['price_high'],
            'mid_price': weighted_price,
            'volume': total_volume,
            'bin_count': len(cluster),
            'bins': [bin_data['bin_idx'] for bin_data in cluster]
        }
    
    def _position_in_profile(self, current_price: float, profile_data: List[Dict]) -> Dict:
        """
        Determine where the current price sits in the volume profile.
        
        Args:
            current_price: Current price
            profile_data: List of dictionaries with bin data
            
        Returns:
            Dictionary describing position in profile
        """
        # Find the bin that contains the current price
        current_bin = None
        
        for bin_data in profile_data:
            if bin_data['price_low'] <= current_price <= bin_data['price_high']:
                current_bin = bin_data
                break
        
        if not current_bin:
            if current_price < profile_data[0]['price_low']:
                return {'position': 'below_profile', 'bin_data': None}
            else:
                return {'position': 'above_profile', 'bin_data': None}
        
        # Determine position relative to POC and value area
        position_desc = []
        
        if current_bin['is_poc']:
            position_desc.append('at_poc')
        elif current_price < profile_data[0]['price'] and profile_data[0]['is_poc']:
            position_desc.append('below_poc')
        elif current_price > profile_data[-1]['price'] and profile_data[-1]['is_poc']:
            position_desc.append('above_poc')
        else:
            # Find POC bin
            poc_bin = next(bin_data for bin_data in profile_data if bin_data['is_poc'])
            position_desc.append('below_poc' if current_price < poc_bin['price'] else 'above_poc')
        
        if current_bin['in_value_area']:
            position_desc.append('in_value_area')
        else:
            position_desc.append('outside_value_area')
        
        # Volume at current level
        volume_percentile = sum(1 for bin_data in profile_data 
                              if bin_data['volume'] <= current_bin['volume']) / len(profile_data)
        
        return {
            'position': '_'.join(position_desc),
            'bin_data': current_bin,
            'volume_percentile': volume_percentile
        }
