import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import os
import jinja2
import markdown
import base64
import io
from pathlib import Path
import json

from ..models.scan_result import ScanResult
from ..models.report_parameters import ReportParameters
from ..core.scanner import FibCycleScanner
from ..storage.results_repository import ResultsRepository


class ReportGenerator:
    """
    Generate comprehensive reports from analysis results.
    """
    
    def __init__(self, 
                 config: Dict,
                 scanner: Optional[FibCycleScanner] = None,
                 repository: Optional[ResultsRepository] = None):
        """
        Initialize the ReportGenerator.
        
        Args:
            config: Configuration dictionary
            scanner: Optional FibCycleScanner instance
            repository: Optional ResultsRepository instance
        """
        self.config = config
        self.scanner = scanner or FibCycleScanner(config)
        self.repository = repository or ResultsRepository(config)
        
        # Set up templates
        self.template_dir = config.get('report', {}).get('template_dir', 'templates/reports')
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def generate_report(self, 
                       params: ReportParameters,
                       results: Optional[List[ScanResult]] = None) -> str:
        """
        Generate a report based on the specified parameters.
        
        Args:
            params: Report parameters
            results: Optional list of scan results (fetched if not provided)
            
        Returns:
            Generated report content
        """
        # Fetch results if not provided
        if results is None:
            results = self._fetch_results(params)
        
        # Select appropriate template
        template_name = params.template
        if template_name is None:
            # Choose default template based on output format
            if params.output_format == 'html':
                template_name = 'default_report.html'
            elif params.output_format == 'markdown':
                template_name = 'default_report.md'
            else:
                template_name = 'default_report.html'
        
        # Load template
        template = self.jinja_env.get_template(template_name)
        
        # Prepare report data
        report_data = self._prepare_report_data(params, results)
        
        # Render template
        report_content = template.render(**report_data)
        
        # Post-process for different formats
        if params.output_format == 'markdown' and template_name.endswith('.html'):
            # Extract body content if HTML template used for Markdown output
            import re
            body_match = re.search(r'<body>(.*?)</body>', report_content, re.DOTALL)
            if body_match:
                report_content = body_match.group(1)
                # Strip HTML tags (basic conversion)
                report_content = re.sub(r'<[^>]+>', '', report_content)
        
        return report_content
    
    def save_report(self, 
                   content: str, 
                   output_path: str) -> str:
        """
        Save report content to a file.
        
        Args:
            content: Report content
            output_path: Path to save the report
            
        Returns:
            Path to the saved file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Write content to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return output_path
    
    def generate_daily_report(self, 
                             symbols: List[str], 
                             output_path: Optional[str] = None,
                             include_market_regime: bool = True) -> str:
        """
        Generate a daily report for the specified symbols.
        
        Args:
            symbols: List of symbols to include in the report
            output_path: Optional path to save the report
            include_market_regime: Whether to include market regime analysis
            
        Returns:
            Path to the saved report or report content
        """
        # Define report parameters
        now = datetime.now()
        params = ReportParameters(
            title=f"Daily Market Analysis Report",
            subtitle=f"Generated on {now.strftime('%Y-%m-%d %H:%M')}",
            date=now,
            symbols=symbols,
            intervals=["daily"],
            include_charts=True,
            include_signals=True,
            include_cycles=True,
            include_harmonic_relationships=True,
            include_market_regime=include_market_regime,
            output_format="html",
            template="daily_report.html"
        )
        
        # Generate report
        report_content = self.generate_report(params)
        
        # Save if output path provided
        if output_path:
            return self.save_report(report_content, output_path)
        else:
            # Generate a default output path
            report_dir = self.config.get('report', {}).get('report_dir', 'reports')
            os.makedirs(report_dir, exist_ok=True)
            default_path = os.path.join(
                report_dir, 
                f"daily_report_{now.strftime('%Y%m%d_%H%M')}.html"
            )
            return self.save_report(report_content, default_path)
    
    def generate_signals_report(self, 
                              min_strength: float = 0.3,
                              min_alignment: float = 0.6,
                              output_path: Optional[str] = None) -> str:
        """
        Generate a report focusing on current trading signals.
        
        Args:
            min_strength: Minimum signal strength to include
            min_alignment: Minimum cycle alignment to include
            output_path: Optional path to save the report
            
        Returns:
            Path to the saved report or report content
        """
        # Get recent results from repository
        recent_results = self.repository.get_recent_results(hours=24)
        
        # Filter results by signal strength and alignment
        filtered_results = []
        for result in recent_results:
            if (result.success and 
                abs(result.signal.get('strength', 0)) >= min_strength and
                result.signal.get('alignment', 0) >= min_alignment):
                filtered_results.append(result)
        
        # Sort by signal strength
        filtered_results.sort(
            key=lambda r: abs(r.signal.get('strength', 0)), 
            reverse=True
        )
        
        # Define report parameters
        now = datetime.now()
        params = ReportParameters(
            title=f"Trading Signals Report",
            subtitle=f"High-Conviction Signals as of {now.strftime('%Y-%m-%d %H:%M')}",
            date=now,
            include_charts=True,
            include_signals=True,
            include_cycles=True,
            include_harmonic_relationships=False,
            include_market_regime=False,
            output_format="html",
            template="signals_report.html"
        )
        
        # Generate report
        report_content = self.generate_report(params, filtered_results)
        
        # Save if output path provided
        if output_path:
            return self.save_report(report_content, output_path)
        else:
            # Generate a default output path
            report_dir = self.config.get('report', {}).get('report_dir', 'reports')
            os.makedirs(report_dir, exist_ok=True)
            default_path = os.path.join(
                report_dir, 
                f"signals_report_{now.strftime('%Y%m%d_%H%M')}.html"
            )
            return self.save_report(report_content, default_path)
    
    def generate_backtest_report(self, 
                               backtest_results: Dict,
                               output_path: Optional[str] = None) -> str:
        """
        Generate a report for backtest results.
        
        Args:
            backtest_results: Dictionary containing backtest results
            output_path: Optional path to save the report
            
        Returns:
            Path to the saved report or report content
        """
        # Load template
        template = self.jinja_env.get_template('backtest_report.html')
        
        # Generate equity curve chart
        equity_curve_img = self._generate_equity_curve_chart(backtest_results)
        
        # Prepare trade statistics
        trade_stats = self._analyze_trades(backtest_results.get('trades', []))
        
        # Prepare report data
        report_data = {
            'title': f"Backtest Results: {backtest_results.get('symbol', 'Unknown')}",
            'subtitle': f"Period: {backtest_results.get('start_date', 'Unknown')} to {backtest_results.get('end_date', 'Unknown')}",
            'date': datetime.now(),
            'symbol': backtest_results.get('symbol', 'Unknown'),
            'interval': backtest_results.get('interval', 'daily'),
            'duration': backtest_results.get('duration', 0),
            'initial_capital': backtest_results.get('initial_capital', 0),
            'final_capital': backtest_results.get('final_capital', 0),
            'equity_curve_img': equity_curve_img,
            'trades': backtest_results.get('trades', []),
            'metrics': backtest_results.get('metrics', {}),
            'trade_stats': trade_stats
        }
        
        # Render template
        report_content = template.render(**report_data)
        
        # Save if output path provided
        if output_path:
            return self.save_report(report_content, output_path)
        else:
            # Generate a default output path
            report_dir = self.config.get('report', {}).get('report_dir', 'reports')
            os.makedirs(report_dir, exist_ok=True)
            now = datetime.now()
            default_path = os.path.join(
                report_dir, 
                f"backtest_{backtest_results.get('symbol', 'unknown')}_{now.strftime('%Y%m%d_%H%M')}.html"
            )
            return self.save_report(report_content, default_path)
    
    def _fetch_results(self, params: ReportParameters) -> List[ScanResult]:
        """
        Fetch results based on report parameters.
        
        Args:
            params: Report parameters
            
        Returns:
            List of ScanResult instances
        """
        results = []
        
        # Check if we have explicit symbols and intervals
        if params.symbols and params.intervals:
            # Fetch or generate results for each symbol and interval
            for symbol in params.symbols:
                for interval in params.intervals:
                    # Try to get from repository first
                    result = self.repository.get_latest_result(symbol, interval)
                    
                    # If not found or too old, generate new
                    if (result is None or 
                        (datetime.now() - result.timestamp).total_seconds() > 86400):  # More than a day old
                        scan_params = ScanParameters(
                            symbol=symbol,
                            exchange=self.config.get('general', {}).get('default_exchange', 'NSE'),
                            interval=interval,
                            lookback=1000,
                            num_cycles=3,
                            price_source="close",
                            generate_chart=params.include_charts
                        )
                        result = self.scanner.analyze_symbol(scan_params)
                        
                        # Save to repository
                        if result.success:
                            self.repository.save_result(result)
                    
                    # Add to results list
                    if result and result.success:
                        results.append(result)
        else:
            # Just fetch recent results
            results = self.repository.get_recent_results(hours=24)
            
            # Filter by interval if specified
            if params.intervals:
                results = [r for r in results if r.interval in params.intervals]
        
        return results
    
    def _prepare_report_data(self, 
                            params: ReportParameters, 
                            results: List[ScanResult]) -> Dict:
        """
        Prepare data for report template.
        
        Args:
            params: Report parameters
            results: List of ScanResult instances
            
        Returns:
            Dictionary of data for template rendering
        """
        # Group results by interval
        results_by_interval = {}
        for result in results:
            if result.interval not in results_by_interval:
                results_by_interval[result.interval] = []
            results_by_interval[result.interval].append(result)
        
        # Group by bull/bear signals
        bullish_signals = [r for r in results if 'buy' in r.signal.get('signal', '')]
        bearish_signals = [r for r in results if 'sell' in r.signal.get('signal', '')]
        neutral_signals = [r for r in results if 'neutral' in r.signal.get('signal', '')]
        
        # Sort by signal strength
        bullish_signals.sort(key=lambda r: r.signal.get('strength', 0), reverse=True)
        bearish_signals.sort(key=lambda r: abs(r.signal.get('strength', 0)), reverse=True)
        
        # Calculate market breadth
        total_signals = len(results)
        breadth = {
            'bullish_percentage': len(bullish_signals) / total_signals * 100 if total_signals > 0 else 0,
            'bearish_percentage': len(bearish_signals) / total_signals * 100 if total_signals > 0 else 0,
            'neutral_percentage': len(neutral_signals) / total_signals * 100 if total_signals > 0 else 0,
            'net_bullish': len(bullish_signals) - len(bearish_signals),
            'breadth_indicator': (len(bullish_signals) - len(bearish_signals)) / total_signals if total_signals > 0 else 0
        }
        
        # Extract cycle information
        all_cycles = set()
        cycle_frequencies = {}
        
        for result in results:
            for cycle in result.detected_cycles:
                all_cycles.add(cycle)
                cycle_frequencies[cycle] = cycle_frequencies.get(cycle, 0) + 1
        
        # Prepare chart images
        chart_images = {}
        
        if params.include_charts:
            for result in results:
                if result.chart_image is not None:
                    # Convert image to base64
                    img_bytes = io.BytesIO()
                    plt.figure(figsize=(10, 6))
                    plt.imshow(result.chart_image)
                    plt.axis('off')
                    plt.savefig(img_bytes, format='PNG', bbox_inches='tight')
                    img_bytes.seek(0)
                    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
                    chart_images[result.symbol] = img_base64
        
        # Build report data dictionary
        report_data = {
            'title': params.title,
            'subtitle': params.subtitle,
            'author': params.author,
            'date': params.date,
            'report_type': 'Standard Analysis Report',
            'results': results,
            'results_by_interval': results_by_interval,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'neutral_signals': neutral_signals,
            'total_symbols': len(set(r.symbol for r in results)),
            'total_signals': total_signals,
            'market_breadth': breadth,
            'cycle_data': {
                'all_cycles': sorted(all_cycles),
                'frequencies': cycle_frequencies
            },
            'chart_images': chart_images,
            'include_charts': params.include_charts,
            'include_signals': params.include_signals,
            'include_cycles': params.include_cycles,
            'include_harmonic_relationships': params.include_harmonic_relationships,
            'include_market_regime': params.include_market_regime
        }
        
        return report_data
    
    def _generate_equity_curve_chart(self, backtest_results: Dict) -> str:
        """
        Generate equity curve chart for backtest report.
        
        Args:
            backtest_results: Dictionary containing backtest results
            
        Returns:
            Base64 encoded image
        """
        equity_curve = backtest_results.get('equity_curve', [])
        if not equity_curve:
            return ""
        
        # Extract data
        dates = [datetime.fromisoformat(e['date'].replace('Z', '+00:00')) if isinstance(e['date'], str) else e['date'] 
                for e in equity_curve]
        equity = [e['equity'] for e in equity_curve]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot equity curve
        ax.plot(dates, equity, label='Equity', color='blue', linewidth=2)
        
        # Add initial capital reference line
        ax.axhline(y=backtest_results.get('initial_capital', 0), color='gray', linestyle='--', alpha=0.7,
                  label=f"Initial Capital ({backtest_results.get('initial_capital', 0)})")
        
        # Format plot
        ax.set_title(f"Equity Curve - {backtest_results.get('symbol', 'Unknown')}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add metrics annotation
        metrics = backtest_results.get('metrics', {})
        metrics_text = (
            f"Total Return: {metrics.get('profit_loss_pct', 0):.2f}%\n"
            f"Win Rate: {metrics.get('win_rate', 0)*100:.2f}%\n"
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
            f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%\n"
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}"
        )
        
        # Add metrics box
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        # Convert to base64
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='PNG', bbox_inches='tight')
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64
    
    def _analyze_trades(self, trades: List[Dict]) -> Dict:
        """
        Analyze trades for detailed statistics.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary of trade statistics
        """
        if not trades:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(trades)
        
        # Basic stats
        total_trades = len(df)
        winning_trades = len(df[df['profit_loss'] > 0])
        losing_trades = len(df[df['profit_loss'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit/loss stats
        gross_profit = df[df['profit_loss'] > 0]['profit_loss'].sum()
        gross_loss = abs(df[df['profit_loss'] <= 0]['profit_loss'].sum())
        net_profit = gross_profit - gross_loss
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade stats
        avg_winning_trade = df[df['profit_loss'] > 0]['profit_loss'].mean() if winning_trades > 0 else 0
        avg_losing_trade = abs(df[df['profit_loss'] <= 0]['profit_loss'].mean()) if losing_trades > 0 else 0
        avg_win_loss_ratio = avg_winning_trade / avg_losing_trade if avg_losing_trade > 0 else float('inf')
        
        # Time stats
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        df['duration'] = (df['exit_date'] - df['entry_date']).dt.total_seconds() / (60 * 60 * 24)  # days
        
        avg_trade_duration = df['duration'].mean()
        avg_winning_duration = df[df['profit_loss'] > 0]['duration'].mean() if winning_trades > 0 else 0
        avg_losing_duration = df[df['profit_loss'] <= 0]['duration'].mean() if losing_trades > 0 else 0
        
        # Exit reason analysis
        exit_reasons = df['exit_reason'].value_counts().to_dict()
        
        # Direction analysis
        long_trades = len(df[df['direction'] == 'long'])
        short_trades = len(df[df['direction'] == 'short'])
        long_win_rate = len(df[(df['direction'] == 'long') & (df['profit_loss'] > 0)]) / long_trades if long_trades > 0 else 0
        short_win_rate = len(df[(df['direction'] == 'short') & (df['profit_loss'] > 0)]) / short_trades if short_trades > 0 else 0
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        current_streak = 0
        
        for i, row in df.sort_values('entry_date').iterrows():
            if row['profit_loss'] > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
            
            consecutive_wins = max(consecutive_wins, current_streak if current_streak > 0 else 0)
            consecutive_losses = max(consecutive_losses, -current_streak if current_streak < 0 else 0)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'avg_win_loss_ratio': avg_win_loss_ratio,
            'avg_trade_duration': avg_trade_duration,
            'avg_winning_duration': avg_winning_duration,
            'avg_losing_duration': avg_losing_duration,
            'exit_reasons': exit_reasons,
            'long_trades': long_trades,
            'short_trades': short_trades,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses
        }
