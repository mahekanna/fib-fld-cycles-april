import logging
import io
import asyncio
import pandas as pd
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, filters

from ..core.scanner import FibCycleScanner
from ..models.scan_parameters import ScanParameters
from ..models.scan_result import ScanResult
from ..utils.config import load_config
from ..storage.results_repository import ResultsRepository


class TelegramReporter:
    """
    Telegram bot integration for the Fibonacci Harmonic Trading System.
    Provides commands for scanning symbols, viewing reports, and receiving alerts.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the TelegramReporter.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Set up components
        self.scanner = FibCycleScanner(self.config)
        self.repository = ResultsRepository(self.config)
        
        # Telegram configuration
        self.token = self.config['notifications']['telegram_token']
        self.chat_id = self.config['notifications']['telegram_chat_id']
        
        # Set up application
        self.application = Application.builder().token(self.token).build()
        
        # Register handlers
        self._register_handlers()
        
        # Scheduled tasks
        self.scheduled_jobs = []
    
    def _register_handlers(self):
        """Register command handlers for the bot."""
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        self.application.add_handler(CommandHandler("scan", self.cmd_scan))
        self.application.add_handler(CommandHandler("status", self.cmd_status))
        self.application.add_handler(CommandHandler("reports", self.cmd_reports))
        self.application.add_handler(CommandHandler("top_10", self.cmd_top_10))
        self.application.add_handler(CommandHandler("global_markets", self.cmd_global_markets))
        
        # Add callback query handler for inline buttons
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /start command."""
        user = update.effective_user
        await update.message.reply_text(
            f"👋 Hello {user.first_name}!\n\n"
            f"Welcome to the Fibonacci Harmonic Trading System bot. "
            f"This bot helps you analyze market cycles and generate trading signals.\n\n"
            f"Use /help to see available commands."
        )
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /help command."""
        help_text = (
            "🔍 *Fibonacci Harmonic Trading System Commands*\n\n"
            "/scan <symbol> <timeframe> - Analyze a symbol\n"
            "  Example: `/scan NIFTY daily`\n\n"
            "/status - Show system status\n\n"
            "/reports - Toggle hourly automated reports\n\n"
            "/top_10 - Show top 10 trading opportunities\n\n"
            "/global_markets - Show cycle analysis of global markets\n\n"
            "/help - Show this help message"
        )
        await update.message.reply_text(help_text, parse_mode="Markdown")
    
    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /scan command."""
        # Parse arguments
        args = context.args
        
        if not args or len(args) < 1:
            await update.message.reply_text(
                "❌ Please provide a symbol to scan.\n"
                "Example: `/scan NIFTY daily`"
            )
            return
        
        symbol = args[0].upper()
        interval = args[1] if len(args) > 1 else "daily"
        
        # Send processing message
        message = await update.message.reply_text(f"🔍 Analyzing {symbol} on {interval} timeframe...")
        
        try:
            # Create scan parameters
            params = ScanParameters(
                symbol=symbol,
                exchange=self.config['general']['default_exchange'],
                interval=interval,
                lookback=1000,
                num_cycles=3,
                price_source="close",
                generate_chart=True
            )
            
            # Run the scan
            result = self.scanner.analyze_symbol(params)
            
            # Send the result
            await self.send_scan_result(result, update.effective_chat.id)
            
            # Delete processing message
            await message.delete()
            
        except Exception as e:
            self.logger.error(f"Error in scan command: {e}", exc_info=True)
            await message.edit_text(f"❌ Error analyzing {symbol}: {str(e)}")
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /status command."""
        # Get some basic status info
        reports_active = bool(self.scheduled_jobs)
        
        # Count results in repository
        recent_results = len(self.repository.get_recent_results(hours=24))
        
        # Get server timestamp
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        status_text = (
            "📊 *System Status*\n\n"
            f"🕒 Server Time: {now}\n"
            f"📈 Recent Scans: {recent_results}\n"
            f"🔄 Automated Reports: {'Active ✅' if reports_active else 'Inactive ❌'}\n\n"
        )
        
        await update.message.reply_text(status_text, parse_mode="Markdown")
    
    async def cmd_reports(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Toggle automated reports."""
        if self.scheduled_jobs:
            # Cancel existing jobs
            for job in self.scheduled_jobs:
                job.cancel()
            self.scheduled_jobs = []
            await update.message.reply_text("🔄 Automated reports have been disabled.")
        else:
            # Schedule new jobs
            self.scheduled_jobs = self._schedule_reports(update.effective_chat.id)
            await update.message.reply_text(
                "🔄 Automated reports have been enabled.\n"
                "You will receive hourly market updates during trading hours and a daily summary."
            )
    
    async def cmd_top_10(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show top 10 trading opportunities."""
        message = await update.message.reply_text("🔍 Finding top trading opportunities...")
        
        try:
            # Get symbols from repository
            symbols = self.config['analysis'].get('default_symbols', ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"])
            
            # Create parameters for batch scan
            params_list = [
                ScanParameters(
                    symbol=symbol,
                    exchange=self.config['general']['default_exchange'],
                    interval="daily",
                    lookback=1000,
                    num_cycles=3,
                    price_source="close",
                    generate_chart=False
                )
                for symbol in symbols
            ]
            
            # Run batch scan
            results = self.scanner.scan_batch(params_list)
            
            # Filter and rank results
            buy_signals = self.scanner.filter_signals(results, signal_type="buy")
            sell_signals = self.scanner.filter_signals(results, signal_type="sell")
            
            buy_signals = self.scanner.rank_results(buy_signals, "strength")[:5]
            sell_signals = self.scanner.rank_results(sell_signals, "strength")[:5]
            
            # Create report text
            report_text = "🏆 *Top Trading Opportunities*\n\n"
            
            if buy_signals:
                report_text += "🟢 *Top Buy Signals*\n"
                for i, result in enumerate(buy_signals, 1):
                    confidence = "⭐" * {"low": 1, "medium": 2, "high": 3}.get(result.signal['confidence'], 1)
                    report_text += (
                        f"{i}. *{result.symbol}* ({confidence})\n"
                        f"   Price: {result.price:.2f} | Strength: {result.signal['strength']:.2f}\n"
                        f"   Entry: {result.position_guidance['entry_price']:.2f} | "
                        f"Stop: {result.position_guidance['stop_loss']:.2f} | "
                        f"Target: {result.position_guidance['target_price']:.2f}\n"
                        f"   R/R: {result.position_guidance['risk_reward_ratio']:.2f}\n\n"
                    )
            else:
                report_text += "🟢 *No Buy Signals Found*\n\n"
            
            if sell_signals:
                report_text += "🔴 *Top Sell Signals*\n"
                for i, result in enumerate(sell_signals, 1):
                    confidence = "⭐" * {"low": 1, "medium": 2, "high": 3}.get(result.signal['confidence'], 1)
                    report_text += (
                        f"{i}. *{result.symbol}* ({confidence})\n"
                        f"   Price: {result.price:.2f} | Strength: {result.signal['strength']:.2f}\n"
                        f"   Entry: {result.position_guidance['entry_price']:.2f} | "
                        f"Stop: {result.position_guidance['stop_loss']:.2f} | "
                        f"Target: {result.position_guidance['target_price']:.2f}\n"
                        f"   R/R: {result.position_guidance['risk_reward_ratio']:.2f}\n\n"
                    )
            else:
                report_text += "🔴 *No Sell Signals Found*\n\n"
            
            # Create inline keyboard with scan buttons
            keyboard = []
            for result in buy_signals + sell_signals:
                keyboard.append([
                    InlineKeyboardButton(f"Scan {result.symbol}", callback_data=f"scan_{result.symbol}")
                ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Send the report
            await message.edit_text(report_text, parse_mode="Markdown", reply_markup=reply_markup)
            
        except Exception as e:
            self.logger.error(f"Error in top 10 command: {e}", exc_info=True)
            await message.edit_text(f"❌ Error generating top trading opportunities: {str(e)}")
    
    async def cmd_global_markets(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show analysis of global markets."""
        message = await update.message.reply_text("🌐 Analyzing global markets...")
        
        try:
            # Global market symbols
            global_symbols = [
                ("US", "SPX500"),
                ("US", "NASDAQ"),
                ("US", "DOWJONES"),
                ("US", "VIX"),
                ("JP", "NIKKEI"),
                ("DE", "DAX"),
                ("IN", "NIFTY"),
                ("IN", "BANKNIFTY"),
                ("HK", "HSI")
            ]
            
            # Create parameters for batch scan
            params_list = [
                ScanParameters(
                    symbol=symbol,
                    exchange=exchange,
                    interval="daily",
                    lookback=1000,
                    num_cycles=3,
                    price_source="close",
                    generate_chart=False
                )
                for exchange, symbol in global_symbols
            ]
            
            # Run batch scan
            results = self.scanner.scan_batch(params_list)
            
            # Create summary report
            report_text = "🌐 *Global Markets Cycle Analysis*\n\n"
            
            for result in results:
                if result.success:
                    # Signal icon
                    icon = "🟢" if "buy" in result.signal['signal'] else (
                        "🔴" if "sell" in result.signal['signal'] else "⚪"
                    )
                    
                    # Signal strength indicator
                    strength = abs(result.signal['strength'])
                    bars = "▓" * int(strength * 5)
                    
                    report_text += (
                        f"{icon} *{result.symbol}*: {result.signal['signal'].replace('_', ' ').title()}\n"
                        f"  Strength: {bars} ({strength:.2f})\n"
                        f"  Cycles: {', '.join(map(str, result.detected_cycles))}\n\n"
                    )
                else:
                    report_text += f"❌ *{result.symbol}*: Analysis failed\n\n"
            
            # Create inline keyboard with scan buttons
            keyboard = []
            for result in results:
                if result.success:
                    keyboard.append([
                        InlineKeyboardButton(f"Details: {result.symbol}", callback_data=f"scan_{result.symbol}")
                    ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Send the report
            await message.edit_text(report_text, parse_mode="Markdown", reply_markup=reply_markup)
            
        except Exception as e:
            self.logger.error(f"Error in global markets command: {e}", exc_info=True)
            await message.edit_text(f"❌ Error analyzing global markets: {str(e)}")
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle button callbacks."""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data.startswith("scan_"):
            symbol = data[5:]
            
            # Update message to show processing
            await query.edit_message_text(
                f"🔍 Analyzing {symbol}...",
                parse_mode="Markdown"
            )
            
            try:
                # Create scan parameters
                params = ScanParameters(
                    symbol=symbol,
                    exchange=self.config['general']['default_exchange'],
                    interval="daily",
                    lookback=1000,
                    num_cycles=3,
                    price_source="close",
                    generate_chart=True
                )
                
                # Run the scan
                result = self.scanner.analyze_symbol(params)
                
                # Send the result as a new message
                await self.send_scan_result(result, update.effective_chat.id)
                
                # Restore original message
                await query.edit_message_text(
                    query.message.text,
                    parse_mode="Markdown",
                    reply_markup=query.message.reply_markup
                )
                
            except Exception as e:
                self.logger.error(f"Error in button callback: {e}", exc_info=True)
                await query.edit_message_text(f"❌ Error analyzing {symbol}: {str(e)}")
    
    async def send_scan_result(self, result: ScanResult, chat_id: int) -> None:
        """
        Send a scan result to a Telegram chat.
        
        Args:
            result: ScanResult instance
            chat_id: Telegram chat ID to send to
        """
        if not result.success:
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Error analyzing {result.symbol}: {result.error}"
            )
            return
        
        # Create result message
        signal_emoji = "🟢" if "buy" in result.signal['signal'] else (
            "🔴" if "sell" in result.signal['signal'] else "⚪"
        )
        
        confidence = "⭐" * {"low": 1, "medium": 2, "high": 3}.get(result.signal['confidence'], 1)
        
        message_text = (
            f"{signal_emoji} *{result.symbol} Analysis* {confidence}\n\n"
            f"*Signal:* {result.signal['signal'].replace('_', ' ').upper()}\n"
            f"*Price:* {result.price:.2f}\n"
            f"*Strength:* {result.signal['strength']:.2f}\n"
            f"*Alignment:* {result.signal['alignment']:.2f}\n\n"
            
            f"*Position Guidance:*\n"
            f"Entry: {result.position_guidance['entry_price']:.2f}\n"
            f"Stop Loss: {result.position_guidance['stop_loss']:.2f}\n"
            f"Target: {result.position_guidance['target_price']:.2f}\n"
            f"R/R Ratio: {result.position_guidance['risk_reward_ratio']:.2f}\n\n"
            
            f"*Detected Cycles:* {', '.join(map(str, result.detected_cycles))}\n"
        )
        
        # Create keyboard for actions
        keyboard = [
            [InlineKeyboardButton("Update Analysis", callback_data=f"scan_{result.symbol}")],
            [InlineKeyboardButton("Daily Chart", callback_data=f"chart_{result.symbol}_daily"),
             InlineKeyboardButton("4h Chart", callback_data=f"chart_{result.symbol}_4h")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Check if we have a chart image
        if result.chart_image:
            # Convert image to bytes
            image_bytes = io.BytesIO()
            plt.figure(figsize=(10, 6))
            plt.imshow(result.chart_image)
            plt.axis('off')
            plt.savefig(image_bytes, format='PNG', bbox_inches='tight')
            image_bytes.seek(0)
            
            # Send image with caption
            await self.application.bot.send_photo(
                chat_id=chat_id,
                photo=image_bytes,
                caption=message_text,
                parse_mode="Markdown",
                reply_markup=reply_markup
            )
        else:
            # Send text only
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=message_text,
                parse_mode="Markdown",
                reply_markup=reply_markup
            )
    
    def _schedule_reports(self, chat_id: int) -> List:
        """
        Schedule automated reports.
        
        Args:
            chat_id: Telegram chat ID to send reports to
            
        Returns:
            List of scheduled job handlers
        """
        jobs = []
        
        # Hourly market updates during trading hours
        for hour in range(9, 16):  # 9 AM to 3 PM
            job = self.application.job_queue.run_daily(
                callback=self._send_hourly_report,
                time=time(hour=hour, minute=0),
                days=(0, 1, 2, 3, 4),  # Monday to Friday
                chat_id=chat_id,
                name=f"hourly_{hour}"
            )
            jobs.append(job)
        
        # End of day report
        job = self.application.job_queue.run_daily(
            callback=self._send_daily_report,
            time=time(hour=15, minute=30),  # 3:30 PM
            days=(0, 1, 2, 3, 4),  # Monday to Friday
            chat_id=chat_id,
            name="daily_report"
        )
        jobs.append(job)
        
        return jobs
    
    async def _send_hourly_report(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send hourly market update."""
        chat_id = context.job.chat_id
        hour = datetime.now().hour
        
        await self.application.bot.send_message(
            chat_id=chat_id,
            text=f"🕒 *Hourly Market Update ({hour}:00)*\n\n"
                 f"Preparing market analysis..."
        )
        
        # Here you would add code to generate and send the hourly report
    
    async def _send_daily_report(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send end of day report."""
        chat_id = context.job.chat_id
        
        await self.application.bot.send_message(
            chat_id=chat_id,
            text="📊 *End of Day Market Report*\n\n"
                 f"Preparing comprehensive market analysis..."
        )
        
        # Here you would add code to generate and send the daily report
    
    def run(self):
        """Run the Telegram bot."""
        self.logger.info("Starting Telegram bot")
        self.application.run_polling()


def run_telegram_bot(config_path: str = "config/default_config.json"):
    """
    Run the Telegram bot.
    
    Args:
        config_path: Path to configuration file
    """
    # Set up logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    # Create and run bot
    bot = TelegramReporter(config_path)
    bot.run()


if __name__ == "__main__":
    run_telegram_bot()
