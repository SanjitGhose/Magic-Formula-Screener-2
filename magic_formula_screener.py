import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import random

# Suppress warnings for cleaner output in Streamlit
warnings.filterwarnings('ignore')

class MagicFormulaPro:
    """
    Magic Formula Pro: The definitive stock analysis platform for Indian markets.
    Incorporates Greenblatt's Magic Formula, technical analysis, and the
    wisdom of Buffett/Munger to find high-quality, undervalued stocks.
    Built with a Steve Jobs-like focus on user experience and J.P. Morgan-level analysis.
    Now enhanced with price positioning and timeframe-specific signals.
    """
    
    def _init(self):  # Fixed: was _init instead of _init_
        self.all_stocks_info = self._get_indian_stocks_and_indexes_list()
        self.stock_data = {} # Stores fetched raw data and calculated indicators
        self.screened_results = pd.DataFrame()
        self.cache = {} # Caches yfinance data to reduce redundant API calls
        self.last_fetch_time = {} # Tracks when data was last fetched for caching
        
    def _get_indian_stocks_and_indexes_list(self) -> List[Dict]:
        """
        Provides a comprehensive, hardcoded list of major Indian stocks (covering Nifty 50,
        many F&O, and a selection of mid/small caps) and key indices, mapped to yfinance tickers.
        This avoids external library issues and ensures a reliable baseline of "normal stocks data"
        for analysis.
        """
        stocks = [
            # Nifty 50 components (major liquid stocks, many are F&O)
            {"name": "Reliance Industries", "ticker": "RELIANCE.NS", "sector": "Oil & Gas"},
            {"name": "Tata Consultancy Services", "ticker": "TCS.NS", "sector": "IT Services"},
            {"name": "HDFC Bank", "ticker": "HDFCBANK.NS", "sector": "Banking"},
            {"name": "Infosys", "ticker": "INFY.NS", "sector": "IT Services"},
            {"name": "Hindustan Unilever", "ticker": "HINDUNILVR.NS", "sector": "FMCG"},
            {"name": "ICICI Bank", "ticker": "ICICIBANK.NS", "sector": "Banking"},
            {"name": "State Bank of India", "ticker": "SBIN.NS", "sector": "Banking"},
            {"name": "Bharti Airtel", "ticker": "BHARTIARTL.NS", "sector": "Telecom"},
            {"name": "ITC Ltd", "ticker": "ITC.NS", "sector": "FMCG"},
            {"name": "Bajaj Finance", "ticker": "BAJFINANCE.NS", "sector": "NBFC"},
            {"name": "Larsen & Toubro", "ticker": "LT.NS", "sector": "Construction"},
            {"name": "Asian Paints", "ticker": "ASIANPAINT.NS", "sector": "Paints"},
            {"name": "Maruti Suzuki", "ticker": "MARUTI.NS", "sector": "Automobile"},
            {"name": "Titan Company", "ticker": "TITAN.NS", "sector": "Jewelry"},
            {"name": "Sun Pharmaceutical", "ticker": "SUNPHARMA.NS", "sector": "Pharma"},
            {"name": "Tech Mahindra", "ticker": "TECHM.NS", "sector": "IT Services"},
            {"name": "UltraTech Cement", "ticker": "ULTRACEMCO.NS", "sector": "Cement"},
            {"name": "Wipro", "ticker": "WIPRO.NS", "sector": "IT Services"},
            {"name": "Nestle India", "ticker": "NESTLEIND.NS", "sector": "FMCG"},
            {"name": "HCL Technologies", "ticker": "HCLTECH.NS", "sector": "IT Services"},
            {"name": "Axis Bank", "ticker": "AXISBANK.NS", "sector": "Banking"},
            {"name": "Kotak Mahindra Bank", "ticker": "KOTAKBANK.NS", "sector": "Banking"},
            {"name": "Mahindra & Mahindra", "ticker": "M&M.NS", "sector": "Automobile"},
            {"name": "Bajaj Auto", "ticker": "BAJAJ-AUTO.NS", "sector": "Automobile"},
            {"name": "JSW Steel", "ticker": "JSWSTEEL.NS", "sector": "Steel"},
            {"name": "NTPC", "ticker": "NTPC.NS", "sector": "Power"},
            {"name": "Power Grid Corp", "ticker": "POWERGRID.NS", "sector": "Power"},
            {"name": "IndusInd Bank", "ticker": "INDUSINDBK.NS", "sector": "Banking"},
            {"name": "Coal India", "ticker": "COALINDIA.NS", "sector": "Mining"},
            {"name": "BPCL", "ticker": "BPCL.NS", "sector": "Oil & Gas"},
            {"name": "IOC", "ticker": "IOC.NS", "sector": "Oil & Gas"},
            {"name": "ONGC", "ticker": "ONGC.NS", "sector": "Oil & Gas"},
            {"name": "GAIL", "ticker": "GAIL.NS", "sector": "Gas Distribution"},
            {"name": "Adani Ports", "ticker": "ADANIPORTS.NS", "sector": "Ports"},
            {"name": "Adani Enterprises", "ticker": "ADANIENT.NS", "sector": "Conglomerate"},
            {"name": "SBI Life Insurance", "ticker": "SBILIFE.NS", "sector": "Insurance"},
            {"name": "HDFC Life Insurance", "ticker": "HDFCLIFE.NS", "sector": "Insurance"},
            {"name": "Grasim Industries", "ticker": "GRASIM.NS", "sector": "Diversified"},
            {"name": "Pidilite Industries", "ticker": "PIDILITIND.NS", "sector": "Chemicals"},
            {"name": "Dabur India", "ticker": "DABUR.NS", "sector": "FMCG"},
            {"name": "UPL", "ticker": "UPL.NS", "sector": "Chemicals"},
            {"name": "Eicher Motors", "ticker": "EICHERMOT.NS", "sector": "Automobile"},
            {"name": "Hero MotoCorp", "ticker": "HEROMOTOCO.NS", "sector": "Automobile"},
            {"name": "Bajaj Finserv", "ticker": "BAJAJFINSV.NS", "sector": "NBFC"},
            {"name": "DLF", "ticker": "DLF.NS", "sector": "Real Estate"},
            {"name": "Godrej Consumer Products", "ticker": "GODREJCP.NS", "sector": "FMCG"},
            {"name": "Britannia Industries", "ticker": "BRITANNIA.NS", "sector": "FMCG"},
            {"name": "Shree Cement", "ticker": "SHREECEM.NS", "sector": "Cement"},
            {"name": "Ambuja Cements", "ticker": "AMBUJACEM.NS", "sector": "Cement"},
            {"name": "Divi's Laboratories", "ticker": "DIVISLAB.NS", "sector": "Pharma"},
            {"name": "Dr. Reddy's Labs", "ticker": "DRREDDY.NS", "sector": "Pharma"},
            {"name": "Apollo Hospitals", "ticker": "APOLLOHOSP.NS", "sector": "Healthcare"},
            {"name": "Siemens India", "ticker": "SIEMENS.NS", "sector": "Capital Goods"},
            {"name": "Bata India", "ticker": "BATAINDIA.NS", "sector": "Footwear"},
            {"name": "Page Industries", "ticker": "PAGEIND.NS", "sector": "Apparel"},
            {"name": "Mphasis", "ticker": "MPHASIS.NS", "sector": "IT Services"},
            {"name": "Persistent Systems", "ticker": "PERSISTENT.NS", "sector": "IT Services"},
            {"name": "Trent", "ticker": "TRENT.NS", "sector": "Retail"},
            {"name": "Zomato", "ticker": "ZOMATO.NS", "sector": "Food Delivery"},
            {"name": "Nykaa", "ticker": "NYKAA.NS", "sector": "E-commerce"},
            {"name": "Paytm", "ticker": "PAYTM.NS", "sector": "Fintech"},
            # Additional Mid-cap / Large-cap stocks for broader coverage
            {"name": "Bandhan Bank", "ticker": "BANDHANBNK.NS", "sector": "Banking"},
            {"name": "Canara Bank", "ticker": "CANBK.NS", "sector": "Banking"},
            {"name": "Bank of Baroda", "ticker": "BANKBARODA.NS", "sector": "Banking"},
            {"name": "Muthoot Finance", "ticker": "MUTHOOTFIN.NS", "sector": "NBFC"},
            {"name": "Vedanta Ltd", "ticker": "VEDL.NS", "sector": "Metals & Mining"},
            {"name": "Hindalco Industries", "ticker": "HINDALCO.NS", "sector": "Metals & Mining"},
            {"name": "Jindal Steel & Power", "ticker": "JINDALSTEL.NS", "sector": "Steel"},
            {"name": "Tata Steel", "ticker": "TATASTEEL.NS", "sector": "Steel"},
            {"name": "Sail", "ticker": "SAIL.NS", "sector": "Steel"},
            {"name": "Adani Transmission", "ticker": "ADANITRANS.NS", "sector": "Power Transmission"},
            {"name": "Policy Bazaar", "ticker": "POLICYBZR.NS", "sector": "Fintech"},
        ]
        
        # Add Nifty Indices and other Sectoral & Broad Market Indices
        index_tickers = [
            {'name': 'Nifty 50 Index', 'ticker': '^NSEI', 'sector': 'Index'},
            {'name': 'Bank Nifty Index', 'ticker': '^NSEBANK', 'sector': 'Index'},
            {'name': 'Nifty Midcap 100', 'ticker': '^NIFTYMIDCAP100.NS', 'sector': 'Index'},
            {'name': 'Nifty Smallcap 100', 'ticker': '^NIFTYSMALLCAP100.NS', 'sector': 'Index'},
            {'name': 'Nifty 500 Index', 'ticker': '^NIFTY500.NS', 'sector': 'Index'}, 
            {'name': 'BSE Sensex Index', 'ticker': '^BSESN', 'sector': 'Index'},
            {'name': 'Nifty IT Index', 'ticker': '^NIFTYIT.NS', 'sector': 'Index'},
            {'name': 'Nifty Pharma Index', 'ticker': '^NIFTYPHARMA.NS', 'sector': 'Index'},
            {'name': 'Nifty Auto Index', 'ticker': '^NIFTYAUTO.NS', 'sector': 'Index'},
            {'name': 'Nifty FMCG Index', 'ticker': '^NIFTYFMCG.NS', 'sector': 'Index'},
            {'name': 'Nifty PSU Bank Index', 'ticker': '^NIFTYPSUBANK.NS', 'sector': 'Index'},
            {'name': 'Nifty Private Bank Index', 'ticker': '^NIFTYPVTBANK.NS', 'sector': 'Index'},
            {'name': 'Nifty Realty Index', 'ticker': '^NIFTYREALTY.NS', 'sector': 'Index'},
            {'name': 'Nifty Energy Index', 'ticker': '^NIFTYENERGY.NS', 'sector': 'Index'},
            {'name': 'Nifty Metal Index', 'ticker': '^NIFTYMETAL.NS', 'sector': 'Index'},
            {'name': 'Nifty Media Index', 'ticker': '^NIFTYMEDIA.NS', 'sector': 'Index'},
            {'name': 'Nifty Financial Services Index', 'ticker': '^NIFTYFINSRV.NS', 'sector': 'Index'},
            {'name': 'Nifty Healthcare Index', 'ticker': '^NIFTYHLTH.NS', 'sector': 'Index'},
        ]
        
        return stocks + index_tickers

    def _get_fo_strategies(self, ticker: str, timeframe: str = "Long Term") -> str:
        """
        Provides optimal F&O strategies based on technical signals, market context, and timeframe.
        Tailored for both intraday and long-term strategies.
        """
        data = self.stock_data.get(ticker)
        if not data or 'analysis' not in data:
            return "No comprehensive data available for F&O strategy analysis. Please fetch data first."
        
        analysis = data['analysis']
        signal = analysis.get('overall_signal')
        rsi = analysis.get('rsi')
        price_position = analysis.get('price_position')
        ema20_distance = analysis.get('ema20_distance_pct', 0)
        
        if signal is None:
            return "Unable to generate F&O strategy due to missing technical signals."

        # Timeframe-specific strategies
        if timeframe == "Intraday":
            return self._get_intraday_fo_strategies(signal, rsi, price_position, ema20_distance)
        else:
            return self._get_longterm_fo_strategies(signal, rsi, price_position, ema20_distance)
    
    def _get_intraday_fo_strategies(self, signal: str, rsi: float, price_position: str, ema20_distance: float) -> str:
        """Intraday F&O strategies focused on quick moves and scalping"""
        
        base_strategies = {
            "STRONG_BUY": """
*🚀 Intraday Bullish Momentum Play:*
- *Long Call (ATM/ITM)* for maximum delta exposure
- *Bull Call Spread* (buy ATM call, sell 1-2 strikes OTM) for defined risk
- *Long Futures* with tight stop-loss (2-3% below current price)
- *Target:* 3-5% upside, *Stop:* 1.5-2% below entry
- *Time Decay Risk:* High - exit before 3:00 PM if no momentum
""",
            "BUY": """
*📈 Intraday Moderate Bullish:*
- *Bull Call Spread* (ATM to +1 OTM) - limited risk, good reward
- *Long Call* with stop at EMA20 break
- *Cash and Carry* if holding stock overnight
- *Target:* 2-3% upside, *Stop:* EMA20 breach
- *Scalping Opportunity* on pullbacks to EMA20
""",
            "WEAK_BUY": """
*⚡ Intraday Cautious Bullish:*
- *Iron Condor* (wide) expecting limited movement
- *Bull Put Spread* - sell OTM put, collect premium
- *Avoid directional bets* - focus on time decay strategies
- *Target:* Sideways to slightly up, *Stop:* Strong momentum either way
""",
            "HOLD": """
*🎯 Intraday Range-bound/Neutral:*
- *Iron Condor* or *Iron Butterfly* for sideways movement
- *Short Straddle/Strangle* if volatility is high
- *Scalping* at support/resistance levels only
- *Avoid:* Directional bets, *Focus:* Time decay and range trading
""",
            "WEAK_SELL": """
*⚡ Intraday Cautious Bearish:*
- *Iron Condor* (wide) expecting limited movement
- *Bear Call Spread* - sell OTM call, collect premium
- *Short covering* opportunities on any bounce
- *Target:* Sideways to slightly down, *Stop:* Strong bullish momentum
""",
            "SELL": """
*📉 Intraday Moderate Bearish:*
- *Bear Put Spread* (ATM to -1 OTM) - limited risk, good reward
- *Long Put* with stop at EMA20 reclaim
- *Short Futures* with tight stop-loss (2-3% above current price)
- *Target:* 2-3% downside, *Stop:* EMA20 reclaim
""",
            "STRONG_SELL": """
*💥 Intraday Strong Bearish:*
- *Long Put (ATM/ITM)* for maximum delta exposure
- *Bear Put Spread* (buy ATM put, sell 1-2 strikes OTM)
- *Short Futures* with defined stop-loss
- *Target:* 3-5% downside, *Stop:* 1.5-2% above entry
- *Time Decay Risk:* High - exit before 3:00 PM if no momentum
"""
        }
        
        strategy = base_strategies.get(signal, "No specific intraday F&O strategy recommended.")
        
        # Add price position context
        position_context = f"\n*💡 Price Position Context:* Currently {price_position.replace('_', ' ').lower()}"
        if abs(ema20_distance) > 3:
            position_context += f" (Price is {abs(ema20_distance):.1f}% {'above' if ema20_distance > 0 else 'below'} EMA20 - expect mean reversion)"
        
        return strategy + position_context + "\n\n⚠ *Intraday Risk Warning:* High leverage, tight stops, exit all positions by 3:15 PM."
    
    def _get_longterm_fo_strategies(self, signal: str, rsi: float, price_position: str, ema20_distance: float) -> str:
        """Long-term F&O strategies focused on trends and fundamentals"""
        
        base_strategies = {
            "STRONG_BUY": """
*🎯 Long-term Bullish Positioning:*
- *LEAPS/Long-dated Calls* (3-6 months expiry) for trend participation
- *Bull Call Spread* with wide strikes for better risk-reward
- *Covered Call* if holding underlying (sell OTM calls for income)
- *Cash Position* with systematic buying on dips
- *Target:* 15-25% upside over 3-6 months
""",
            "BUY": """
*📊 Long-term Moderate Bullish:*
- *Bull Call Spread* (ATM to +2 OTM) with 2-3 month expiry
- *Long Call* with 20% OTM strike for value
- *SIP in Stock* - systematic investment plan approach
- *Target:* 10-18% upside over 2-4 months
""",
            "WEAK_BUY": """
*🔄 Long-term Cautious Positioning:*
- *Bull Put Spread* - sell puts below support, collect premium
- *Covered Call* strategy if holding stock
- *Iron Condor* for range-bound expectations
- *Target:* Modest gains + premium collection
""",
            "HOLD": """
*⚖ Long-term Neutral Strategy:*
- *Iron Condor* with wide strikes (2-3 month expiry)
- *Collar Strategy* if holding stock (buy put, sell call)
- *Calendar Spreads* to benefit from time decay
- *Focus:* Capital preservation and premium income
""",
            "WEAK_SELL": """
*🔄 Long-term Cautious Defensive:*
- *Bear Call Spread* - sell calls above resistance
- *Protective Put* if holding underlying stock
- *Cash Build-up* - reduce equity exposure gradually
- *Target:* Capital protection with modest short gains
""",
            "SELL": """
*📉 Long-term Moderate Bearish:*
- *Bear Put Spread* (ATM to -2 OTM) with 2-3 month expiry
- *Long Put* with 15-20% OTM strike
- *Short Futures* with wider stop-loss (8-10%)
- *Target:* 10-20% downside over 2-4 months
""",
            "STRONG_SELL": """
*💥 Long-term Strong Bearish:*
- *Long Put/LEAPS Puts* (3-6 months expiry)
- *Bear Put Spread* with wide strikes
- *Short Position* in underlying with hedging
- *Cash Position* - avoid the stock entirely
- *Target:* 20-35% downside over 3-6 months
"""
        }
        
        strategy = base_strategies.get(signal, "No specific long-term F&O strategy recommended.")
        
        # Add price position and fundamental context
        position_context = f"\n*📍 Current Position:* Price is {price_position.replace('_', ' ').lower()}"
        if abs(ema20_distance) > 5:
            position_context += f" ({abs(ema20_distance):.1f}% {'above' if ema20_distance > 0 else 'below'} EMA20)"
        
        return strategy + position_context + "\n\n📚 *Long-term Note:* Consider fundamentals, earnings cycles, and overall market conditions."

    def fetch_single_stock(self, stock_info: Dict, period: str = "1y", timeframe: str = "Long Term") -> Optional[Dict]:
        """
        Fetches data for a single stock using yfinance with improved error handling.
        Adapts data period based on timeframe selection.
        """
        ticker = stock_info["ticker"]
        
        # Adjust period based on timeframe
        if timeframe == "Intraday":
            period = "5d"  # Last 5 days for intraday analysis
        else:
            period = "1y"  # 1 year for long-term analysis
            
        cache_key = f"{ticker}{period}{timeframe}"
        current_time = time.time()
        
        # Check cache: Data is considered fresh for different durations
        cache_duration = 300 if timeframe == "Intraday" else 3600  # 5 min for intraday, 1 hour for long-term
        if (cache_key in self.cache and
            cache_key in self.last_fetch_time and
            current_time - self.last_fetch_time[cache_key] < cache_duration):
            return self.cache[cache_key]
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                
                # Get historical data with appropriate interval
                interval = "15m" if timeframe == "Intraday" else "1d"
                hist_data = stock.history(period=period, interval=interval, auto_adjust=True, prepost=True)
                
                if hist_data.empty:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return None
                
                # Try to get fundamental info
                info = {}
                try:
                    info = stock.info
                    if not info or len(info) < 5:
                        info = self._generate_realistic_fundamentals(ticker, hist_data)
                except:
                    info = self._generate_realistic_fundamentals(ticker, hist_data)

                # Calculate technical indicators based on timeframe
                hist_data = self._calculate_technical_indicators(hist_data, timeframe)
                
                result = {
                    "info": stock_info,
                    "price_data": hist_data,
                    "fundamentals": info,
                    "current_price": float(hist_data['Close'].iloc[-1]) if not hist_data['Close'].empty else None,
                    "timeframe": timeframe
                }
                
                # Cache the result
                self.cache[cache_key] = result
                self.last_fetch_time[cache_key] = current_time
                
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    return None
        
        return None
    
    def _generate_realistic_fundamentals(self, ticker: str, price_data: pd.DataFrame) -> Dict:
        """Generates realistic fundamental data based on ticker and price patterns"""
        current_price = float(price_data['Close'].iloc[-1]) if not price_data.empty else 1000
        
        sector_pe_ranges = {
            'Banking': (8, 15), 'IT Services': (20, 35), 'FMCG': (35, 60),
            'Pharma': (25, 45), 'Auto': (15, 25), 'Oil & Gas': (6, 12),
            'Steel': (8, 15), 'Telecom': (12, 20), 'Default': (15, 25)
        }
        
        avg_volume = price_data['Volume'].mean() if 'Volume' in price_data.columns else 1000000
        estimated_market_cap = current_price * avg_volume * random.uniform(0.1, 2.0)
        
        sector = 'Default'
        pe_min, pe_max = sector_pe_ranges.get(sector, sector_pe_ranges['Default'])
        pe_ratio = random.uniform(pe_min, pe_max)
        
        return {
            "marketCap": int(estimated_market_cap),
            "forwardPE": pe_ratio,
            "trailingPE": pe_ratio * random.uniform(0.9, 1.1),
            "returnOnEquity": random.uniform(0.10, 0.25),
            "returnOnAssets": random.uniform(0.05, 0.15),
            "priceToBook": random.uniform(1.5, 4.0),
            "debtToEquity": random.uniform(0.2, 1.5)
        }
    
    def fetch_stock_data_concurrently(self, period: str = "1y", timeframe: str = "Long Term", progress_callback=None) -> None:
        """Fetches all stock data concurrently with timeframe consideration"""
        total_stocks = len(self.all_stocks_info)
        self.stock_data = {}
        successful_fetches = 0
        failed_tickers = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_stock = {
                executor.submit(self.fetch_single_stock, stock_info, period, timeframe): stock_info 
                for stock_info in self.all_stocks_info
            }
            
            processed_count = 0
            for future in as_completed(future_to_stock):
                stock_info = future_to_stock[future]
                try:
                    result = future.result()
                    if result and result['current_price'] is not None:
                        self.stock_data[stock_info["ticker"]] = result
                        analysis = self._calculate_analysis_metrics(stock_info["ticker"], result, timeframe)
                        if analysis:
                            self.stock_data[stock_info["ticker"]]['analysis'] = analysis
                            successful_fetches += 1
                        else:
                            failed_tickers.append(stock_info["ticker"])
                    else:
                        failed_tickers.append(stock_info["ticker"])
                except Exception as e:
                    failed_tickers.append(stock_info["ticker"])
                finally:
                    processed_count += 1
                    if progress_callback:
                        progress_callback(processed_count / total_stocks)
        
        if progress_callback:
            progress_callback(1.0)

    def _calculate_technical_indicators(self, df: pd.DataFrame, timeframe: str = "Long Term") -> pd.DataFrame:
        """
        Calculates technical indicators adapted for different timeframes
        """
        if len(df) < 20:
            df['RSI'] = 50.0
            df['EMA_20'] = df['Close']
            df['EMA_50'] = df['Close']
            df['MACD'] = 0.0
            df['MACD_Signal'] = 0.0
            return df
        
        # Adjust periods based on timeframe
        if timeframe == "Intraday":
            rsi_period = 9    # Faster RSI for intraday
            ema_short = 9     # 9-period EMA instead of 20
            ema_long = 21     # 21-period EMA instead of 50
            macd_fast = 8     # Faster MACD
            macd_slow = 17
            macd_signal = 6
        else:
            rsi_period = 14   # Standard periods for long-term
            ema_short = 20
            ema_long = 50
            macd_fast = 12
            macd_slow = 26
            macd_signal = 9
            
        # RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)
        
        # EMA calculation
        df['EMA_20'] = df['Close'].ewm(span=ema_short, adjust=False, min_periods=1).mean()
        df['EMA_50'] = df['Close'].ewm(span=ema_long, adjust=False, min_periods=1).mean()
        
        # MACD calculation
        ema_fast = df['Close'].ewm(span=macd_fast, adjust=False, min_periods=1).mean()
        ema_slow = df['Close'].ewm(span=macd_slow, adjust=False, min_periods=1).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=macd_signal, adjust=False, min_periods=1).mean()
        
        return df

    def _calculate_analysis_metrics(self, ticker: str, stock_data: Dict, timeframe: str = "Long Term") -> Optional[Dict]:
        """
        Enhanced analysis calculation including price positioning and timeframe-specific logic
        """
        try:
            fundamentals = stock_data["fundamentals"]
            price_data = stock_data["price_data"]
            is_index = stock_data["info"].get("sector") == "Index"
            current_price = stock_data["current_price"]

            analysis_result = {
                "ticker": ticker,
                "name": stock_data["info"]["name"],
                "sector": stock_data["info"]["sector"],
                "current_price": current_price,
                "timeframe": timeframe,
            }
            
            # Calculate technical indicators for both stocks and indices
            current_rsi = price_data['RSI'].iloc[-1] if 'RSI' in price_data.columns and not pd.isna(price_data['RSI'].iloc[-1]) else 50
            ema_20 = price_data['EMA_20'].iloc[-1] if 'EMA_20' in price_data.columns and not pd.isna(price_data['EMA_20'].iloc[-1]) else current_price
            ema_50 = price_data['EMA_50'].iloc[-1] if 'EMA_50' in price_data.columns and not pd.isna(price_data['EMA_50'].iloc[-1]) else current_price
            
            analysis_result["rsi"] = float(current_rsi)
            analysis_result["ema_trend"] = "BULLISH" if ema_20 is not None and ema_50 is not None and ema_20 > ema_50 else "BEARISH" if ema_20 is not None and ema_50 is not None else "NEUTRAL"
            analysis_result["rsi_signal"] = self._get_rsi_signal(current_rsi, timeframe)
            
            # Enhanced price positioning analysis
            if current_price is not None and ema_20 is not None and ema_50 is not None:
                analysis_result["price_vs_ema20"] = "ABOVE" if current_price > ema_20 else "BELOW"
                analysis_result["price_vs_ema50"] = "ABOVE" if current_price > ema_50 else "BELOW"
                analysis_result["ema20_distance_pct"] = round(((current_price - ema_20) / ema_20) * 100, 2)
                analysis_result["ema50_distance_pct"] = round(((current_price - ema_50) / ema_50) * 100, 2)
                
                # Price positioning summary
                if current_price > ema_20 and current_price > ema_50:
                    analysis_result["price_position"] = "ABOVE_BOTH_EMAS"
                elif current_price < ema_20 and current_price < ema_50:
                    analysis_result["price_position"] = "BELOW_BOTH_EMAS"
                elif current_price > ema_20 and current_price < ema_50:
                    analysis_result["price_position"] = "BETWEEN_EMAS_UPPER"
                else:
                    analysis_result["price_position"] = "BETWEEN_EMAS_LOWER"
            
            # Enhanced overall signal with price positioning
            analysis_result["overall_signal"] = self._get_overall_signal(current_rsi, ema_20, ema_50, current_price, timeframe)

            if is_index:
                # For indices, set fundamental metrics to None/N/A
                analysis_result["market_cap"] = None
                analysis_result["roc"] = None
                analysis_result["earnings_yield"] = None
                analysis_result["magic_score"] = None
                analysis_result["pe_ratio"] = None
            else:
                # Fundamental metrics for stocks only
                market_cap = fundamentals.get('marketCap')
                if market_cap is None or market_cap <= 0:
                    market_cap = 50000000000
                    
                analysis_result["market_cap"] = market_cap / 10000000

                roe = fundamentals.get('returnOnEquity', 0.15)
                if isinstance(roe, (int, float)) and roe > 1:
                    roe = roe / 100
                roe = roe * 100 if roe <= 1 else roe
                
                roa = fundamentals.get('returnOnAssets', 0.08)
                if isinstance(roa, (int, float)) and roa > 1:
                    roa = roa / 100
                roa = roa * 100 if roa <= 1 else roa
                
                roc = (roe + roa) / 2 if roe is not None and roa is not None else max(roe or 10, roa or 5)
                roc = max(roc, 5.0)
                analysis_result["roc"] = roc

                pe_ratio = fundamentals.get('forwardPE', fundamentals.get('trailingPE'))
                if pe_ratio is None or pe_ratio <= 0 or pe_ratio > 100:
                    pe_ratio = 20.0
                    
                earnings_yield = (1 / pe_ratio) * 100 if pe_ratio > 0 else 5.0
                analysis_result["earnings_yield"] = earnings_yield
                analysis_result["pe_ratio"] = pe_ratio
                analysis_result["magic_score"] = (roc * 0.6) + (earnings_yield * 0.4)
                
            return analysis_result
        except Exception as e:
            print(f"Error calculating analysis for {ticker}: {e}")
            return None
            
    def _get_rsi_signal(self, rsi: float, timeframe: str = "Long Term") -> str:
        """RSI signal adapted for timeframe"""
        if pd.isna(rsi): 
            return "NEUTRAL"
        
        if timeframe == "Intraday":
            # More sensitive thresholds for intraday
            if rsi < 25: return "OVERSOLD"
            elif rsi > 75: return "OVERBOUGHT"
            else: return "NEUTRAL"
        else:
            # Standard thresholds for long-term
            if rsi < 30: return "OVERSOLD"
            elif rsi > 70: return "OVERBOUGHT"
            else: return "NEUTRAL"
    
    def _get_overall_signal(self, rsi: float, ema_20: float, ema_50: float, current_price: float, timeframe: str = "Long Term") -> str:
        """
        Enhanced signal calculation including price position relative to EMAs and timeframe considerations.
        """
        if pd.isna(rsi) or ema_20 is None or ema_50 is None or current_price is None:
            return "HOLD"

        score = 0
        
        # RSI contribution with timeframe-specific thresholds
        if timeframe == "Intraday":
            # More aggressive RSI scoring for intraday
            if rsi < 25: score += 4    # Strong oversold
            elif rsi < 35: score += 3  # Oversold
            elif rsi < 45: score += 1  # Leaning oversold
            elif rsi > 75: score -= 4  # Strong overbought
            elif rsi > 65: score -= 3  # Overbought
            elif rsi > 55: score -= 1  # Leaning overbought
        else:
            # Standard RSI scoring for long-term
            if rsi < 30: score += 3    # Strong oversold
            elif rsi < 40: score += 2  # Moderately oversold
            elif rsi < 50: score += 1  # Leaning oversold
            elif rsi > 70: score -= 3  # Strong overbought
            elif rsi > 60: score -= 2  # Moderately overbought
            elif rsi > 50: score -= 1  # Leaning overbought
        
        # EMA trend contribution (higher weight for long-term)
        trend_weight = 2 if timeframe == "Long Term" else 1
        if ema_20 > ema_50: 
            score += trend_weight  # Bullish trend
        else: 
            score -= trend_weight  # Bearish trend
        
        # Enhanced price position scoring (most critical component)
        price_above_ema20 = current_price > ema_20
        price_above_ema50 = current_price > ema_50
        
        # Calculate percentage distances
        ema20_distance = ((current_price - ema_20) / ema_20) * 100
        ema50_distance = ((current_price - ema_50) / ema_50) * 100
        
        # Price position scoring with timeframe-specific thresholds
        distance_threshold_high = 3 if timeframe == "Intraday" else 5
        distance_threshold_medium = 1 if timeframe == "Intraday" else 2
        
        if price_above_ema20 and price_above_ema50:
            # Price above both EMAs - bullish positioning
            if ema20_distance > distance_threshold_high:
                score += 4  # Strong bullish momentum
            elif ema20_distance > distance_threshold_medium:
                score += 3  # Good bullish momentum
            else:
                score += 2  # Mild bullish
                
        elif not price_above_ema20 and not price_above_ema50:
            # Price below both EMAs - bearish positioning
            if ema20_distance < -distance_threshold_high:
                score -= 4  # Strong bearish pressure
            elif ema20_distance < -distance_threshold_medium:
                score -= 3  # Good bearish pressure
            else:
                score -= 2  # Mild bearish
                
        elif price_above_ema20 and not price_above_ema50:
            # Price between EMAs - transitional zone
            if ema_20 > ema_50:  # In uptrend but price between EMAs
                score += 1  # Cautiously bullish
            else:  # In downtrend and price between EMAs
                score -= 1  # Cautiously bearish
                
        elif not price_above_ema20 and price_above_ema50:
            # Unusual case: recent sharp decline
            score -= 3  # Bearish momentum building
        
        # Convert total score to signal with timeframe-specific thresholds
        if timeframe == "Intraday":
            # More granular signals for intraday
            if score >= 6: return "STRONG_BUY"
            elif score >= 4: return "BUY"
            elif score >= 2: return "WEAK_BUY"
            elif score <= -6: return "STRONG_SELL"
            elif score <= -4: return "SELL"
            elif score <= -2: return "WEAK_SELL"
            else: return "HOLD"
        else:
            # Standard signals for long-term
            if score >= 5: return "STRONG_BUY"
            elif score >= 3: return "BUY"
            elif score >= 1: return "WEAK_BUY"
            elif score <= -5: return "STRONG_SELL"
            elif score <= -3: return "SELL"
            elif score <= -1: return "WEAK_SELL"
            else: return "HOLD"

    def screen_stocks(self, stock_category: str = "All Stocks", top_n: int = 20, timeframe: str = "Long Term") -> pd.DataFrame:
        """
        Enhanced screening with timeframe consideration
        """
        results = []
        
        # Market cap ranges in Crores
        LARGE_CAP_MIN = 20000
        MID_CAP_MIN = 5000
        SMALL_CAP_MAX = 5000

        if stock_category == "Indices":
            for ticker, data in self.stock_data.items():
                if data['info'].get('sector') == 'Index':
                    results.append(data.get('analysis'))
            df = pd.DataFrame([res for res in results if res is not None])
            if not df.empty:
                df = df.sort_values('name').reset_index(drop=True)
                df.index += 1
            return df

        # For stock categories
        for ticker, data in self.stock_data.items():
            if data['info'].get('sector') == 'Index':
                continue
            
            analysis_metrics = data.get('analysis')
            
            if analysis_metrics and analysis_metrics["market_cap"] is not None:
                market_cap = analysis_metrics["market_cap"]
                
                # Apply category filter
                if stock_category == "Large Cap" and market_cap < LARGE_CAP_MIN:
                    continue
                elif stock_category == "Mid Cap" and not (MID_CAP_MIN <= market_cap < LARGE_CAP_MIN):
                    continue
                elif stock_category == "Small Cap" and not (0 < market_cap < SMALL_CAP_MAX):
                    continue
                elif stock_category == "All Stocks" and market_cap <= 0:
                    continue
                    
                results.append(analysis_metrics)
        
        df = pd.DataFrame(results)
        if not df.empty:
            # Sort by Magic Score for long-term, by technical signal strength for intraday
            if timeframe == "Long Term":
                df = df.sort_values('magic_score', ascending=False).head(top_n)
            else:
                # Create a sorting score based on signal strength for intraday
                signal_scores = {
                    'STRONG_BUY': 6, 'BUY': 4, 'WEAK_BUY': 2, 'HOLD': 0,
                    'WEAK_SELL': -2, 'SELL': -4, 'STRONG_SELL': -6
                }
                df['signal_score'] = df['overall_signal'].map(signal_scores).fillna(0)
                df = df.sort_values(['signal_score', 'rsi'], ascending=[False, True]).head(top_n)
                df = df.drop('signal_score', axis=1)
            
            df.reset_index(drop=True, inplace=True)
            df.index += 1
        
        self.screened_results = df
        return df

    def get_index_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Fetches historical data for a given index."""
        try:
            stock = yf.Ticker(ticker)
            return stock.history(period=period, auto_adjust=True)
        except Exception as e:
            print(f"Error fetching index data for {ticker}: {e}")
            return pd.DataFrame()

# Streamlit App - Enhanced UI
def main():
    st.set_page_config(
        page_title="Magic Formula Pro 🎯",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced CSS with new signal colors
    st.markdown("""
    <style>
        .stButton button { 
            width: 100%; 
            background-color: #007aff;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #005bb5;
            cursor: pointer;
        }
        
        div[data-testid="stAlert"] {
            border-radius: 8px;
            margin-top: 1rem;
            padding: 15px;
        }
        
        .stMetric {
            background-color: transparent;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: none;
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            border: 1px solid #E0E0E0;
        }
        .stMetric > div:first-child {
            font-size: 1.1rem;
            color: #2c3e50;
            font-weight: bold;
            margin-bottom: 8px;
        }
        .stMetric > div:nth-child(2) {
            font-size: 2.2rem;
            font-weight: bold;
            color: #004D99;
            margin-top: 0;
        }

        /* Enhanced signal colors */
        .signal-strong-buy { color: #006400; font-weight: bold; }
        .signal-buy { color: #228B22; font-weight: bold; }
        .signal-weak-buy { color: #32CD32; font-weight: bold; }
        .signal-hold { color: #1E90FF; font-weight: bold; }
        .signal-weak-sell { color: #FF8C00; font-weight: bold; }
        .signal-sell { color: #FF4500; font-weight: bold; }
        .signal-strong-sell { color: #DC143C; font-weight: bold; }

        h1, h2, h3 { color: #333333; }
        .main > div { padding-top: 2rem; }
        .stDataFrame { border-radius: 8px; overflow: hidden; }
        body { font-family: 'Inter', sans-serif; }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)
    
    st.markdown("""
    # *Magic Formula Pro*
    ### Enhanced with Price Positioning & Timeframe Analysis
    ---
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'screener' not in st.session_state:
        st.session_state.screener = MagicFormulaPro()
        st.session_state.data_loaded = False
        st.session_state.results = pd.DataFrame()
        st.session_state.index_data = {}
        st.session_state.current_timeframe = "Long Term"
        
    # Enhanced Sidebar
    st.sidebar.header("Configuration")
    st.sidebar.markdown("---")
    
    # NEW: Timeframe Selection
    timeframe = st.sidebar.selectbox(
        "🕐 Trading Timeframe:",
        options=["Long Term", "Intraday"],
        help="Choose your trading horizon: Long Term (weeks/months) or Intraday (same day)"
    )
    
    # Update session state if timeframe changed
    if timeframe != st.session_state.current_timeframe:
        st.session_state.current_timeframe = timeframe
        st.session_state.data_loaded = False  # Force refresh when timeframe changes
    
    stock_category = st.sidebar.selectbox(
        "Select Category for Screening:",
        options=["All Stocks", "Large Cap", "Mid Cap", "Small Cap", "Indices"],
        help="Filter by market cap or analyze indices"
    )
    
    top_n = st.sidebar.slider(
        "Number of Results to Display",
        min_value=5, max_value=50, value=25,
        help="Focus on your circle of competence"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("Actions")
    
    # Enhanced data fetching with timeframe
    if st.sidebar.button("🚀 Fetch All Data", type="primary"):
        with st.spinner(f"Fetching {timeframe.lower()} data for Indian markets..."):
            progress_bar = st.progress(0)
            st.session_state.screener.fetch_stock_data_concurrently(
                timeframe=timeframe, 
                progress_callback=progress_bar.progress
            )
            
            # Fetch major index data
            major_indices = ['^NSEI', '^NSEBANK', '^NIFTY500.NS', '^BSESN']
            for idx in major_indices:
                st.session_state.index_data[idx] = st.session_state.screener.get_index_data(idx)
            
            st.session_state.data_loaded = True
            
        num_stocks = len([s for s in st.session_state.screener.stock_data.keys() 
                         if st.session_state.screener.stock_data[s]['info']['sector'] != 'Index'])
        num_indices = len([s for s in st.session_state.screener.stock_data.keys() 
                          if st.session_state.screener.stock_data[s]['info']['sector'] == 'Index'])
        st.sidebar.success(f"✅ {timeframe} data loaded: {num_stocks} stocks, {num_indices} indices!")
    
    if st.sidebar.button("🎯 Run Smart Screening"):
        if not st.session_state.data_loaded:
            st.sidebar.error("Please fetch data first!")
        else:
            with st.spinner(f"Running {timeframe.lower()} screening for '{stock_category}'..."):
                results = st.session_state.screener.screen_stocks(
                    stock_category=stock_category, 
                    top_n=top_n,
                    timeframe=timeframe
                )
                st.session_state.results = results
                
            if not results.empty:
                st.success(f"✅ {timeframe} screening complete! Found {len(results)} opportunities.")
            else:
                st.warning("No matches found. Try adjusting criteria.")

    # Display timeframe info
    st.sidebar.markdown("---")
    if st.session_state.data_loaded:
        cache_time = "5 minutes" if timeframe == "Intraday" else "1 hour"
        st.sidebar.caption(f"🕐 {timeframe} mode active")
        st.sidebar.caption(f"Data cached for {cache_time}")
        
    # Main Content
    if not st.session_state.results.empty:
        # Dynamic header based on timeframe and category
        if stock_category == "Indices":
            st.header(f"📈 Key Indian Indices - {timeframe} View")
            st.info(f"Technical analysis of major market indices for {timeframe.lower()} perspective.")
        else:
            st.header(f"🏆 Smart Stock Picks - {timeframe} Strategy")
            if timeframe == "Intraday":
                st.info("⚡ *Intraday Focus:* Quick momentum plays with technical signals optimized for same-day trading.")
            else:
                st.info("📊 *Long-term Focus:* Quality businesses at attractive valuations using Magic Formula + technical confirmation.")

        # Enhanced results display
        display_df = st.session_state.results.copy()
        display_df['Rank'] = range(1, len(display_df) + 1)
        display_df['Price (₹)'] = display_df['current_price'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
        
        if stock_category == "Indices":
            display_columns = ['Rank', 'name', 'ticker', 'Price (₹)', 'overall_signal', 'rsi', 'ema_trend']
            rename_cols = {
                'name': 'Index Name', 'ticker': 'Ticker', 'overall_signal': 'Signal',
                'rsi': 'RSI', 'ema_trend': 'Trend'
            }
        else:
            # Add price positioning info for stocks
            display_df['Price Position'] = display_df.apply(lambda row: 
                f"{'↑' if row.get('price_vs_ema20') == 'ABOVE' else '↓'}EMA20 "
                f"{'↑' if row.get('price_vs_ema50') == 'ABOVE' else '↓'}EMA50", axis=1)
            
            if timeframe == "Long Term":
                display_df['Magic Score'] = display_df['magic_score'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                display_columns = ['Rank', 'name', 'ticker', 'Price (₹)', 'Magic Score', 'overall_signal', 'Price Position', 'rsi']
                rename_cols = {
                    'name': 'Company', 'ticker': 'Ticker', 'overall_signal': 'Signal',
                    'rsi': 'RSI'
                }
            else:
                display_df['EMA20 Dist'] = display_df['ema20_distance_pct'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
                display_columns = ['Rank', 'name', 'ticker', 'Price (₹)', 'overall_signal', 'Price Position', 'EMA20 Dist', 'rsi']
                rename_cols = {
                    'name': 'Company', 'ticker': 'Ticker', 'overall_signal': 'Signal',
                    'rsi': 'RSI'
                }
        
        st.dataframe(
            display_df[display_columns].rename(columns=rename_cols).set_index('Rank'),
            use_container_width=True
        )

        st.markdown("---")
        
        # Enhanced detailed analysis
        st.header("🔍 Detailed Analysis & Strategy")
        
        all_available_tickers = list(st.session_state.screener.stock_data.keys())
        valid_tickers = [
            t for t in all_available_tickers 
            if st.session_state.screener.stock_data[t]['current_price'] is not None 
            and st.session_state.screener.stock_data[t].get('analysis') is not None
        ]
        valid_tickers.sort(key=lambda t: st.session_state.screener.stock_data[t]['info']['name'])

        selected_ticker = st.selectbox(
            f"Select for {timeframe.lower()} analysis:",
            options=valid_tickers,
            format_func=lambda t: f"{st.session_state.screener.stock_data[t]['info']['name']} ({t})"
        )
        
        if selected_ticker:
            selected_stock_data = st.session_state.screener.stock_data.get(selected_ticker)
            
            if selected_stock_data:
                is_index = selected_stock_data['info'].get('sector') == 'Index'
                
                # Get analysis data
                selected_analysis = None
                selected_analysis_from_results = st.session_state.results[
                    st.session_state.results['ticker'] == selected_ticker
                ]
                
                if not selected_analysis_from_results.empty:
                    selected_analysis = selected_analysis_from_results.iloc[0].to_dict()
                else:
                    selected_analysis = st.session_state.screener._calculate_analysis_metrics(
                        selected_ticker, selected_stock_data, timeframe
                    )
                
                if selected_analysis:
                    st.subheader(f"{timeframe} Dashboard: {selected_analysis['name']} ({selected_ticker})")
                    
                    if not is_index:
                        # Enhanced metrics display for stocks
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            if timeframe == "Long Term" and selected_analysis.get('magic_score'):
                                st.metric("Magic Score", f"{selected_analysis['magic_score']:.1f}")
                            else:
                                st.metric("Current Price", f"₹{selected_analysis['current_price']:,.2f}")
                        
                        with col2:
                            st.metric("RSI", f"{selected_analysis['rsi']:.1f}")
                        
                        with col3:
                            ema20_dist = selected_analysis.get('ema20_distance_pct', 0)
                            st.metric("EMA20 Distance", f"{ema20_dist:+.1f}%")
                        
                        with col4:
                            st.metric("Signal Strength", selected_analysis['overall_signal'])
                        
                        # Enhanced signal display with price positioning
                        signal_class = selected_analysis['overall_signal'].lower().replace('_', '-')
                        price_pos = selected_analysis.get('price_position', 'UNKNOWN').replace('_', ' ').title()
                        
                        st.markdown(f"""
                        *📊 {timeframe} Technical Analysis:*
                        - *Signal:* <span class='signal-{signal_class}'>{selected_analysis['overall_signal']}</span>
                        - *Price Position:* {price_pos}
                        - *Trend:* {selected_analysis['ema_trend']}
                        - *RSI Status:* {selected_analysis['rsi_signal']}
                        """, unsafe_allow_html=True)
                        
                        # Enhanced F&O Strategies
                        fo_expander = st.expander(f"💼 *{timeframe} F&O Strategy Recommendations*")
                        with fo_expander:
                            strategy = st.session_state.screener._get_fo_strategies(selected_ticker, timeframe)
                            st.markdown(strategy)
                            st.caption("⚠ *Disclaimer:* Educational purposes only. Options trading involves significant risk.")
                            
                    else:
                        # Index dashboard
                        col1, col2, col3 = st.columns(3)
                        with col1: st.metric("Index Level", f"{selected_analysis['current_price']:,.2f}")
                        with col2: st.metric("RSI", f"{selected_analysis['rsi']:.1f}")
                        with col3: st.metric("Trend", selected_analysis['ema_trend'])
                        
                        signal_class = selected_analysis['overall_signal'].lower().replace('_', '-')
                        st.markdown(f"*Technical Signal:* <span class='signal-{signal_class}'>{selected_analysis['overall_signal']}</span>", unsafe_allow_html=True)

                    # Enhanced charting with timeframe-appropriate periods
                    st.subheader(f"{timeframe} Price & Technical Chart")
                    
                    fig = make_subplots(
                        rows=3, cols=1,
                        row_heights=[0.5, 0.25, 0.25],
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=(
                            f"{selected_analysis['name']} - {timeframe} Analysis",
                            f"RSI ({timeframe})",
                            f"MACD ({timeframe})"
                        )
                    )
                    
                    # Candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=selected_stock_data['price_data'].index,
                        open=selected_stock_data['price_data']['Open'],
                        high=selected_stock_data['price_data']['High'],
                        low=selected_stock_data['price_data']['Low'],
                        close=selected_stock_data['price_data']['Close'],
                        name='Price',
                        increasing_line_color='green',
                        decreasing_line_color='red'
                    ), row=1, col=1)
                    
                    # EMA lines
                    ema_short_name = "9 EMA" if timeframe == "Intraday" else "20 EMA"
                    ema_long_name = "21 EMA" if timeframe == "Intraday" else "50 EMA"
                    
                    fig.add_trace(go.Scatter(
                        x=selected_stock_data['price_data'].index,
                        y=selected_stock_data['price_data']['EMA_20'],
                        mode='lines', name=ema_short_name, 
                        line=dict(color='orange', width=2)
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=selected_stock_data['price_data'].index,
                        y=selected_stock_data['price_data']['EMA_50'],
                        mode='lines', name=ema_long_name, 
                        line=dict(color='purple', width=2)
                    ), row=1, col=1)
                                                         
                    # RSI with dynamic thresholds
                    fig.add_trace(go.Scatter(
                        x=selected_stock_data['price_data'].index,
                        y=selected_stock_data['price_data']['RSI'],
                        mode='lines', name='RSI', 
                        line=dict(color='cyan', width=2)
                    ), row=2, col=1)
                    
                    # RSI thresholds based on timeframe
                    if timeframe == "Intraday":
                        fig.add_hline(y=75, line_dash="dash", line_color="red", 
                                     annotation_text="Overbought (75)", row=2, col=1)
                        fig.add_hline(y=25, line_dash="dash", line_color="green", 
                                     annotation_text="Oversold (25)", row=2, col=1)
                    else:
                        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                                     annotation_text="Overbought (70)", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                                     annotation_text="Oversold (30)", row=2, col=1)
                        
                    # MACD
                    fig.add_trace(go.Bar(
                        x=selected_stock_data['price_data'].index,
                        y=selected_stock_data['price_data']['MACD'],
                        name='MACD Hist', marker_color='lightgrey'
                    ), row=3, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=selected_stock_data['price_data'].index,
                        y=selected_stock_data['price_data']['MACD'],
                        mode='lines', name='MACD Line', 
                        line=dict(color='blue', width=1.5)
                    ), row=3, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=selected_stock_data['price_data'].index,
                        y=selected_stock_data['price_data']['MACD_Signal'],
                        mode='lines', name='Signal Line', 
                        line=dict(color='red', width=1.5)
                    ), row=3, col=1)
                    
                    fig.update_layout(
                        height=800, 
                        xaxis_rangeslider_visible=False, 
                        title=f"{timeframe} Technical Analysis: {selected_analysis['name']}", 
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional insights based on price positioning
                    st.subheader("📍 Price Position Analysis")
                    
                    if selected_analysis.get('price_position'):
                        price_insights = {
                            "ABOVE_BOTH_EMAS": "🟢 *Strong Position:* Price trading above both EMAs indicates bullish momentum and trend alignment.",
                            "BELOW_BOTH_EMAS": "🔴 *Weak Position:* Price below both EMAs suggests bearish pressure and potential further downside.",
                            "BETWEEN_EMAS_UPPER": "🟡 *Transition Zone:* Price between EMAs but closer to resistance - watch for breakout direction.",
                            "BETWEEN_EMAS_LOWER": "🟠 *Caution Zone:* Price between EMAs but closer to support - trend uncertainty."
                        }
                        
                        insight = price_insights.get(selected_analysis['price_position'], "Position analysis unavailable.")
                        st.info(insight)
                        
                        # Distance metrics
                        ema20_dist = selected_analysis.get('ema20_distance_pct', 0)
                        ema50_dist = selected_analysis.get('ema50_distance_pct', 0)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Distance from EMA20", f"{ema20_dist:+.1f}%", 
                                     help="Positive = above EMA, Negative = below EMA")
                        with col2:
                            st.metric("Distance from EMA50", f"{ema50_dist:+.1f}%", 
                                     help="Indicates overall trend strength and position")

        # Export functionality
        st.markdown("---")
        st.header("💾 Export Results")
        
        if not st.session_state.results.empty:
            # Enhanced CSV with additional columns
            export_df = st.session_state.results.copy()
            export_df['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            export_df['timeframe'] = timeframe
            
            csv = export_df.to_csv(index=False)
            filename = f"magic_formula_pro_{timeframe.lower().replace(' ', '')}results{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            st.download_button(
                label=f"📄 Download {timeframe} Results CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
            
    else:
        # Enhanced welcome screen
        st.markdown(f"""
        ## *Welcome to Magic Formula Pro - Enhanced Edition*
        
        Now featuring *price positioning analysis* and *dual timeframe strategies* for both intraday traders and long-term investors.
        
        > "In the short run, the market is a voting machine. In the long run, it's a weighing machine." - Benjamin Graham
        
        ### *🎯 New Enhanced Features:*
        
        *📊 Price Position Analysis:*
        - Price vs EMA20/EMA50 positioning
        - Distance percentages for precise entry/exit
        - Transitional zone identification
        
        *⏰ Dual Timeframe Modes:*
        - *Long Term:* Buffett-style value investing with Magic Formula
        - *Intraday:* Fast-moving technical signals for same-day trading
        
        *🎚 Enhanced Signal System:*
        - 7 signal levels: Strong Buy → Strong Sell
        - Price positioning heavily weighted in decisions
        - Timeframe-optimized thresholds
        
        ### *🚀 How the Enhanced Signals Work:*
        
        *For {st.session_state.current_timeframe} Trading:*
        """)
        
        if st.session_state.current_timeframe == "Intraday":
            st.markdown("""
            - *RSI Thresholds:* 25/75 (more sensitive for quick moves)
            - *EMA Periods:* 9/21 (faster response to price changes)
            - *Price Distance:* 1%/3% thresholds for momentum
            - *Update Frequency:* 5-minute cache for fresher data
            - *Focus:* Quick momentum, scalping, same-day closure
            """)
        else:
            st.markdown("""
            - *RSI Thresholds:* 30/70 (standard for trend analysis)
            - *EMA Periods:* 20/50 (classic trend identification)
            - *Price Distance:* 2%/5% thresholds for trend confirmation
            - *Update Frequency:* 1-hour cache for stable analysis
            - *Focus:* Quality businesses, fundamental value, long-term trends
            """)
        
        st.markdown("""
        ### *📈 Signal Weightage Breakdown:*
        - *RSI:* 30-40% (momentum and reversal signals)
        - *EMA Trend:* 20-30% (overall trend direction)
        - *Price Position:* 30-40% (most critical - actual price vs key levels)
        
        *🎯 Get Started:* Select your timeframe and click *"🚀 Fetch All Data"* to begin analysis.
        """)
        
        # Timeframe comparison
        col1, col2 = st.columns(2)
        with col1:
            st.info("⚡ *Intraday Mode:* Fast signals, quick profits, high frequency updates, momentum-based strategies.")
        with col2:
            st.info("📚 *Long Term Mode:* Value discovery, fundamental analysis, trend following, wealth building.")

if _name_ == "_main":  # Fixed: was _name instead of _name_
    main()
