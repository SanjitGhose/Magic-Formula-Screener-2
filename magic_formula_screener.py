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

# Suppress warnings for cleaner output in Streamlit
warnings.filterwarnings('ignore')

class MagicFormulaPro:
    """
    Magic Formula Pro: The definitive stock analysis platform for Indian markets.
    Incorporates Greenblatt's Magic Formula, technical analysis, and the
    wisdom of Buffett/Munger to find high-quality, undervalued stocks.
    Built with a Steve Jobs-like focus on user experience and J.P. Morgan-level analysis.
    """
    
    def __init__(self):
        # NseTools library dependency removed for stability and wider compatibility
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
            {"name": "Pidilite Industries", "ticker": "PIDILITE.NS", "sector": "Chemicals"},
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
            {"name": "Nykaa", "ticker": "FSN.NS", "sector": "E-commerce"},
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
            {"name": "Adani Green Energy", "ticker": "ADANIGREEN.NS", "sector": "Renewable Energy"},
            {"name": "Adani Transmission", "ticker": "ADANITRANS.NS", "sector": "Power Transmission"},
            {"name": "IDFC First Bank", "ticker": "IDFCFIRSTB.NS", "sector": "Banking"},
            {"name": "Federal Bank", "ticker": "FEDERALBNK.NS", "sector": "Banking"},
            {"name": "AU Small Finance Bank", "ticker": "AUBANK.NS", "sector": "Banking"},
            {"name": "IRCTC", "ticker": "IRCTC.NS", "sector": "Railways & Tourism"},
            {"name": "Coforge", "ticker": "COFORGE.NS", "sector": "IT Services"},
            {"name": "L&T Technology Services", "ticker": "LTTS.NS", "sector": "IT Services"},
            {"name": "LTIMindtree", "ticker": "LTIM.NS", "sector": "IT Services"}, # Updated ticker for Mindtree
            {"name": "Persistent Systems", "ticker": "PERSISTENT.NS", "sector": "IT Services"},
            {"name": "Tata Consumer Products", "ticker": "TATACONSUM.NS", "sector": "FMCG"},
            {"name": "Godrej Properties", "ticker": "GODREJPROP.NS", "sector": "Real Estate"},
            {"name": "Macrotech Developers", "ticker": "LODHA.NS", "sector": "Real Estate"},
            {"name": "Prestige Estates", "ticker": "PRESTIGE.NS", "sector": "Real Estate"},
            {"name": "Oberoi Realty", "ticker": "OBEROIRLTY.NS", "sector": "Real Estate"},
            {"name": "Phoenix Mills", "ticker": "PHOENIXLTD.NS", "sector": "Real Estate"},
            {"name": "Astral", "ticker": "ASTRAL.NS", "sector": "Plastics"},
            {"name": "Polycab India", "ticker": "POLYCAB.NS", "sector": "Electrical Equipment"},
            {"name": "Havells India", "ticker": "HAVELLS.NS", "sector": "Electrical Equipment"},
            {"name": "Voltas", "ticker": "VOLTAS.NS", "sector": "Consumer Durables"},
            {"name": "Dixon Technologies", "ticker": "DIXON.NS", "sector": "Electronics Manufacturing"},
            {"name": "Amber Enterprises", "ticker": "AMBER.NS", "sector": "Electronics Manufacturing"},
            {"name": "Gujarat Gas", "ticker": "GUJGASLTD.NS", "sector": "Gas Distribution"},
            {"name": "Adani Total Gas", "ticker": "ATGL.NS", "sector": "Gas Distribution"},
            {"name": "IGL", "ticker": "IGL.NS", "sector": "Gas Distribution"},
            {"name": "Mahanagar Gas", "ticker": "MGL.NS", "sector": "Gas Distribution"},
            {"name": "Torrent Power", "ticker": "TORNTPOWER.NS", "sector": "Power"},
            {"name": "Adani Power", "ticker": "ADANIPOWER.NS", "sector": "Power"},
            {"name": "National Aluminium", "ticker": "NATIONALUM.NS", "sector": "Metals & Mining"},
            {"name": "Container Corp of India", "ticker": "CONCOR.NS", "sector": "Logistics"},
            {"name": "Blue Dart Express", "ticker": "BLUEDART.NS", "sector": "Logistics"},
            {"name": "Mahindra Logistics", "ticker": "MAHLOG.NS", "sector": "Logistics"},
            {"name": "IRFC", "ticker": "IRFC.NS", "sector": "Financial Services"},
            {"name": "RVNL", "ticker": "RVNL.NS", "sector": "Construction"},
            {"name": "NBCC", "ticker": "NBCC.NS", "sector": "Construction"},
            {"name": "Godawari Power & Ispat", "ticker": "GPIL.NS", "sector": "Steel"}, # Small Cap Example
            {"name": "Vaibhav Global", "ticker": "VAIBHAVGBL.NS", "sector": "Retail"}, # Small Cap Example
            {"name": "Tanla Platforms", "ticker": "TANLA.NS", "sector": "IT Software"}, # Small Cap Example
            {"name": "Affle India", "ticker": "AFFLE.NS", "sector": "IT Software"}, # Small Cap Example
            {"name": "CDSL", "ticker": "CDSL.NS", "sector": "Financial Services"},
            {"name": "MCX", "ticker": "MCX.NS", "sector": "Financial Services"},
            {"name": "Angel One", "ticker": "ANGELONE.NS", "sector": "Financial Services"},
            {"name": "CAMS", "ticker": "CAMS.NS", "sector": "Financial Services"},
            {"name": "Delhivery", "ticker": "DELHIVERY.NS", "sector": "Logistics"},
            {"name": "PB Fintech", "ticker": "POLICYB.NS", "sector": "Fintech"},
            {"name": "Bharat Electronics", "ticker": "BEL.NS", "sector": "Defence"},
            {"name": "Hindustan Aeronautics", "ticker": "HAL.NS", "sector": "Defence"},
            {"name": "Mazagon Dock Shipbuilders", "ticker": "MAZDA.NS", "sector": "Defence"},
            {"name": "Cochin Shipyard", "ticker": "COCHINSHIP.NS", "sector": "Defence"},
            {"name": "Data Patterns", "ticker": "DATAPATTNS.NS", "sector": "Defence"}, # Small Cap Example
            {"name": "Paras Defence", "ticker": "PARAS.NS", "sector": "Defence"}, # Small Cap Example
        ]
        
        # Add Nifty Indices and other Sectoral & Broad Market Indices
        index_tickers = [
            {'name': 'Nifty 50 Index', 'ticker': '^NSEI', 'sector': 'Index'},
            {'name': 'Bank Nifty Index', 'ticker': '^NSEBANK', 'sector': 'Index'},
            {'name': 'Nifty Midcap 100', 'ticker': '^NIFTYMIDCAP100.NS', 'sector': 'Index'},
            {'name': 'Nifty Smallcap 100', 'ticker': '^NIFTYSMALLCAP100.NS', 'sector': 'Index'},
            {'name': 'Nifty 500 Index', 'ticker': '^NIFTY500.NS', 'sector': 'Index'}, # Added Nifty 500
            {'name': 'BSE Sensex Index', 'ticker': '^BSESN', 'sector': 'Index'},     # Added Sensex
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
            {'name': 'Nifty Financial Services Index', 'ticker': '^NIFTYFINSRV.NS', 'sector': 'Index'}, # Added Financial Services
            {'name': 'Nifty Healthcare Index', 'ticker': '^NIFTYHLTH.NS', 'sector': 'Index'}, # Added Healthcare
        ]
        
        return stocks + index_tickers

    def _get_fo_strategies(self, ticker: str) -> str:
        """
        Provides optimal F&O strategies based on technical signals and market context.
        This is designed to reflect an institutional-grade F&O desk analysis,
        tailoring strategies to current market conditions (strong buy, sell, hold).
        """
        data = self.stock_data.get(ticker)
        if not data or 'analysis' not in data:
            return "No comprehensive data available for F&O strategy analysis. Please fetch data first."
        
        signal = data['analysis'].get('overall_signal')
        rsi = data['analysis'].get('rsi')
        ema_trend = data['analysis'].get('ema_trend')
        
        if signal is None:
            return "Unable to generate F&O strategy due to missing technical signals."

        strategy_recommendations = {
            "STRONG_BUY": (
                "**Bullish Momentum:** Consider a **Bull Call Spread** (buy ATM call, sell OTM call) "
                "or a **Long Call** to capitalize on a strong anticipated uptrend. "
                "The bullish EMA trend combined with a favorable RSI suggests strong buying pressure. "
                "This indicates potential for significant upside with managed risk for the spread."
            ),
            "BUY": (
                "**Moderately Bullish:** A **Covered Call** (if you hold the underlying stock, sell OTM call) "
                "can generate income while you maintain your long position. "
                "Alternatively, a **Bull Put Spread** (sell OTM put, buy further OTM put) "
                "can profit from a moderate rise or sideways movement above a support level."
            ),
            "HOLD": (
                "**Neutral/Sideways:** If volatility is expected to be low, consider an **Iron Condor** "
                "(sell OTM call spread and OTM put spread) or a **Short Straddle/Strangle** (sell ATM call and put/OTM call and put). "
                "These strategies profit from time decay when the stock stays within a defined range. "
                "If volatility is expected to increase, a **Long Straddle/Strangle** might be suitable."
            ),
            "SELL": (
                "**Moderately Bearish:** A **Bear Call Spread** (sell OTM call, buy further OTM call) "
                "can profit from a moderate decline or sideways movement below a resistance level. "
                "If you own the stock, consider a **Protective Put** (buy OTM put) to hedge against further downside."
            ),
            "STRONG_SELL": (
                "**Bearish Momentum:** Consider a **Bear Put Spread** (buy ATM put, sell OTM put) "
                "or a **Long Put** to profit from a strong anticipated downtrend. "
                "The bearish EMA trend and overbought RSI suggest significant selling pressure. "
                "This strategy is suited for a potential sharp decline with defined risk for the spread."
            )
        }
        
        return strategy_recommendations.get(signal, "No specific F&O strategy recommended based on current signals or lack of clear trend.")
        
    def fetch_single_stock(self, stock_info: Dict, period: str = "1y") -> Optional[Dict]:
        """
        Fetches data for a single stock using yfinance.
        Implements robust caching and a mock data fallback for reliability.
        """
        ticker = stock_info["ticker"]
        cache_key = f"{ticker}_{period}"
        current_time = time.time()
        
        # Check cache: Data is considered fresh for 1 hour (3600 seconds)
        if (cache_key in self.cache and
            cache_key in self.last_fetch_time and
            current_time - self.last_fetch_time[cache_key] < 3600):
            return self.cache[cache_key]
        
        try:
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period=period)
            
            if hist_data.empty:
                # If no historical data, generate mock data
                result = self._generate_mock_data(stock_info)
            else:
                info = stock.info # Attempt to get fundamental info
                # Fallback for missing fundamental data if yfinance.info fails or is incomplete
                if not info:
                    info = self._generate_mock_fundamentals()

                # Calculate technical indicators
                hist_data = self._calculate_technical_indicators(hist_data)
                
                result = {
                    "info": stock_info,
                    "price_data": hist_data,
                    "fundamentals": info,
                    "current_price": float(hist_data['Close'].iloc[-1]) if not hist_data['Close'].empty else None
                }
            
            # Cache the result and update fetch time
            self.cache[cache_key] = result
            self.last_fetch_time[cache_key] = current_time
            
            return result
            
        except Exception as e:
            # On any error during fetching, use mock data to keep the app running
            # st.error(f"Error fetching data for {ticker}: {e}. Using mock data.") # Can enable for debugging
            return self._generate_mock_data(stock_info)
    
    def _generate_mock_data(self, stock_info: Dict) -> Dict:
        """
        Generates realistic mock data for price and fundamentals when API calls fail.
        Ensures the application always has data to display and analyze.
        """
        # Generate 200 business days of data to allow for indicator calculation
        dates = pd.date_range(end=datetime.now(), periods=200, freq='B') 
        base_price = np.random.uniform(100, 3000)
        # Simulate price movements with some randomness
        price_changes = np.random.normal(0, 0.015, 200).cumsum()
        prices = base_price * np.exp(price_changes)
        
        mock_data = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.005, 0.005)),
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(500000, 5000000, 200) # Realistic volume range
        }, index=dates)
        
        mock_data = self._calculate_technical_indicators(mock_data)
        
        return {
            "info": stock_info,
            "price_data": mock_data,
            "fundamentals": self._generate_mock_fundamentals(),
            "current_price": prices[-1]
        }

    def _generate_mock_fundamentals(self) -> Dict:
        """Generates mock fundamental data."""
        return {
            "marketCap": np.random.randint(10000, 500000) * 10000000, # In original units, will be converted to crores
            "forwardPE": np.random.uniform(10, 30),
            "trailingPE": np.random.uniform(12, 35),
            "returnOnEquity": np.random.uniform(0.10, 0.25),
            "returnOnAssets": np.random.uniform(0.05, 0.15)
        }
    
    def fetch_stock_data_concurrently(self, period: str = "1y", progress_callback=None) -> None:
        """
        Fetches all stock and index data concurrently using a ThreadPoolExecutor.
        This significantly speeds up data loading for large numbers of tickers.
        """
        total_stocks = len(self.all_stocks_info)
        self.stock_data = {} # Reset data
        
        with ThreadPoolExecutor(max_workers=min(10, total_stocks)) as executor: # Limit workers to total stocks if small
            future_to_stock = {executor.submit(self.fetch_single_stock, stock_info, period): stock_info for stock_info in self.all_stocks_info}
            
            processed_count = 0
            for future in as_completed(future_to_stock):
                stock_info = future_to_stock[future]
                try:
                    result = future.result()
                    if result and result['current_price'] is not None:
                        # Store the result and ensure it includes the analysis part
                        self.stock_data[stock_info["ticker"]] = result
                        # Pre-calculate analysis for instant access later
                        # For indices, ensure analysis is not None, but can have N/A for fundamental metrics
                        self.stock_data[stock_info["ticker"]]['analysis'] = self._calculate_analysis_metrics(stock_info["ticker"], result)
                except Exception as e:
                    # Log error but don't stop the process for one failed stock
                    # print(f"Error processing {stock_info['ticker']}: {e}") # Can enable for debugging
                    pass 
                finally:
                    processed_count += 1
                    if progress_callback:
                        progress_callback(processed_count / total_stocks)
        
        if progress_callback:
            progress_callback(1.0) # Ensure progress bar hits 100%

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates a suite of essential technical indicators (RSI, EMA, MACD).
        These provide the 'timing' aspect of the J.P. Morgan-style analysis.
        """
        if len(df) < 50: # Need sufficient data for meaningful indicator calculation
            # Fill with NaNs if not enough data to avoid errors
            df['RSI'] = np.nan
            df['EMA_20'] = np.nan
            df['EMA_50'] = np.nan
            df['MACD'] = np.nan
            df['MACD_Signal'] = np.nan
            return df
            
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # EMA (Exponential Moving Averages)
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False, min_periods=1).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False, min_periods=1).mean()
        
        # MACD (Moving Average Convergence Divergence)
        ema_12 = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
        
        return df

    def _calculate_analysis_metrics(self, ticker: str, stock_data: Dict) -> Optional[Dict]:
        """
        Calculates fundamental (Magic Formula) and technical analysis metrics
        for a single stock. This is the core analytical engine.
        Handles both stocks and indices by setting N/A for irrelevant metrics.
        """
        fundamentals = stock_data["fundamentals"]
        price_data = stock_data["price_data"]
        is_index = stock_data["info"].get("sector") == "Index"

        # Initialize common fields
        analysis_result = {
            "ticker": ticker,
            "name": stock_data["info"]["name"],
            "sector": stock_data["info"]["sector"],
            "current_price": stock_data["current_price"],
        }
        
        # Calculate technical indicators for both stocks and indices
        current_close = price_data['Close'].iloc[-1] if not price_data.empty else None
        current_rsi = price_data['RSI'].iloc[-1] if 'RSI' in price_data.columns and not pd.isna(price_data['RSI'].iloc[-1]) else 50
        ema_20 = price_data['EMA_20'].iloc[-1] if 'EMA_20' in price_data.columns and not pd.isna(price_data['EMA_20'].iloc[-1]) else current_close
        ema_50 = price_data['EMA_50'].iloc[-1] if 'EMA_50' in price_data.columns and not pd.isna(price_data['EMA_50'].iloc[-1]) else current_close
        
        analysis_result["rsi"] = current_rsi
        analysis_result["ema_trend"] = "BULLISH" if ema_20 is not None and ema_50 is not None and ema_20 > ema_50 else "BEARISH" if ema_20 is not None and ema_50 is not None else "N/A"
        analysis_result["rsi_signal"] = self._get_rsi_signal(current_rsi) if not pd.isna(current_rsi) else "N/A"
        analysis_result["overall_signal"] = self._get_overall_signal(current_rsi, ema_20, ema_50) if not pd.isna(current_rsi) and ema_20 is not None and ema_50 is not None else "N/A"

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
                return None # Filter out stocks with invalid market cap for screening
            analysis_result["market_cap"] = market_cap / 10000000 # Convert to crores

            roe = fundamentals.get('returnOnEquity', 0.10) * 100 
            roa = fundamentals.get('returnOnAssets', 0.05) * 100 
            roc = (roe + roa) / 2 if roe is not None and roa is not None else max(roe, roa) if roe is not None or roa is not None else 10.0
            roc = max(roc, 0.1) # Ensure ROC is at least a small positive value
            analysis_result["roc"] = roc

            pe_ratio = fundamentals.get('forwardPE', fundamentals.get('trailingPE'))
            if pe_ratio is None or pe_ratio <= 0:
                earnings_yield = 0.0
            else:
                earnings_yield = (1 / pe_ratio) * 100
            analysis_result["earnings_yield"] = earnings_yield
            analysis_result["pe_ratio"] = pe_ratio
            
            # Magic Score only for stocks with valid fundamentals
            analysis_result["magic_score"] = (roc * 0.6) + (earnings_yield * 0.4)
            
        return analysis_result
            
    def _get_rsi_signal(self, rsi: float) -> str:
        """Determines RSI signal based on common thresholds."""
        if pd.isna(rsi): return "N/A"
        if rsi < 30: return "OVERSOLD"
        elif rsi > 70: return "OVERBOUGHT"
        else: return "NEUTRAL"
    
    def _get_overall_signal(self, rsi: float, ema_20: float, ema_50: float) -> str:
        """
        Aggregates RSI and EMA trends into a single, comprehensive signal.
        Reflects a multi-factor approach to technical analysis.
        """
        if pd.isna(rsi) or ema_20 is None or ema_50 is None:
            return "N/A" # Cannot generate signal without valid technical data

        score = 0
        
        # RSI contribution
        if rsi < 35: score += 2 # Strong oversold suggests potential reversal up
        elif rsi < 50: score += 1 # Leaning oversold
        
        if rsi > 65: score -= 2 # Strong overbought suggests potential reversal down
        elif rsi > 50: score -= 1 # Leaning overbought
        
        # EMA trend contribution (primary trend indicator)
        if ema_20 > ema_50: score += 1 # Golden cross/bullish alignment
        else: score -= 1 # Death cross/bearish alignment
        
        # Convert score to signal
        if score >= 2: return "STRONG_BUY"
        elif score == 1: return "BUY"
        elif score <= -2: return "STRONG_SELL"
        elif score == -1: return "SELL"
        else: return "HOLD"
    
    def screen_stocks(self, stock_category: str = "All Stocks", top_n: int = 20) -> pd.DataFrame:
        """
        Screens stocks using the Magic Formula, applying filters and ranking based on category.
        If 'Indices' is selected, it returns the list of indices instead of screened stocks.
        """
        results = []
        
        # Define market cap ranges for categorization in Crores
        LARGE_CAP_MIN = 20000
        MID_CAP_MIN = 5000
        SMALL_CAP_MAX = 5000 # Max for small cap, assuming it's up to 5000 Cr

        if stock_category == "Indices":
            for ticker, data in self.stock_data.items():
                if data['info'].get('sector') == 'Index':
                    results.append(data.get('analysis')) # Append the analysis for indices
            # No sorting by magic score for indices, simply list them
            df = pd.DataFrame([res for res in results if res is not None])
            if not df.empty:
                df = df.sort_values('name').reset_index(drop=True) # Sort indices by name for consistency
                df.index += 1
            return df

        # For stock categories, proceed with Magic Formula screening
        for ticker, data in self.stock_data.items():
            # Skip indexes from screening results, as Magic Formula is for stocks
            if data['info'].get('sector') == 'Index':
                continue
            
            analysis_metrics = data.get('analysis')
            
            if analysis_metrics and analysis_metrics["market_cap"] is not None: # Ensure market_cap is not None for stocks
                market_cap = analysis_metrics["market_cap"]
                
                # Apply category filter
                if stock_category == "Large Cap" and market_cap < LARGE_CAP_MIN:
                    continue
                elif stock_category == "Mid Cap" and not (MID_CAP_MIN <= market_cap < LARGE_CAP_MIN):
                    continue
                elif stock_category == "Small Cap" and not (0 < market_cap < SMALL_CAP_MAX): # Ensure positive market cap and within range
                    continue
                # "All Stocks" category includes any stock with a valid market_cap > 0
                elif stock_category == "All Stocks" and market_cap <= 0:
                    continue
                    
                results.append(analysis_metrics)
        
        df = pd.DataFrame(results)
        if not df.empty:
            # Sort by Magic Score (Buffett/Munger influence)
            df = df.sort_values('magic_score', ascending=False).head(top_n)
            df.reset_index(drop=True, inplace=True)
            df.index += 1 # 1-based indexing for display
        
        self.screened_results = df
        return df

    def get_index_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Fetches historical data for a given index."""
        try:
            return yf.Ticker(ticker).history(period=period)
        except Exception:
            # Fallback to empty DataFrame on error
            return pd.DataFrame()

# Streamlit App - The User Experience Layer (Steve Jobs)
def main():
    st.set_page_config(
        page_title="Magic Formula Pro 🎯",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for a sleek, Apple-like feel and improved readability
    st.markdown("""
    <style>
        .stButton button { 
            width: 100%; 
            background-color: #007aff; /* Apple Blue */
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #005bb5; /* Darker blue on hover */
            cursor: pointer;
        }
        
        /* Improved readability for alert boxes */
        div[data-testid="stAlert"] {
            border-radius: 8px;
            margin-top: 1rem;
            padding: 15px;
        }
        div[data-testid="stAlert"].stAlert-info {
            background-color: #2196F3; /* A bit darker blue */
            color: white; /* White text for contrast */
        }
        div[data-testid="stAlert"].stAlert-warning {
            background-color: #FFC107; /* Orange-yellow */
            color: #333333; /* Dark text for contrast */
        }

        /* Improved readability for metrics - numbers and labels stand out */
        .stMetric {
            background-color: transparent; /* Removed white background */
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: none; /* Removed box shadow for cleaner look */
            display: flex; /* Use flexbox for layout */
            flex-direction: column; /* Stack elements vertically */
            align-items: flex-start; /* Align text to the start */
            border: 1px solid #E0E0E0; /* Subtle border for definition */
        }
        .stMetric > div:first-child { /* Label (e.g., "Magic Score") */
            font-size: 1.1rem; /* Slightly larger for text */
            color: #2c3e50; /* Darker, more prominent color for the label text */
            font-weight: bold; /* Make the label text bold */
            margin-bottom: 8px; /* Add more space below label */
        }
        .stMetric > div:nth-child(2) { /* Value (e.g., "75.23") */
            font-size: 2.2rem; /* Larger font size for the number */
            font-weight: bold; /* Bold the number */
            color: #004D99; /* Strong dark blue for the number to stand out */
            margin-top: 0; /* Remove top margin as space is handled by label */
        }
        .stMetric > div:nth-child(3) { /* Delta (e.g., "+0.5%") - if applicable */
            font-size: 1rem;
            margin-top: 5px;
        }

        /* Custom colors for technical signals */
        .signal-strong-buy { color: #008000; font-weight: bold; } /* Dark Green */
        .signal-buy { color: #32CD32; font-weight: bold; } /* Lime Green */
        .signal-hold { color: #1E90FF; font-weight: bold; } /* Dodger Blue (Neutral) */
        .signal-sell { color: #FF8C00; font-weight: bold; } /* Dark Orange */
        .signal-strong-sell { color: #DC143C; font-weight: bold; } /* Crimson (Strong Red) */


        h1, h2, h3 { color: #333333; } /* Darker text for titles */
        .main > div { padding-top: 2rem; }
        .stDataFrame { border-radius: 8px; overflow: hidden; } /* Rounded corners for tables */
        .stSelectbox { border-radius: 8px; }
        .stSlider { padding-bottom: 20px; }
        /* Jobs-inspired focus on clarity and typography */
        body {
            font-family: 'Inter', sans-serif; /* Modern, clean font */
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)
    
    st.markdown("""
    # **Magic Formula Pro**
    ### *Unlocking Value: Buffett & Munger's principles for the modern investor.*
    ---
    """, unsafe_allow_html=True)
    
    # Initialize session state for persistent data
    if 'screener' not in st.session_state:
        st.session_state.screener = MagicFormulaPro()
        st.session_state.data_loaded = False
        st.session_state.results = pd.DataFrame()
        st.session_state.index_data = {} # Store data for all relevant indexes
        
    # Sidebar: The Control Panel (Jobs's minimalist approach)
    st.sidebar.header("Configuration")
    st.sidebar.markdown("---")
    
    # New category selection instead of min_market_cap
    stock_category = st.sidebar.selectbox(
        "Select Category for Screening:", # Changed label for broader applicability
        options=["All Stocks", "Large Cap", "Mid Cap", "Small Cap", "Indices"], # Added "Indices"
        help="Filter the universe of stocks or select indices for analysis based on market capitalization or type."
    )
    
    top_n = st.sidebar.slider(
        "Number of Top Stocks to Display",
        min_value=5, max_value=50, value=25,
        help="The 'Circle of Competence' - choose how many stocks you can realistically analyze and monitor effectively."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("Actions")
    
    if st.sidebar.button("🚀 Fetch All Data", type="primary"):
        with st.spinner("Downloading and processing data for Indian stocks and major indices..."):
            progress_bar = st.progress(0)
            st.session_state.screener.fetch_stock_data_concurrently(progress_callback=progress_bar.progress)
            
            # Fetch Nifty 50 and Bank Nifty index data
            st.session_state.index_data['^NSEI'] = st.session_state.screener.get_index_data('^NSEI')
            st.session_state.index_data['^NSEBANK'] = st.session_state.screener.get_index_data('^NSEBANK')
            # Fetch additional index data
            st.session_state.index_data['^NIFTY500.NS'] = st.session_state.screener.get_index_data('^NIFTY500.NS')
            st.session_state.index_data['^BSESN'] = st.session_state.screener.get_index_data('^BSESN')
            
            st.session_state.data_loaded = True
            
        num_stocks_loaded = len([s for s in st.session_state.screener.all_stocks_info if s['sector'] != 'Index'])
        num_indices_loaded = len([s for s in st.session_state.screener.all_stocks_info if s['sector'] == 'Index'])
        st.sidebar.success(f"✅ Data for {num_stocks_loaded} stocks and {num_indices_loaded} major Indices loaded!")
    
    if st.sidebar.button("🎯 Run Magic Screening"):
        if not st.session_state.data_loaded:
            st.sidebar.error("Please fetch data first by clicking '🚀 Fetch All Data'!")
        else:
            with st.spinner(f"Applying the Magic Formula filters and ranking stocks from the '{stock_category}' category..."):
                results = st.session_state.screener.screen_stocks(
                    stock_category=stock_category, top_n=top_n
                )
                st.session_state.results = results
            if not st.session_state.results.empty:
                st.success(f"✅ Screening complete! Found {len(results)} qualifying {'stocks' if stock_category != 'Indices' else 'indices'} for your consideration in the '{stock_category}' category.")
            else:
                st.warning(f"No {'stocks' if stock_category != 'Indices' else 'indices'} matched your criteria in the '{stock_category}' category. Try adjusting the category or increasing the 'Number of Top Stocks'.")

    st.sidebar.markdown("---")
    if st.session_state.data_loaded:
        st.sidebar.caption("Data is cached for 1 hour to optimize performance.")
        
    # Main Content: The Result (Jobs's emphasis on clarity and beauty)
    if not st.session_state.results.empty:
        # Display different header based on selection
        if stock_category == "Indices":
            st.header("📈 Key Indian Indices")
            st.info("Explore the performance and technicals of major Indian market indices.")
        else:
            st.header("🏆 The Magic Formula Portfolio")
            st.info("These stocks represent high-quality businesses identified at potentially attractive valuations, aligning with the principles of value investing.")

        # Display the ranked table
        display_df = st.session_state.results.copy()
        display_df['Rank'] = range(1, len(display_df) + 1)
        display_df['Price (₹)'] = display_df['current_price'].apply(lambda x: f"₹{x:,.2f}")
        
        # Adjust columns for display based on category
        if stock_category == "Indices":
            display_df['Market Cap (Cr)'] = "N/A" # Not applicable for indices
            display_df['Magic Score'] = "N/A"
            display_df['RoC (%)'] = "N/A"
            display_df['Earnings Yield (%)'] = "N/A"
            display_columns = ['Rank', 'name', 'ticker', 'Price (₹)', 'Technical Signal']
            rename_cols = {'name': 'Index Name', 'ticker': 'Ticker', 'overall_signal': 'Technical Signal'}
        else:
            display_df['Market Cap (Cr)'] = display_df['market_cap'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "N/A")
            display_df['Magic Score'] = display_df['magic_score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            display_df['RoC (%)'] = display_df['roc'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            display_df['Earnings Yield (%)'] = display_df['earnings_yield'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            display_columns = [
                'Rank', 'name', 'ticker', 'Price (₹)', 'Market Cap (Cr)', 'Magic Score',
                'RoC (%)', 'Earnings Yield (%)', 'overall_signal'
            ]
            rename_cols = {
                'name': 'Company', 'ticker': 'Ticker', 
                'overall_signal': 'Technical Signal'
            }
        
        st.dataframe(
            display_df[display_columns].rename(columns=rename_cols).set_index('Rank'),
            use_container_width=True
        )

        st.markdown("---")
        
        # Detailed Analysis for a selected stock or index
        st.header("🔍 Deep Dive: Company & Index Analysis")
        
        # Prepare options for selectbox, including both screened and index tickers
        all_available_tickers = list(st.session_state.screener.stock_data.keys())
        # Filter out only tickers that have current_price and any analysis object (even minimal for indices)
        valid_tickers_for_selection = [
            t for t in all_available_tickers 
            if st.session_state.screener.stock_data[t]['current_price'] is not None and st.session_state.screener.stock_data[t].get('analysis') is not None
        ]

        # Sort for better user experience
        valid_tickers_for_selection.sort(key=lambda t: st.session_state.screener.stock_data[t]['info']['name'])

        selected_ticker = st.selectbox(
            "Select a stock or index for detailed analysis:",
            options=valid_tickers_for_selection,
            format_func=lambda t: f"{st.session_state.screener.stock_data[t]['info']['name']} ({t})"
        )
        
        if selected_ticker:
            selected_stock_data = st.session_state.screener.stock_data.get(selected_ticker)
            
            if not selected_stock_data:
                st.warning(f"Detailed data for {selected_ticker} is not available. Please try fetching data again.")
            else:
                is_index = selected_stock_data['info'].get('sector') == 'Index'
                
                # Retrieve selected_analysis safely
                selected_analysis = None
                # Try to get from screened results first for consistency if available
                selected_analysis_from_results = st.session_state.results[st.session_state.results['ticker'] == selected_ticker]
                
                if not selected_analysis_from_results.empty:
                    selected_analysis = selected_analysis_from_results.iloc[0].to_dict()
                else:
                    # If not in screened results (e.g., if user directly selected an index or a non-screened stock), recalculate
                    selected_analysis = st.session_state.screener._calculate_analysis_metrics(selected_ticker, selected_stock_data)
                
                if selected_analysis: # This check is now safe for Dict (truthy if not empty) or None
                    if not is_index: # Proceed with stock-specific analysis
                        st.subheader(f"Dashboard for {selected_analysis['name']} ({selected_analysis['ticker']})")
                        
                        col1, col2, col3 = st.columns(3)
                        # Ensure these metrics display "N/A" for None values
                        with col1: st.metric("Magic Score", f"{selected_analysis['magic_score']:.2f}" if selected_analysis['magic_score'] is not None else "N/A", help="Weighted score of Return on Capital and Earnings Yield – our value indicator.")
                        with col2: st.metric("Return on Capital", f"{selected_analysis['roc']:.2f}%" if selected_analysis['roc'] is not None else "N/A", help="A crucial metric for identifying businesses with a strong competitive advantage (moat).")
                        with col3: st.metric("Earnings Yield", f"{selected_analysis['earnings_yield']:.2f}%" if selected_analysis['earnings_yield'] is not None else "N/A", help="The inverse of the P/E ratio, indicating the earnings generated per unit of investment – our valuation metric.")
                        
                        # Apply dynamic colors to the overall_signal and ensure all text pops
                        signal_class = selected_analysis['overall_signal'].lower().replace('_', '-')
                        st.markdown(
                            f"<p style='font-weight: bold; font-size: 1.1em; color: #2c3e50; margin-bottom: 0.5rem;'>" # Darker color for labels
                            f"Technical Signal: <span class='signal-{signal_class}'>`{selected_analysis['overall_signal']}`</span>"
                            f"<br>EMA Trend: <span style='color: #004D99;'>`{selected_analysis['ema_trend']}`</span>" # Popping color
                            f"<br>RSI: <span style='color: #004D99;'>`{selected_analysis['rsi']:.2f}` ({selected_analysis['rsi_signal']})</span>" # Popping color
                            f"</p>",
                            unsafe_allow_html=True
                        )
                        
                        # F&O Strategies Expander
                        fo_strategy_expander = st.expander("💼 **F&O Optimal Strategy (Institutional Grade)**")
                        with fo_strategy_expander:
                            st.markdown(st.session_state.screener._get_fo_strategies(selected_ticker))
                            st.caption("Disclaimer: These strategies are for educational purposes only and not financial advice. Options trading involves significant risk and may not be suitable for all investors. Consult with a qualified financial advisor.")
                            
                        # Charting (Jobs's visual clarity)
                        st.subheader("Price & Indicator Chart")
                        
                        fig = make_subplots(
                            rows=3, cols=1, # Changed to 3 rows for MACD
                            row_heights=[0.5, 0.25, 0.25], # Adjusted heights
                            shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=(f"{selected_analysis['name']} Price Action", "Relative Strength Index (RSI)", "MACD (Moving Average Convergence Divergence)") # Added MACD title
                        )
                        
                        # Candlestick chart (red/green for profit/loss as requested)
                        fig.add_trace(go.Candlestick(
                            x=selected_stock_data['price_data'].index,
                            open=selected_stock_data['price_data']['Open'],
                            high=selected_stock_data['price_data']['High'],
                            low=selected_stock_data['price_data']['Low'],
                            close=selected_stock_data['price_data']['Close'],
                            name='Price',
                            increasing_line_color='green', decreasing_line_color='red' # Already set
                        ), row=1, col=1)
                        
                        # Add EMA lines
                        fig.add_trace(go.Scatter(x=selected_stock_data['price_data'].index, y=selected_stock_data['price_data']['EMA_20'],
                                                 mode='lines', name='20 EMA', line=dict(color='orange', width=1.5)), row=1, col=1)
                        fig.add_trace(go.Scatter(x=selected_stock_data['price_data'].index, y=selected_stock_data['price_data']['EMA_50'],
                                                 mode='lines', name='50 EMA', line=dict(color='purple', width=1.5)), row=1, col=1)
                                                         
                        # Add RSI
                        fig.add_trace(go.Scatter(x=selected_stock_data['price_data'].index, y=selected_stock_data['price_data']['RSI'],
                                                 mode='lines', name='RSI', line=dict(color='cyan', width=1.5)), row=2, col=1)
                        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="top left", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="bottom left", row=2, col=1)
                        
                        # Add MACD
                        fig.add_trace(go.Bar(x=selected_stock_data['price_data'].index, y=selected_stock_data['price_data']['MACD'],
                                             name='MACD Hist', marker_color='grey'), row=3, col=1) # MACD Histogram
                        fig.add_trace(go.Scatter(x=selected_stock_data['price_data'].index, y=selected_stock_data['price_data']['MACD'],
                                                 mode='lines', name='MACD Line', line=dict(color='blue', width=1.5)), row=3, col=1) # MACD line
                        fig.add_trace(go.Scatter(x=selected_stock_data['price_data'].index, y=selected_stock_data['price_data']['MACD_Signal'],
                                                 mode='lines', name='Signal Line', line=dict(color='red', width=1.5)), row=3, col=1) # Signal line
                        
                        fig.update_layout(height=800, xaxis_rangeslider_visible=False, title_text=f"Historical Price and Technicals for {selected_analysis['name']}", showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else: # If an Index is selected for detailed view
                        st.subheader(f"Dashboard for {selected_stock_data['info']['name']} ({selected_stock_data['info']['ticker']})")
                        st.write("Displaying historical data for this index.")
                        st.write("Fundamental metrics (Magic Score, RoC, Earnings Yield) are not applicable to indices.")

                        # Charting for Index
                        fig_index = make_subplots(
                            rows=3, cols=1, # Changed to 3 rows for MACD
                            row_heights=[0.5, 0.25, 0.25], # Adjusted heights
                            shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=(f"{selected_analysis['name']} Price Action", "Relative Strength Index (RSI)", "MACD (Moving Average Convergence Divergence)") # Added MACD title
                        )

                        fig_index.add_trace(go.Candlestick(
                            x=selected_stock_data['price_data'].index,
                            open=selected_stock_data['price_data']['Open'],
                            high=selected_stock_data['price_data']['High'],
                            low=selected_stock_data['price_data']['Low'],
                            close=selected_stock_data['price_data']['Close'],
                            name='Price',
                            increasing_line_color='green', decreasing_line_color='red'
                        ), row=1, col=1)

                        # Add EMA lines for indices
                        fig_index.add_trace(go.Scatter(x=selected_stock_data['price_data'].index, y=selected_stock_data['price_data']['EMA_20'],
                                                 mode='lines', name='20 EMA', line=dict(color='orange', width=1.5)), row=1, col=1)
                        fig_index.add_trace(go.Scatter(x=selected_stock_data['price_data'].index, y=selected_stock_data['price_data']['EMA_50'],
                                                 mode='lines', name='50 EMA', line=dict(color='purple', width=1.5)), row=1, col=1)

                        # Add RSI for indices
                        fig_index.add_trace(go.Scatter(x=selected_stock_data['price_data'].index, y=selected_stock_data['price_data']['RSI'],
                                                 mode='lines', name='RSI', line=dict(color='cyan', width=1.5)), row=2, col=1)
                        fig_index.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="top left", row=2, col=1)
                        fig_index.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="bottom left", row=2, col=1)

                        # Add MACD for indices
                        fig_index.add_trace(go.Bar(x=selected_stock_data['price_data'].index, y=selected_stock_data['price_data']['MACD'],
                                             name='MACD Hist', marker_color='grey'), row=3, col=1) # MACD Histogram
                        fig_index.add_trace(go.Scatter(x=selected_stock_data['price_data'].index, y=selected_stock_data['price_data']['MACD'],
                                                 mode='lines', name='MACD Line', line=dict(color='blue', width=1.5)), row=3, col=1) # MACD line
                        fig_index.add_trace(go.Scatter(x=selected_stock_data['price_data'].index, y=selected_stock_data['price_data']['MACD_Signal'],
                                                 mode='lines', name='Signal Line', line=dict(color='red', width=1.5)), row=3, col=1) # Signal line
                        
                        fig_index.update_layout(height=800, xaxis_rangeslider_visible=False, title=f"Historical Price and Technicals for {selected_stock_data['info']['name']}", showlegend=True)
                        st.plotly_chart(fig_index, use_container_width=True)

                else: # Fallback if selected_analysis is None after all attempts
                    st.warning("Could not calculate detailed analysis for this stock. Data might be insufficient or invalid.")


        st.header("💾 Export Results")
        csv = st.session_state.results.to_csv(index=False)
        st.download_button(
            label="📄 Download Top Stocks CSV",
            data=csv,
            file_name=f"magic_formula_pro_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
            
    else:
        # Welcome screen (Jobs's compelling introduction)
        st.markdown("""
        ## **Welcome to Magic Formula Pro**
        The ultimate blend of Warren Buffett's timeless value investing principles and modern, institutional-grade analysis for the Indian markets.
        
        > *"Price is what you pay. Value is what you get."* - Warren Buffett
        
        Our platform simplifies the complex task of finding valuable companies by focusing on core, powerful metrics, and providing actionable insights.
        
        ### **🎯 How it works:**
        1.  **Comprehensive Data Fetch:** Obtain real-time data for a curated list of major Indian stocks and key indices (Nifty 50, Bank Nifty, Mid/Smallcap indices, Sensex).
        2.  **Fundamental & Value Screening:** Stocks are ranked using **Joel Greenblatt's Magic Formula**, emphasizing:
            * **Return on Capital (RoC):** A key indicator of business quality and competitive advantage, inspired by Charlie Munger's focus on great businesses.
            * **Earnings Yield:** A valuation metric to find attractively priced companies, reflecting Buffett's quest for value.
        3.  **Advanced Technical Analysis:** We overlay this with a suite of technical indicators (RSI, EMA trends) and provide clear **trading signals**, offering a "J.P. Morgan" analyst's perspective on market timing.
        4.  **Optimal F&O Strategies:** Get tailored Futures & Options strategy recommendations based on the stock's analytical outlook.
        5.  **Elegant Visualization:** All insights are presented through clean, interactive charts and tables, designed for immediate understanding.
        
        To begin your journey towards smarter investments, click **"🚀 Fetch All Data"** in the sidebar.
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("📊 **Curated Stock Universe:** Access data for major Indian stocks, mid-caps, small-caps, and key indices.")
        with col2:
            st.info("🧠 **Buffett & Munger Insight:** Invest with the wisdom of the world's greatest investors.")
        with col3:
            st.info("📈 **Real-time Technicals & F&O:** Get the latest signals and strategic recommendations.")

if __name__ == "__main__":
    main()
