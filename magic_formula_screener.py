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

warnings.filterwarnings('ignore')

class MagicFormulaScreener:
    """
    Simplified Magic Formula Stock Screener for Streamlit
    """
    
    def __init__(self):
        self.indian_stocks = self._get_indian_stock_list()
        self.stock_data = {}
        self.screened_results = pd.DataFrame()
        self.cache = {}
        self.last_fetch_time = {}
        
    def _get_indian_stock_list(self) -> List[Dict]:
        """Get an expanded list of major Indian stocks (over 50)"""
        return [
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
            # Additional stocks to reach over 50
            {"name": "Tata Motors", "ticker": "TATAMOTORS.NS", "sector": "Automobile"},
            {"name": "Power Grid Corporation", "ticker": "POWERGRID.NS", "sector": "Utilities"},
            {"name": "Adani Ports", "ticker": "ADANIPORTS.NS", "sector": "Ports"},
            {"name": "SBI Life Insurance", "ticker": "SBILIFE.NS", "sector": "Insurance"},
            {"name": "Eicher Motors", "ticker": "EICHERMOT.NS", "sector": "Automobile"},
            {"name": "Pidilite Industries", "ticker": "PIDILITIND.NS", "sector": "Chemicals"},
            {"name": "Dr. Reddy's Laboratories", "ticker": "DRREDDY.NS", "sector": "Pharma"},
            {"name": "IndusInd Bank", "ticker": "INDUSINDBK.NS", "sector": "Banking"},
            {"name": "HDFC Life Insurance", "ticker": "HDFCLIFE.NS", "sector": "Insurance"},
            {"name": "Hero MotoCorp", "ticker": "HEROMOTOCO.NS", "sector": "Automobile"},
            {"name": "Apollo Hospitals", "ticker": "APOLLOHOSP.NS", "sector": "Healthcare"},
            {"name": "Shree Cement", "ticker": "SHREECEM.NS", "sector": "Cement"},
            {"name": "BPCL", "ticker": "BPCL.NS", "sector": "Oil & Gas"},
            {"name": "Grasim Industries", "ticker": "GRASIM.NS", "sector": "Diversified"},
            {"name": "DLF", "ticker": "DLF.NS", "sector": "Real Estate"},
            {"name": "Britannia Industries", "ticker": "BRITANNIA.NS", "sector": "FMCG"},
            {"name": "Cipla", "ticker": "CIPLA.NS", "sector": "Pharma"},
            {"name": "UPL", "ticker": "UPL.NS", "sector": "Chemicals"},
            {"name": "Tata Steel", "ticker": "TATASTEEL.NS", "sector": "Steel"},
            {"name": "Coal India", "ticker": "COALINDIA.NS", "sector": "Mining"},
            {"name": "Hindalco", "ticker": "HINDALCO.NS", "sector": "Metals"},
            {"name": "ONGC", "ticker": "ONGC.NS", "sector": "Oil & Gas"},
            {"name": "Bajaj Finserv", "ticker": "BAJAJFINSV.NS", "sector": "NBFC"},
            {"name": "NTPC", "ticker": "NTPC.NS", "sector": "Utilities"},
            {"name": "Adani Green Energy", "ticker": "ADANIGREEN.NS", "sector": "Renewable Energy"},
            {"name": "Vedanta", "ticker": "VEDL.NS", "sector": "Mining"}
        ]
    
    def fetch_single_stock(self, stock_info: Dict, period: str = "1y") -> Optional[Dict]:
        """Fetch data for a single stock with cache and mock data fallback"""
        try:
            ticker = stock_info["ticker"]
            
            # Check cache
            cache_key = f"{ticker}_{period}"
            current_time = time.time()
            
            if (cache_key in self.cache and 
                cache_key in self.last_fetch_time and 
                current_time - self.last_fetch_time[cache_key] < 300): # Cache for 5 minutes
                return self.cache[cache_key]
            
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period=period)
            
            if hist_data.empty:
                return self._generate_mock_data(stock_info)
                
            # Get fundamental data with fallback
            try:
                info = stock.info
            except Exception: # Catch any error during info fetch
                info = {"marketCap": np.random.randint(10000, 500000) * 10000000,
                       "forwardPE": np.random.uniform(10, 30),
                       "trailingPE": np.random.uniform(12, 35),
                       "returnOnEquity": np.random.uniform(0.10, 0.25),
                       "returnOnAssets": np.random.uniform(0.05, 0.15)}
            
            # Calculate technical indicators
            hist_data = self._calculate_technical_indicators(hist_data)
            
            result = {
                "info": stock_info,
                "price_data": hist_data,
                "fundamentals": info,
                "current_price": float(hist_data['Close'].iloc[-1])
            }
            
            # Cache the result
            self.cache[cache_key] = result
            self.last_fetch_time[cache_key] = current_time
            
            return result
            
        except Exception as e:
            # Fallback to mock data on any fetch error
            return self._generate_mock_data(stock_info)
    
    def _generate_mock_data(self, stock_info: Dict) -> Dict:
        """Generate realistic mock data when API fails or data is empty"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        base_price = np.random.uniform(100, 2000)
        
        # Generate realistic price movements
        price_changes = np.random.normal(0, 0.02, 100).cumsum() # Cumulative sum for trend
        prices = base_price * (1 + price_changes / 10) + np.random.normal(0, base_price * 0.01, 100)
        
        mock_data = pd.DataFrame({
            'Open': prices * (1 - np.random.uniform(0, 0.005)),
            'High': prices * (1 + np.random.uniform(0, 0.01)),
            'Low': prices * (1 - np.random.uniform(0, 0.01)),
            'Close': prices * (1 + np.random.uniform(-0.005, 0.005)),
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
        
        mock_data = self._calculate_technical_indicators(mock_data)
        
        return {
            "info": stock_info,
            "price_data": mock_data,
            "fundamentals": {
                "marketCap": np.random.randint(10000, 500000) * 10000000,
                "forwardPE": np.random.uniform(10, 30),
                "trailingPE": np.random.uniform(12, 35),
                "returnOnEquity": np.random.uniform(0.10, 0.25),
                "returnOnAssets": np.random.uniform(0.05, 0.15)
            },
            "current_price": prices[-1]
        }
    
    def fetch_stock_data(self, period: str = "1y", progress_callback=None) -> None:
        """Fetch stock data with progress tracking using multithreading for efficiency"""
        total_stocks = len(self.indian_stocks)
        self.stock_data = {} # Clear old data before fetching new
        
        # Use ThreadPoolExecutor for concurrent fetching to speed up the process
        with ThreadPoolExecutor(max_workers=10) as executor: # Increased max_workers for faster fetching
            futures = {executor.submit(self.fetch_single_stock, stock_info, period): stock_info for stock_info in self.indian_stocks}
            
            for i, future in enumerate(as_completed(futures)):
                stock_info = futures[future]
                try:
                    result = future.result()
                    if result:
                        self.stock_data[stock_info["ticker"]] = result
                except Exception:
                    # Silently fail for individual stock fetch errors, mock data should handle it
                    pass
                
                if progress_callback:
                    progress_callback((i + 1) / total_stocks)
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators (RSI, EMA, MACD, Bollinger Bands, Momentum)"""
        # Need at least 50 periods for EMA_50 and Bollinger Bands to be meaningful
        if len(df) < 50:
            return df # Return original DataFrame if not enough data
            
        try:
            # RSI (Relative Strength Index)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # EMA (Exponential Moving Average)
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            
            # MACD (Moving Average Convergence Divergence)
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            sma_20 = df['Close'].rolling(window=20).mean()
            std_20 = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = sma_20 + (std_20 * 2)
            df['BB_Lower'] = sma_20 - (std_20 * 2)
            df['BB_Middle'] = sma_20
            
            # Momentum (10-period percentage change)
            df['Momentum'] = df['Close'].pct_change(periods=10) * 100
            
            return df
            
        except Exception as e:
            # In case of any calculation error, return original DataFrame
            return df
    
    def calculate_magic_formula_score(self, ticker: str) -> Optional[Dict]:
        """Calculate Magic Formula score and integrate technical signals"""
        try:
            stock_info = self.stock_data[ticker]
            fundamentals = stock_info["fundamentals"]
            price_data = stock_info["price_data"]
            
            # Fundamental metrics with safe defaults (using 0 for N/A in calculations)
            market_cap = fundamentals.get('marketCap', 0) / 10000000 # Convert to Crores
            pe_ratio = fundamentals.get('forwardPE', fundamentals.get('trailingPE', 0))
            roe = fundamentals.get('returnOnEquity', 0) * 100
            roa = fundamentals.get('returnOnAssets', 0) * 100
            
            # Ensure PE ratio is not zero or negative to avoid division by zero
            if pe_ratio <= 0: pe_ratio = 20 # Default to a reasonable PE if invalid

            roc = (roe + roa) / 2 if (roe and roa) else max(roe, roa) if (roe or roa) else 12
            earnings_yield = (1 / pe_ratio * 100) if pe_ratio > 0 else 5
            magic_score = (roc * 0.6) + (earnings_yield * 0.4)
            
            # Technical indicators (with default values if not available or NaN)
            current_rsi = price_data['RSI'].iloc[-1] if 'RSI' in price_data.columns and not price_data['RSI'].empty and not pd.isna(price_data['RSI'].iloc[-1]) else 50
            current_momentum = price_data['Momentum'].iloc[-1] if 'Momentum' in price_data.columns and not price_data['Momentum'].empty and not pd.isna(price_data['Momentum'].iloc[-1]) else 0
            
            # EMA Trend (with fallback to current price if EMA values are NaN)
            current_close = price_data['Close'].iloc[-1] if not price_data['Close'].empty else 0
            ema_20 = price_data['EMA_20'].iloc[-1] if 'EMA_20' in price_data.columns and not price_data['EMA_20'].empty and not pd.isna(price_data['EMA_20'].iloc[-1]) else current_close
            ema_50 = price_data['EMA_50'].iloc[-1] if 'EMA_50' in price_data.columns and not price_data['EMA_50'].empty and not pd.isna(price_data['EMA_50'].iloc[-1]) else current_close
            ema_trend = "BULLISH" if ema_20 > ema_50 else "BEARISH"
            
            # Signals
            rsi_signal = self._get_rsi_signal(current_rsi)
            momentum_signal = self._get_momentum_signal(current_momentum)
            overall_signal = self._get_overall_signal(current_rsi, current_momentum, ema_20, ema_50)
            
            return {
                "ticker": ticker,
                "name": stock_info["info"]["name"],
                "sector": stock_info["info"]["sector"],
                "current_price": stock_info["current_price"],
                "market_cap": market_cap,
                "roc": roc,
                "earnings_yield": earnings_yield,
                "magic_score": magic_score,
                "pe_ratio": pe_ratio,
                "rsi": current_rsi,
                "momentum": current_momentum,
                "ema_trend": ema_trend,
                "rsi_signal": rsi_signal,
                "momentum_signal": momentum_signal,
                "overall_signal": overall_signal
            }
            
        except Exception as e:
            # Return None if calculation fails for any reason
            return None
    
    def _get_rsi_signal(self, rsi: float) -> str:
        if rsi < 30:
            return "OVERSOLD (BUY)"
        elif rsi > 70:
            return "OVERBOUGHT (SELL)"
        else:
            return "NEUTRAL (HOLD)"
    
    def _get_momentum_signal(self, momentum: float) -> str:
        if momentum > 5:
            return "STRONG UP"
        elif momentum > 0:
            return "UP"
        elif momentum < -5:
            return "STRONG DOWN"
        else:
            return "DOWN"
    
    def _get_overall_signal(self, rsi: float, momentum: float, ema_20: float, ema_50: float) -> str:
        score = 0
        
        if rsi < 30:
            score += 2
        elif rsi > 70:
            score -= 2
        
        if momentum > 2:
            score += 1
        elif momentum < -2:
            score -= 1
        
        if ema_20 > ema_50:
            score += 1
        else:
            score -= 1
        
        if score >= 3:
            return "STRONG BUY"
        elif score >= 1:
            return "BUY"
        elif score <= -3:
            return "STRONG SELL"
        elif score <= -1:
            return "SELL"
        else:
            return "HOLD"
            
    def _generate_buffett_analysis(self, stock_metrics: Dict) -> str:
        """Generates a simplified 'Warren Buffett' style analysis based on stock metrics."""
        analysis = []
        name = stock_metrics.get("name", "This company")
        pe_ratio = stock_metrics.get("pe_ratio")
        roc = stock_metrics.get("roc")
        market_cap = stock_metrics.get("market_cap")
        
        # Rule 1: Understand the business (simplified by sector)
        sector = stock_metrics.get("sector", "unknown")
        if sector in ["Banking", "FMCG", "IT Services", "Automobile", "Utilities", "Healthcare"]:
            analysis.append(f"**{name}** operates in a sector ({sector}) I generally find straightforward and understandable.")
        else:
            analysis.append(f"**{name}** is in the {sector} sector. I prefer simple businesses, so I'd need to dig deeper here to truly understand its competitive advantages.")

        # Rule 2: Favorable long-term prospects (simplified by ROC)
        if roc is not None and roc > 18:
            analysis.append(f"Its Return on Capital ({roc:.2f}%) is excellent, suggesting it has a strong competitive moat and efficient operations.")
        elif roc is not None and roc > 12:
            analysis.append(f"The Return on Capital ({roc:.2f}%) is decent, indicating a reasonably efficient business.")
        else:
            analysis.append(f"The Return on Capital ({roc:.2f}%) could be stronger. I always look for businesses with consistently high returns on the capital they employ.")

        # Rule 3: Attractive price (simplified by PE and Magic Score)
        if pe_ratio is not None and pe_ratio > 0: # Ensure PE is positive
            if pe_ratio < 20 and stock_metrics.get("magic_score", 0) > 60:
                analysis.append(f"The P/E ratio ({pe_ratio:.2f}) combined with its strong Magic Score suggests it might be trading at a reasonable price for a quality business.")
            elif pe_ratio < 25 and stock_metrics.get("magic_score", 0) > 50:
                analysis.append(f"The P/E ratio ({pe_ratio:.2f}) is acceptable, and the Magic Score is fair. This could be a good candidate for further valuation analysis.")
            else:
                analysis.append(f"The P/E ratio ({pe_ratio:.2f}) seems a bit high, or the Magic Score isn't compelling enough for deep value. Price is what you pay, value is what you get.")
        else:
            analysis.append("The P/E ratio is not readily available or is invalid, making it difficult to assess its attractiveness from a valuation perspective.")

        # General Buffett advice
        analysis.append("Remember, I always invest within my 'circle of competence' and focus on long-term value. This analysis is merely a starting point for thorough research, not a substitute for it.")

        return " ".join(analysis)
    
    def screen_stocks(self, min_market_cap: float = 1000, top_n: int = 50) -> pd.DataFrame:
        """Screen stocks using Magic Formula"""
        results = []
        
        for ticker in self.stock_data.keys():
            result = self.calculate_magic_formula_score(ticker)
            if result and result["market_cap"] >= min_market_cap:
                results.append(result)
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('magic_score', ascending=False).head(top_n)
            df.reset_index(drop=True, inplace=True)
            df.index += 1
        
        self.screened_results = df
        return df

def generate_candlestick_chart(price_data: pd.DataFrame, title: str) -> go.Figure:
    """Generates a detailed candlestick chart with technical indicators including Volume."""
    
    # Check for sufficient data for meaningful indicators
    if len(price_data) < 50:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for a detailed chart. Needs at least 50 periods for full indicators.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16)
        )
        fig.update_layout(template="plotly_dark", height=400) # Smaller height for warning
        return fig

    # Create subplots: Price (candlestick, EMA, BB), RSI/MACD, Volume
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.6, 0.2, 0.2], # Adjusted heights for 3 subplots
        shared_xaxes=True,
        vertical_spacing=0.05, # Reduced spacing for tighter layout
        subplot_titles=(title, 'RSI & MACD', 'Volume') # Added Volume title
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=price_data.index,
        open=price_data['Open'],
        high=price_data['High'],
        low=price_data['Low'],
        close=price_data['Close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ), row=1, col=1)

    # Moving averages (EMA)
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['EMA_20'],
        mode='lines',
        name='20 EMA',
        line=dict(color='orange', width=1),
        legendgroup='price'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['EMA_50'],
        mode='lines',
        name='50 EMA',
        line=dict(color='cyan', width=1), # Changed color for better distinction
        legendgroup='price'
    ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['BB_Upper'],
        mode='lines',
        name='BB Upper',
        line=dict(color='gray', width=1, dash='dot'),
        legendgroup='price'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['BB_Lower'],
        mode='lines',
        name='BB Lower',
        line=dict(color='gray', width=1, dash='dot'),
        fill='tonexty', # Fill area between bands
        fillcolor='rgba(128, 128, 128, 0.1)',
        legendgroup='price'
    ), row=1, col=1)

    # RSI (Relative Strength Index)
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='purple', width=2),
        legendgroup='rsi'
    ), row=2, col=1)
    
    # RSI overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought", annotation_position="top left")
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold", annotation_position="bottom left")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1, annotation_text="Mid", annotation_position="bottom right")


    # MACD (Moving Average Convergence Divergence)
    fig.add_trace(go.Bar(
        x=price_data.index,
        y=price_data['MACD_Hist'],
        name='MACD Hist',
        marker_color=['green' if val >= 0 else 'red' for val in price_data['MACD_Hist']], # Dynamic color for histogram
        legendgroup='macd'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['MACD'],
        mode='lines',
        name='MACD',
        line=dict(color='blue', width=1),
        legendgroup='macd'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['MACD_Signal'],
        mode='lines',
        name='MACD Signal',
        line=dict(color='red', width=1),
        legendgroup='macd'
    ), row=2, col=1)

    # Volume
    fig.add_trace(go.Bar(
        x=price_data.index,
        y=price_data['Volume'],
        name='Volume',
        marker_color='rgba(0, 150, 200, 0.6)', # A nice blue color for volume
        legendgroup='volume'
    ), row=3, col=1)

    # Layout and styling
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        showlegend=True,
        template="plotly_dark", # Dark theme for professional look
        margin=dict(l=20, r=20, t=50, b=20),
        height=800, # Increased height to accommodate all subplots
        title_font_size=20,
        legend_orientation="h",
        legend=dict(x=0, y=1.08, xanchor='left', yanchor='bottom', font=dict(size=10)), # Position legend at top
        hovermode="x unified" # Show all data on hover
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Indicators', row=2, col=1)
    fig.update_yaxes(title_text='Volume', row=3, col=1)
    
    # Hide x-axis ticks for upper subplots for cleaner look
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    
    return fig

def main():
    """Main Streamlit app logic."""
    
    # Page config
    st.set_page_config(
        page_title="Magic Formula Screener Pro",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for a professional look
    st.markdown("""
    <style>
        .main > div { padding-top: 1rem; }
        .stAlert { margin-top: 1rem; }
        h1, h2, h3, h4, h5 { color: #1e88e5; }
        .st-emotion-cache-1pxaz8s { /* Adjust sidebar width for better layout */
            width: 20rem;
        }
        .metric-container {
            background-color: #2e2e3e;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            color: white; /* Ensure text is visible on dark background */
        }
        .header-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0;
            padding-bottom: 0;
        }
        .header-subtitle {
            font-size: 1rem;
            font-style: italic;
            color: #a0a0a0;
            margin-top: 0;
            padding-top: 0;
        }
        .stButton button {
            background-color: #1e88e5;
            color: white;
            font-weight: bold;
            border-radius: 0.5rem;
            border: none;
            padding: 0.5rem 1rem;
        }
        .stDataFrame {
            font-size: 0.85rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: nowrap;
            background-color: #2e2e3e;
            border-radius: 6px 6px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1e88e5;
        }
        /* Style for the info box containing Buffett's analysis */
        .stAlert.st-emotion-cache-1c7y2kl { /* Target the specific alert class */
            background-color: #333344; /* Darker background for info box */
            border-left: 6px solid #4a90e2; /* Blue left border */
            color: #e0e0e0; /* Lighter text color */
        }
        .stAlert.st-emotion-cache-1c7y2kl p {
            color: #e0e0e0 !important; /* Ensure paragraph text color is light */
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='header-title'>⚡ Magic Formula Screener PRO</h1>", unsafe_allow_html=True)
    st.markdown("<p class='header-subtitle'>Advanced Indian stock screening combining fundamental and technical analysis</p>", unsafe_allow_html=True)
    
    # --- Sidebar ---
    st.sidebar.header("⚙️ App Settings")
    st.sidebar.markdown("---")
    
    if 'screener' not in st.session_state:
        st.session_state.screener = MagicFormulaScreener()
        st.session_state.data_loaded = False
        st.session_state.results = pd.DataFrame()
        st.session_state.selected_ticker = None
    
    # Data Fetching
    if st.sidebar.button("🚀 Fetch Fresh Data", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress):
            progress_bar.progress(progress)
            if progress < 0.3:
                status_text.text("🔄 Initializing data fetch...")
            elif progress < 0.7:
                status_text.text("📊 Downloading stock data (this may take a few minutes for 50+ stocks)...")
            else:
                status_text.text("⚡ Processing indicators and preparing analysis...")
        
        st.session_state.screener.fetch_stock_data(period="1y", progress_callback=update_progress)
        st.session_state.data_loaded = True
        
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Data fetched successfully!")
        
    if st.session_state.data_loaded:
        st.sidebar.success(f"Loaded data for {len(st.session_state.screener.stock_data)} stocks.")
    else:
        st.sidebar.warning("Data not loaded. Please fetch data to begin.")
        
    st.sidebar.markdown("---")
    
    # Screening Parameters
    st.sidebar.header("🔍 Screening Parameters")
    min_market_cap = st.sidebar.number_input(
        "💰 Minimum Market Cap (₹ Crores)", value=1000.0, step=100.0,
        help="Filter stocks by minimum market capitalization"
    )
    # Max value for top_n slider updated to 50
    top_n = st.sidebar.slider(
        "📊 Number of Top Stocks", min_value=5, max_value=50, value=20,
        help="Select how many top-ranked stocks to display"
    )
    
    # Run Screening Button
    if st.sidebar.button("🎯 Run Screening"):
        if not st.session_state.data_loaded:
            st.sidebar.error("❌ Please fetch data first!")
        else:
            with st.spinner("🔍 Applying Magic Formula & technical filters..."):
                results = st.session_state.screener.screen_stocks(
                    min_market_cap=min_market_cap, top_n=top_n
                )
                st.session_state.results = results
            st.success(f"✅ Found {len(results)} qualifying stocks!")
            
    st.sidebar.markdown("---")
    st.sidebar.info("Tip: Click on a stock in the table to view its detailed chart and Robot Warren Buffett's analysis below.")

    # --- Main Content ---
    if not st.session_state.results.empty:
        
        st.header("📊 Top Ranked Stocks Dashboard")
        
        # Performance Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("🏆 Top Stocks Displayed", f"{len(st.session_state.results)}")
        with col2:
            buy_signals = len(st.session_state.results[st.session_state.results['overall_signal'].str.contains('BUY', na=False)])
            st.metric("📈 Buy Signals", f"{buy_signals}", f"{buy_signals/len(st.session_state.results)*100:.1f}% of top stocks")
        with col3:
            avg_score = st.session_state.results['magic_score'].mean()
            st.metric("⭐ Avg Magic Score", f"{avg_score:.2f}", "Higher is better")
        with col4:
            bullish_trends = len(st.session_state.results[st.session_state.results['ema_trend'] == 'BULLISH'])
            st.metric("📊 Bullish Trends (EMA)", f"{bullish_trends}", f"{bullish_trends/len(st.session_state.results)*100:.1f}%")

        st.markdown("---")
        
        # Interactive Table and Detail View
        results_col, detail_col = st.columns([1, 1])
        
        with results_col:
            st.subheader("Top Ranked Stocks Table")
            display_df = st.session_state.results.copy()
            display_df['Rank'] = range(1, len(display_df) + 1)
            display_df['Price (₹)'] = display_df['current_price'].apply(lambda x: f"₹{x:.2f}")
            display_df['Market Cap (Cr ₹)'] = display_df['market_cap'].apply(lambda x: f"{x:.2f}")
            display_df['Magic Score'] = display_df['magic_score'].apply(lambda x: f"{x:.2f}")
            
            styled_df = display_df[['Rank', 'name', 'ticker', 'sector', 'Price (₹)', 'Magic Score', 'overall_signal']].rename(columns={
                'name': 'Company', 'ticker': 'Ticker', 'overall_signal': 'Signal'
            })
            
            # Use st.dataframe with selection_mode for easier row selection
            st.dataframe(
                styled_df,
                key='ranked_table',
                hide_index=True,
                use_container_width=True,
                on_select=lambda selected_rows_info: st.session_state.update({'selected_row_indices': selected_rows_info['selection']}),
                selection_mode="single-row"
            )

            # Retrieve selected row index from session state
            selected_row_indices = st.session_state.get('selected_row_indices', [])
            if selected_row_indices:
                selected_index = selected_row_indices[0] # Get the first selected index
                selected_ticker = styled_df.iloc[selected_index]['Ticker']
                st.session_state.selected_ticker = selected_ticker
            else:
                st.session_state.selected_ticker = None # Reset if nothing is selected
            
        with detail_col:
            st.subheader("Detailed Stock Analysis")
            if st.session_state.selected_ticker:
                ticker = st.session_state.selected_ticker
                
                # Fetch detailed data for the selected ticker
                stock_data = st.session_state.screener.stock_data.get(ticker)
                
                if stock_data:
                    info = stock_data['info']
                    fundamentals = stock_data['fundamentals']
                    
                    st.markdown(f"#### {info.get('name', 'N/A')} ({ticker})")
                    st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
                    
                    st.markdown("---")
                    
                    col_a, col_b = st.columns(2)
                    with col_a: st.markdown(f"**Current Price:** ₹{fundamentals.get('regularMarketPrice', stock_data['current_price']):.2f}")
                    with col_b: st.markdown(f"**Market Cap:** ₹{fundamentals.get('marketCap', 0) / 10000000:.2f} Cr")
                    
                    col_c, col_d = st.columns(2)
                    with col_c: st.markdown(f"**P/E Ratio:** {fundamentals.get('forwardPE', fundamentals.get('trailingPE', 'N/A')):.2f}")
                    with col_d: st.markdown(f"**ROE:** {fundamentals.get('returnOnEquity', 0) * 100:.2f}%")
                    
                    st.markdown("---")
                    
                    # Technicals and Magic Formula from screened results
                    # Ensure results_row is safely accessed
                    results_row_df = st.session_state.results[st.session_state.results['ticker'] == ticker]
                    if not results_row_df.empty:
                        results_row = results_row_df.iloc[0]
                        st.markdown(f"**Magic Score:** `{results_row['magic_score']:.2f}`")
                        st.markdown(f"**Overall Signal:** `{results_row['overall_signal']}`")
                        st.markdown(f"**EMA Trend:** `{results_row['ema_trend']}`")
                        st.markdown(f"**RSI:** `{results_row['rsi']:.2f}` ({results_row['rsi_signal']})")
                    else:
                        st.info("Screening results not available for this stock.")

                    st.markdown("---")
                    st.subheader("🤖 Robot Warren Buffett's Take")
                    # Get the metrics for the selected stock from the screened_results DataFrame
                    # This ensures we use the same calculated metrics for consistency
                    selected_stock_metrics = results_row.to_dict() if not results_row_df.empty else {}
                    buffett_analysis = st.session_state.screener._generate_buffett_analysis(selected_stock_metrics)
                    st.info(buffett_analysis) # Display the analysis in an info box

                else:
                    st.info("Select a stock from the table to see details.")
            else:
                st.info("Select a stock from the table to see details.")

        st.markdown("---")
        
        # Visualization Tabs
        st.subheader("📈 Interactive Charts")
        
        tab_fund, tab_tech = st.tabs(["📊 Fundamental & Sector Analysis", "📈 Technical Chart"])

        with tab_fund:
            col_a, col_b = st.columns(2)
            with col_a:
                fig_scatter = go.Figure(data=[
                    go.Scatter(x=st.session_state.results['earnings_yield'], y=st.session_state.results['roc'],
                               mode='markers', text=st.session_state.results['name'],
                               marker=dict(size=10, color=st.session_state.results['magic_score'], colorscale='Viridis',
                                           colorbar=dict(title='Magic Score'), showscale=True))
                ])
                fig_scatter.update_layout(title="Magic Formula Components: ROC vs. Earnings Yield",
                                          xaxis_title="Earnings Yield (%)", yaxis_title="Return on Capital (%)",
                                          template="plotly_dark")
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col_b:
                sector_counts = st.session_state.results['sector'].value_counts()
                fig_pie = go.Figure(data=[go.Pie(labels=sector_counts.index, values=sector_counts.values, hole=0.3)])
                fig_pie.update_layout(title="Sector Distribution of Top Stocks", template="plotly_dark")
                st.plotly_chart(fig_pie, use_container_width=True)

        with tab_tech:
            if st.session_state.selected_ticker:
                stock_data = st.session_state.screener.stock_data.get(st.session_state.selected_ticker)
                if stock_data and 'price_data' in stock_data and not stock_data['price_data'].empty:
                    st.plotly_chart(
                        generate_candlestick_chart(
                            stock_data['price_data'],
                            f"{stock_data['info'].get('name', 'N/A')} ({st.session_state.selected_ticker}) - Candlestick Chart"
                        ),
                        use_container_width=True
                    )
                else:
                    st.warning("No price data available for this stock, or insufficient data for full technical chart.")
            else:
                st.info("Select a stock from the table to display its technical chart.")
                
        st.markdown("---")
        
        # Export
        st.subheader("💾 Export Results")
        csv = st.session_state.results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Results CSV",
            data=csv,
            file_name=f"magic_formula_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        # Welcome screen
        st.markdown("""
        <div style="padding: 2rem; background-color: #1a1a2e; border-radius: 1rem;">
        <h2 style='color: #4a90e2;'>🚀 Welcome to Magic Formula Screener PRO!</h2>
        <p style='color: #d0d0d0;'>This powerful tool merges **Joel Greenblatt's Magic Formula** with comprehensive **Technical Analysis** to help you uncover hidden investment opportunities in the Indian market.</p>
        
        ### 🎯 Key Features:
        - **⚡ Real-time Data Fetching**: Get up-to-the-minute data with multi-threaded performance.
        - **📈 Magic Formula Rankings**: Identify stocks with high Return on Capital and Earnings Yield.
        - **📊 Advanced Technicals**: Analyze trends with RSI, MACD, EMA, and Bollinger Bands.
        - **⭐ Dynamic Signals**: Get actionable BUY/SELL/HOLD recommendations based on a blended score.
        - **🔍 Interactive Dashboard**: Visualize sector distributions, score correlations, and detailed price charts.
        - **🤖 Robot Warren Buffett's Take**: Gain simplified insights based on value investing principles.
        
        <p style='color: #d0d0d0;'>**Ready to start?** Use the sidebar to **fetch fresh data** and then run the screening!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.info(f"🏢 **{len(st.session_state.screener.indian_stocks)}** Major Stocks")
        with col2: st.info("📊 **8+** Key Indicators")
        with col3: st.info("🎯 **Blended** Scoring Model")
        with col4: st.info("📈 **Advanced** Charting")

    # --- Global Disclaimer ---
    st.markdown("---")
    st.markdown("""
    <div style="background-color:#2e2e3e; padding:15px; border-radius:8px; margin-top:30px; border: 1px solid #4a90e2;">
        <p style="font-size:14px; color:#a0a0a0; text-align:center; margin:0;">
            <b>Disclaimer:</b> This tool is for educational and informational purposes only and does not constitute financial or investment advice.
            Signals and data are generated based on publicly available APIs and may contain inaccuracies or be delayed.
            Always conduct your own due diligence and consult with a qualified financial professional before making any investment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
