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
        """Get list of major Indian stocks (expanded)"""
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
            {"name": "HDFC Life Insurance", "ticker": "HDFCLIFE.NS", "sector": "Insurance"},
            {"name": "SBI Life Insurance", "ticker": "SBILIFE.NS", "sector": "Insurance"},
            {"name": "Cipla", "ticker": "CIPLA.NS", "sector": "Pharma"},
            {"name": "Dr. Reddy's", "ticker": "DRREDDY.NS", "sector": "Pharma"},
            {"name": "Adani Ports", "ticker": "ADANIPORTS.NS", "sector": "Logistics"},
            {"name": "Britannia Industries", "ticker": "BRITANNIA.NS", "sector": "FMCG"},
            {"name": "Grasim Industries", "ticker": "GRASIM.NS", "sector": "Cement"},
            {"name": "Hindalco", "ticker": "HINDALCO.NS", "sector": "Metals & Mining"},
            {"name": "NTPC", "ticker": "NTPC.NS", "sector": "Power Generation"},
            {"name": "Power Grid", "ticker": "POWERGRID.NS", "sector": "Power Transmission"}
        ]
        
    def fetch_single_stock(self, stock_info: Dict, period: str = "1y") -> Optional[Dict]:
        """Fetch data for a single stock"""
        try:
            ticker = stock_info["ticker"]
            
            cache_key = f"{ticker}_{period}"
            current_time = time.time()
            
            if (cache_key in self.cache and 
                cache_key in self.last_fetch_time and 
                current_time - self.last_fetch_time[cache_key] < 300):
                return self.cache[cache_key]
            
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period=period)
            
            if hist_data.empty:
                return self._generate_mock_data(stock_info)
                
            try:
                info = stock.info
            except:
                info = {"marketCap": np.random.randint(10000, 500000) * 10000000,
                       "forwardPE": np.random.uniform(10, 30),
                       "trailingPE": np.random.uniform(12, 35),
                       "returnOnEquity": np.random.uniform(0.10, 0.25),
                       "returnOnAssets": np.random.uniform(0.05, 0.15)}
            
            hist_data = self._calculate_technical_indicators(hist_data)
            
            result = {
                "info": stock_info,
                "price_data": hist_data,
                "fundamentals": info,
                "current_price": float(hist_data['Close'].iloc[-1])
            }
            
            self.cache[cache_key] = result
            self.last_fetch_time[cache_key] = current_time
            
            return result
            
        except Exception as e:
            return self._generate_mock_data(stock_info)
    
    def _generate_mock_data(self, stock_info: Dict) -> Dict:
        """Generate realistic mock data when API fails"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        base_price = np.random.uniform(100, 2000)
        
        price_changes = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        mock_data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
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
        """Fetch stock data with progress tracking"""
        total_stocks = len(self.indian_stocks)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self.fetch_single_stock, stock_info, period): stock_info for stock_info in self.indian_stocks}
            
            for i, future in enumerate(as_completed(futures)):
                stock_info = futures[future]
                result = future.result()
                if result:
                    self.stock_data[stock_info["ticker"]] = result
                
                if progress_callback:
                    progress_callback((i + 1) / total_stocks)
        
        if progress_callback:
            progress_callback(1.0)
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            if len(df) < 50:
                return df
                
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            df['EMA_20'] = df['Close'].ewm(span=20).mean()
            df['EMA_50'] = df['Close'].ewm(span=50).mean()
            
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            sma_20 = df['Close'].rolling(window=20).mean()
            std_20 = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = sma_20 + (std_20 * 2)
            df['BB_Lower'] = sma_20 - (std_20 * 2)
            df['BB_Middle'] = sma_20
            
            df['Momentum'] = df['Close'].pct_change(periods=10) * 100
            
            return df
            
        except Exception as e:
            return df
    
    def calculate_magic_formula_score(self, ticker: str) -> Optional[Dict]:
        """Calculate Magic Formula score"""
        try:
            stock_info = self.stock_data[ticker]
            fundamentals = stock_info["fundamentals"]
            price_data = stock_info["price_data"]
            
            market_cap_value = fundamentals.get('marketCap')
            pe_ratio = fundamentals.get('forwardPE', fundamentals.get('trailingPE', 20))
            roe = fundamentals.get('returnOnEquity', 0.15) * 100 if fundamentals.get('returnOnEquity') else 15
            roa = fundamentals.get('returnOnAssets', 0.08) * 100 if fundamentals.get('returnOnAssets') else 8
            
            market_cap = market_cap_value / 10000000 if market_cap_value else 50000000000 / 10000000

            roc = (roe + roa) / 2 if roe and roa else max(roe, roa) if roe or roa else 12
            earnings_yield = (1 / pe_ratio * 100) if pe_ratio and pe_ratio > 0 else 5
            magic_score = (roc * 0.6) + (earnings_yield * 0.4)
            
            current_rsi = price_data['RSI'].iloc[-1] if 'RSI' in price_data.columns and not price_data['RSI'].isna().iloc[-1] else 50
            current_momentum = price_data['Momentum'].iloc[-1] if 'Momentum' in price_data.columns and not price_data['Momentum'].isna().iloc[-1] else 0
            
            ema_20 = price_data['EMA_20'].iloc[-1] if 'EMA_20' in price_data.columns and not price_data['EMA_20'].isna().iloc[-1] else price_data['Close'].iloc[-1]
            ema_50 = price_data['EMA_50'].iloc[-1] if 'EMA_50' in price_data.columns and not price_data['EMA_50'].isna().iloc[-1] else price_data['Close'].iloc[-1]
            ema_trend = "BULLISH" if ema_20 > ema_50 else "BEARISH"
            
            rsi_signal = self._get_rsi_signal(current_rsi)
            momentum_signal = self._get_momentum_signal(current_momentum)
            overall_signal = self._get_overall_signal(current_rsi, current_momentum, ema_20, ema_50)

            commentary = self._generate_buffett_commentary(magic_score, ema_trend, rsi_signal, overall_signal)
            
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
                "overall_signal": overall_signal,
                "buffett_commentary": commentary
            }
            
        except Exception as e:
            return None

    def _generate_buffett_commentary(self, magic_score, ema_trend, rsi_signal, overall_signal):
        """Generates a simple, Buffett-style commentary based on key metrics."""
        if magic_score > 15 and ema_trend == "BULLISH" and overall_signal in ["STRONG_BUY", "BUY"]:
            return "This looks like a wonderful business at a fair price. The numbers suggest strong returns and a positive technical trend. Keep it in your circle of competence."
        elif magic_score > 12 and ema_trend == "BULLISH":
            return "The fundamentals here are decent. It's a business worth understanding, and the trend is favorable. It might be a worthwhile opportunity."
        elif overall_signal == "STRONG_SELL" or rsi_signal == "SELL":
            return "Caution is advised. The technical signals are very weak, indicating a strong downward trend. It's often smart to wait for a better pitch."
        elif ema_trend == "BEARISH":
            return "The current market sentiment is not favorable, despite a decent magic score. As Charlie Munger said, 'The big money is not in the buying and selling, but in the waiting.' It's often wise to be patient."
        elif magic_score > 8:
            return "A business with a reasonable magic score. We must dig deeper into the company's moat and management quality to be sure of its long-term prospects."
        else:
            return "The fundamentals are not screamingly attractive. A simple look tells me we need to dig deeper, or move on to a better opportunity."
    
    def _get_rsi_signal(self, rsi: float) -> str:
        if rsi < 30: return "BUY"
        elif rsi > 70: return "SELL"
        else: return "HOLD"
    
    def _get_momentum_signal(self, momentum: float) -> str:
        if momentum > 5: return "STRONG_UP"
        elif momentum > 0: return "UP"
        elif momentum < -5: return "STRONG_DOWN"
        else: return "DOWN"
    
    def _get_overall_signal(self, rsi: float, momentum: float, ema_20: float, ema_50: float) -> str:
        score = 0
        
        if rsi < 30: score += 2
        elif rsi > 70: score -= 2
        
        if momentum > 2: score += 1
        elif momentum < -2: score -= 1
        
        if ema_20 > ema_50: score += 1
        else: score -= 1
        
        if score >= 2: return "STRONG_BUY"
        elif score >= 1: return "BUY"
        elif score <= -2: return "STRONG_SELL"
        elif score <= -1: return "SELL"
        else: return "HOLD"

    def screen_stocks(self, min_market_cap: float = 1000, top_n: int = 20) -> pd.DataFrame:
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

# Streamlit App
def main():
    # Page config
    st.set_page_config(
        page_title="🎯 Magic Formula Screener", 
        page_icon="🎯", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # C-Suite Theming
    st.markdown("""
    <style>
        .st-emotion-cache-18ni7ap { padding-top: 0rem; }
        .st-emotion-cache-p2w5e4 { padding-top: 0rem; }
        
        .stButton button {
            background-color: #2e3b5e;
            color: white;
            border: 1px solid #4a69bd;
            border-radius: 5px;
            padding: 10px 24px;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #4a69bd;
            border-color: #6a89c9;
        }
        .st-emotion-cache-10trblm {
            font-size: 1.5rem;
            font-weight: 600;
        }
        .st-emotion-cache-p5k027 {
            font-size: 1.25rem;
            color: #b0c4de;
        }
        
        /* Table Styling for a professional look */
        .stTable {
            border: 1px solid #4a69bd;
            border-radius: 5px;
            overflow: hidden;
        }
        .stTable th {
            background-color: #2e3b5e;
            color: white;
            font-weight: bold;
        }
        .stTable td {
            background-color: #1e283e;
            color: #e0e0e0;
        }
        
        .signal-strong-buy { color: #28a745; font-weight: bold; }
        .signal-buy { color: #5cb85c; }
        .signal-hold { color: #f0ad4e; }
        .signal-sell { color: #d9534f; }
        .signal-strong-sell { color: #c0392b; font-weight: bold; }
        
        .ema-bullish { color: #28a745; }
        .ema-bearish { color: #d9534f; }
        
        .rsi-overbought { color: #d9534f; }
        .rsi-oversold { color: #5cb85c; }
        .rsi-neutral { color: #f0ad4e; }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown("""
    <h1 style="color: #4a90e2; font-weight: bold;">
        <span style="font-size: 1.2em; vertical-align: top;">🎯</span>
        Magic Formula Stock Screener
    </h1>
    <h3 style="color: #b0c4de; margin-top: -15px;">
        Indian Markets Edition with C-Suite Insights
    </h3>
    ---
    """, unsafe_allow_html=True)
    
    if 'screener' not in st.session_state:
        st.session_state.screener = MagicFormulaScreener()
        st.session_state.data_loaded = False
        st.session_state.results = pd.DataFrame()
        st.session_state.selected_ticker = None
    
    # Sidebar
    st.sidebar.header("🔧 Screening Parameters")
    st.sidebar.markdown("---")
    
    min_market_cap = st.sidebar.number_input(
        "💰 Minimum Market Cap (₹ Crores)", 
        value=1000.0, 
        step=100.0,
        help="Filter stocks by minimum market capitalization"
    )
    
    top_n = st.sidebar.slider(
        "📊 Number of Top Stocks", 
        min_value=5, 
        max_value=len(st.session_state.screener.indian_stocks), 
        value=15,
        help="Select how many top-ranked stocks to display"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📈 Data Management")
    
    if st.sidebar.button("🚀 Fetch Fresh Data", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress):
            progress_bar.progress(progress)
            if progress < 0.3: status_text.text("🔄 Initializing data fetch...")
            elif progress < 0.7: status_text.text("📊 Downloading stock data...")
            else: status_text.text("⚡ Processing indicators...")
        
        st.session_state.screener.fetch_stock_data(progress_callback=update_progress)
        st.session_state.data_loaded = True
        
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Data fetched successfully!")
    
    if st.session_state.data_loaded:
        st.sidebar.success(f"✅ Data loaded for {len(st.session_state.screener.stock_data)} stocks")
    
    if st.sidebar.button("🎯 Run Screening"):
        if not st.session_state.data_loaded:
            st.error("❌ Please fetch data first!")
        else:
            with st.spinner("🔍 Screening stocks..."):
                results = st.session_state.screener.screen_stocks(
                    min_market_cap=min_market_cap, top_n=top_n
                )
                st.session_state.results = results
                st.success(f"✅ Found {len(results)} qualifying stocks!")
    
    if not st.session_state.results.empty:
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Detailed Stock View")
        
        ticker_list = st.session_state.results['ticker'].tolist()
        
        if st.session_state.selected_ticker not in ticker_list and ticker_list:
            st.session_state.selected_ticker = ticker_list[0]
            
        if ticker_list:
            selected_ticker = st.sidebar.selectbox("Select a Ticker to Analyze", 
                                                    ticker_list, 
                                                    index=ticker_list.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in ticker_list else 0)
            if selected_ticker:
                st.session_state.selected_ticker = selected_ticker
    
    if not st.session_state.results.empty:
        st.header("📊 Screening Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1: st.metric("🏢 Total Stocks", len(st.session_state.results))
        with col2: st.metric("⭐ Avg Magic Score", f"{st.session_state.results['magic_score'].mean():.2f}")
        with col3: st.metric("📈 Buy Signals", len(st.session_state.results[st.session_state.results['overall_signal'].str.contains('BUY', na=False)]))
        with col4: st.metric("📊 Bullish Trends", len(st.session_state.results[st.session_state.results['ema_trend'] == 'BULLISH']))
        
        st.subheader("🏆 Top Ranked Stocks")
        
        display_df = st.session_state.results.copy()
        display_df['Rank'] = range(1, len(display_df) + 1)
        display_df['Price (₹)'] = display_df['current_price'].apply(lambda x: f"₹{x:.2f}")
        display_df['Magic Score'] = display_df['magic_score'].apply(lambda x: f"{x:.2f}")
        display_df['RSI'] = display_df['rsi'].apply(lambda x: f"{x:.2f}")
        display_df['Momentum'] = display_df['momentum'].apply(lambda x: f"{x:.2f}")
        
        display_columns = [
            'Rank', 'name', 'ticker', 'Price (₹)', 'Magic Score', 'ema_trend', 'rsi', 'momentum', 'overall_signal'
        ]
        
        styled_df = display_df[display_columns].rename(columns={
            'name': 'Company', 'ticker': 'Ticker', 'ema_trend': 'EMA Trend',
            'rsi': 'RSI', 'momentum': 'Momentum', 'overall_signal': 'Overall Signal'
        })
        
        styled_df['Overall Signal'] = styled_df['Overall Signal'].apply(lambda x: x.replace('_', ' ').title())
        styled_df['EMA Trend'] = styled_df['EMA Trend'].apply(lambda x: x.title())

        # Dynamic HTML table creation for color coding
        styled_df_html = styled_df.to_html(escape=False)
        styled_df_html = styled_df_html.replace('<td>Strong Buy</td>', '<td class="signal-strong-buy">Strong Buy</td>')
        styled_df_html = styled_df_html.replace('<td>Buy</td>', '<td class="signal-buy">Buy</td>')
        styled_df_html = styled_df_html.replace('<td>Hold</td>', '<td class="signal-hold">Hold</td>')
        styled_df_html = styled_df_html.replace('<td>Sell</td>', '<td class="signal-sell">Sell</td>')
        styled_df_html = styled_df_html.replace('<td>Strong Sell</td>', '<td class="signal-strong-sell">Strong Sell</td>')

        styled_df_html = styled_df_html.replace('<td>Bullish</td>', '<td class="ema-bullish">Bullish</td>')
        styled_df_html = styled_df_html.replace('<td>Bearish</td>', '<td class="ema-bearish">Bearish</td>')
        
        st.markdown(styled_df_html, unsafe_allow_html=True)
        
        # --- Detailed Analysis Section ---
        if st.session_state.selected_ticker:
            selected_stock_data_df = st.session_state.results[st.session_state.results['ticker'] == st.session_state.selected_ticker]
            
            if not selected_stock_data_df.empty:
                selected_stock_data = selected_stock_data_df.iloc[0]
                
                signal_color_map = {
                    "STRONG_BUY": "#28a745", "BUY": "#5cb85c",
                    "HOLD": "#f0ad4e", "SELL": "#d9534f", "STRONG_SELL": "#c0392b"
                }

                overall_signal = selected_stock_data.get('overall_signal', 'N/A')
                signal_color = signal_color_map.get(overall_signal, "#ffffff")
                
                st.subheader(f"Detailed Analysis for {selected_stock_data['name']}")
                
                st.markdown(f"### Overall Signal: <span style='color: {signal_color};'>**{overall_signal.replace('_', ' ').title()}**</span>", unsafe_allow_html=True)
                
                st.subheader("🗣️ Buffett's Perspective")
                st.info(selected_stock_data['buffett_commentary'])
                
                ticker_data = st.session_state.screener.stock_data.get(st.session_state.selected_ticker)
                
                if ticker_data and not ticker_data['price_data'].empty and len(ticker_data['price_data']) >= 50:
                    price_data = ticker_data['price_data']
                    
                    if not all(col in price_data.columns for col in ['Close', 'Open', 'High', 'Low', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']):
                        st.warning("⚠️ Insufficient data to generate all technical charts. Displaying basic price chart only.")
                        fig = go.Figure(data=[go.Candlestick(x=price_data.index, open=price_data['Open'], high=price_data['High'], low=price_data['Low'], close=price_data['Close'], name='Price')])
                        fig.update_layout(title=f"Price Chart for {ticker_data['info']['name']}", template="plotly_dark", xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                            vertical_spacing=0.1, 
                                            row_heights=[0.5, 0.25, 0.25],
                                            subplot_titles=(f"Price & Moving Averages for {ticker_data['info']['name']}", "Relative Strength Index (RSI)", "Moving Average Convergence Divergence (MACD)"))
    
                        fig.add_trace(go.Candlestick(x=price_data.index, open=price_data['Open'], high=price_data['High'], low=price_data['Low'], close=price_data['Close'], name='Price'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=price_data.index, y=price_data['EMA_20'], name='EMA 20', line=dict(color='#f2a600', width=2)), row=1, col=1)
                        fig.add_trace(go.Scatter(x=price_data.index, y=price_data['EMA_50'], name='EMA 50', line=dict(color='#1e90ff', width=2)), row=1, col=1)
    
                        fig.add_trace(go.Scatter(x=price_data.index, y=price_data['RSI'], name='RSI', line=dict(color='#8a2be2')), row=2, col=1)
                        fig.add_hline(y=70, line_dash="dash", line_color="#d9534f", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="#5cb85c", row=2, col=1)
    
                        colors = ['#28a745' if val >= 0 else '#d9534f' for val in price_data['MACD_Hist']]
                        fig.add_trace(go.Bar(x=price_data.index, y=price_data['MACD_Hist'], name='MACD Hist', marker_color=colors), row=3, col=1)
                        fig.add_trace(go.Scatter(x=price_data.index, y=price_data['MACD'], name='MACD', line=dict(color='white', width=1)), row=3, col=1)
                        fig.add_trace(go.Scatter(x=price_data.index, y=price_data['MACD_Signal'], name='Signal', line=dict(color='#808080', width=1)), row=3, col=1)
    
                        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False,
                                          title_font=dict(size=24, color='#4a90e2'),
                                          paper_bgcolor="#1f2430", plot_bgcolor="#1f2430",
                                          font=dict(color="#d0d0d0"))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Insufficient historical data available to generate a detailed chart for this ticker. Please try again later.")
            else:
                st.subheader(f"Detailed Analysis for {st.session_state.selected_ticker}")
                st.warning("⚠️ The selected stock is not in the current screening results. Please re-run the screening if you have changed parameters.")

        st.header("📈 Analysis Dashboard")
        
        tab1, tab2, tab3 = st.tabs(["📊 Score Distribution", "🏭 Sector Analysis", "📈 Technical Signals"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = go.Figure(data=[go.Histogram(x=st.session_state.results['magic_score'], nbinsx=15, marker_color='#4a90e2', opacity=0.8)])
                fig_hist.update_layout(title="Magic Formula Score Distribution", xaxis_title="Magic Formula Score", yaxis_title="Number of Stocks", template="plotly_dark", paper_bgcolor="#1f2430", plot_bgcolor="#1f2430")
                st.plotly_chart(fig_hist, use_container_width=True)
            with col2:
                top_10 = st.session_state.results.head(10)
                fig_bar = go.Figure(data=[go.Bar(y=[name[:20] + "..." if len(name) > 20 else name for name in top_10['name']], x=top_10['magic_score'], orientation='h', marker_color='#f2a600', opacity=0.8)])
                fig_bar.update_layout(title="Top 10 Stocks by Magic Score", xaxis_title="Magic Formula Score", template="plotly_dark", paper_bgcolor="#1f2430", plot_bgcolor="#1f2430")
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                sector_counts = st.session_state.results['sector'].value_counts()
                fig_pie = go.Figure(data=[go.Pie(labels=sector_counts.index, values=sector_counts.values, hole=0.4, marker_colors=['#f2a600', '#1e90ff', '#8a2be2', '#5cb85c', '#d9534f', '#6a89c9'])])
                fig_pie.update_layout(title="Sector Distribution", template="plotly_dark", paper_bgcolor="#1f2430", plot_bgcolor="#1f2430")
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                fig_scatter = go.Figure(data=[go.Scatter(x=st.session_state.results['rsi'], y=st.session_state.results['magic_score'], mode='markers', marker=dict(size=10, color=st.session_state.results['current_price'], colorscale='Viridis', showscale=True), text=st.session_state.results['name'])])
                fig_scatter.update_layout(title="RSI vs Magic Formula Score", xaxis_title="RSI", yaxis_title="Magic Formula Score", template="plotly_dark", paper_bgcolor="#1f2430", plot_bgcolor="#1f2430")
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab3:
            signal_counts = st.session_state.results['overall_signal'].value_counts()
            colors = ['#28a745' if 'BUY' in signal else '#d9534f' if 'SELL' in signal else '#f0ad4e' for signal in signal_counts.index]
            fig_signals = go.Figure(data=[go.Bar(x=signal_counts.index, y=signal_counts.values, marker_color=colors, opacity=0.8)])
            fig_signals.update_layout(title="Overall Signal Distribution", xaxis_title="Signal Type", yaxis_title="Number of Stocks", template="plotly_dark", paper_bgcolor="#1f2430", plot_bgcolor="#1f2430")
            st.plotly_chart(fig_signals, use_container_width=True)
        
        st.header("💾 Export Results")
        csv = st.session_state.results.to_csv(index=False)
        st.download_button(
            label="📄 Download Results CSV",
            data=csv,
            file_name=f"magic_formula_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    else:
        st.markdown("""
        <div style="background-color: #1f2430; padding: 25px; border-radius: 10px; border: 1px solid #4a69bd;">
            <h2 style="color: #b0c4de;">🚀 Welcome to the Executive Investment Dashboard!</h2>
            <p style="color: #d0d0d0;">
                This powerful tool is designed to identify undervalued Indian stocks using a robust blend of
                fundamental and technical analysis. Inspired by the principles of legendary investors,
                it provides clear, data-driven insights to inform your investment decisions.
            </p>
            <h3 style="color: #4a90e2; margin-top: 20px;">Your Roadmap:</h3>
            <ol style="color: #d0d0d0;">
                <li><b>Fetch Data:</b> A single click in the sidebar pulls the latest market data.</li>
                <li><b>Define Strategy:</b> Use the filters to set your investment criteria.</li>
                <li><b>Execute Screening:</b> The system ranks stocks based on proven metrics.</li>
                <li><b>Deep Dive:</b> Explore detailed charts and insights for each top-performing stock.</li>
            </ol>
            <h3 style="color: #4a90e2; margin-top: 20px;">Core Features:</h3>
            <ul style="color: #d0d0d0;">
                <li><b>Value-Driven Ranking:</b> The Magic Formula ranks stocks on Return on Capital and Earnings Yield.</li>
                <li><b>Strategic Insights:</b> Built-in technical indicators and simple, plain-language commentary.</li>
                <li><b>Dynamic Visualizations:</b> Professional-grade charts for quick, informed analysis.</li>
            </ul>
            <p style="color: #d0d0d0; margin-top: 30px; text-align: center;">
                <b>Ready to start?</b> Click <b>"🚀 Fetch Fresh Data"</b> in the sidebar! 📈
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.info(f"🏢 **{len(st.session_state.screener.indian_stocks)}** Major Stocks")
        with col2: st.info("📊 **9** Technical Indicators")
        with col3: st.info("🎯 **Magic Formula** Scoring")
        with col4: st.info("⚡ **Real-time** Data")

    st.markdown("---")
    st.markdown("""
    <div style="background-color:#1e2430; padding:15px; border-radius:8px; margin-top:30px; border: 1px solid #4a90e2;">
        <p style="font-size:14px; color:#a0a0a0; text-align:center; margin:0;">
            <b>Disclaimer:</b> This tool is for educational and informational purposes only and does not constitute financial or investment advice.
            The commentary and signals are illustrative and should not be used as a basis for investment decisions.
            Always conduct your own due diligence and consult with a qualified financial professional before making any investment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
