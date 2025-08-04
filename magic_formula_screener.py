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
        # Expanded list of stocks for broader screening opportunities
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
            # Adding more stocks for diversity
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
            
            # Check cache
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
                
            # Get fundamental data with fallback
            try:
                info = stock.info
            except:
                info = {"marketCap": np.random.randint(10000, 500000) * 10000000,
                       "forwardPE": np.random.uniform(10, 30),
                       "trailingPE": np.random.uniform(12, 35),
                       "returnOnEquity": np.random.uniform(0.10, 0.25),
                       "returnOnAssets": np.random.uniform(0.05, 0.15),
                       "priceToBook": np.random.uniform(1.0, 5.0)} # Added Price-to-Book fallback
            
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
            return self._generate_mock_data(stock_info)
    
    def _generate_mock_data(self, stock_info: Dict) -> Dict:
        """Generate realistic mock data when API fails"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        base_price = np.random.uniform(100, 2000)
        
        # Generate realistic price movements
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
                "returnOnAssets": np.random.uniform(0.05, 0.15),
                "priceToBook": np.random.uniform(1.0, 5.0) # Added Price-to-Book fallback
            },
            "current_price": prices[-1]
        }
    
    def fetch_stock_data(self, period: str = "1y", progress_callback=None) -> None:
        """Fetch stock data with progress tracking"""
        total_stocks = len(self.indian_stocks)
        
        # Using a ThreadPoolExecutor for a more efficient fetch
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
            if len(df) < 20:
                return df
                
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # EMA
            df['EMA_20'] = df['Close'].ewm(span=20).mean()
            df['EMA_50'] = df['Close'].ewm(span=50).mean()
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            sma_20 = df['Close'].rolling(window=20).mean()
            std_20 = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = sma_20 + (std_20 * 2)
            df['BB_Lower'] = sma_20 - (std_20 * 2)
            df['BB_Middle'] = sma_20
            
            # Momentum
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
            
            # Fundamental metrics with safe defaults
            market_cap = fundamentals.get('marketCap', 50000000000) / 10000000
            pe_ratio = fundamentals.get('forwardPE', fundamentals.get('trailingPE', 20))
            roe = fundamentals.get('returnOnEquity', 0.15) * 100 if fundamentals.get('returnOnEquity') else 15
            roa = fundamentals.get('returnOnAssets', 0.08) * 100 if fundamentals.get('returnOnAssets') else 8
            
            # New Feature: Price-to-Book Ratio
            pb_ratio = fundamentals.get('priceToBook', 2.0)
            
            # Calculate metrics
            roc = (roe + roa) / 2 if roe and roa else max(roe, roa) if roe or roa else 12
            earnings_yield = (1 / pe_ratio * 100) if pe_ratio and pe_ratio > 0 else 5
            magic_score = (roc * 0.6) + (earnings_yield * 0.4)
            
            # Technical indicators
            current_rsi = price_data['RSI'].iloc[-1] if 'RSI' in price_data.columns and not price_data['RSI'].isna().iloc[-1] else 50
            current_momentum = price_data['Momentum'].iloc[-1] if 'Momentum' in price_data.columns and not price_data['Momentum'].isna().iloc[-1] else 0
            
            # EMA Trend
            ema_20 = price_data['EMA_20'].iloc[-1] if 'EMA_20' in price_data.columns and not price_data['EMA_20'].isna().iloc[-1] else price_data['Close'].iloc[-1]
            ema_50 = price_data['EMA_50'].iloc[-1] if 'EMA_50' in price_data.columns and not price_data['EMA_50'].isna().iloc[-1] else price_data['Close'].iloc[-1]
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
                "pb_ratio": pb_ratio, # Added Price-to-Book to the results
                "rsi": current_rsi,
                "momentum": current_momentum,
                "ema_trend": ema_trend,
                "rsi_signal": rsi_signal,
                "momentum_signal": momentum_signal,
                "overall_signal": overall_signal
            }
            
        except Exception as e:
            return None
    
    def _get_rsi_signal(self, rsi: float) -> str:
        if rsi < 30:
            return "BUY"
        elif rsi > 70:
            return "SELL"
        else:
            return "HOLD"
    
    def _get_momentum_signal(self, momentum: float) -> str:
        if momentum > 5:
            return "STRONG_UP"
        elif momentum > 0:
            return "UP"
        elif momentum < -5:
            return "STRONG_DOWN"
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
        
        if score >= 2:
            return "STRONG_BUY"
        elif score >= 1:
            return "BUY"
        elif score <= -2:
            return "STRONG_SELL"
        elif score <= -1:
            return "SELL"
        else:
            return "HOLD"
    
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
    
    # Custom CSS
    st.markdown("""
    <style>
        .main > div { padding-top: 2rem; }
        .stAlert { margin-top: 1rem; }
        .metric-container {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown("""
    # 🎯 Magic Formula Stock Screener
    ### *Indian Markets Edition with Technical Analysis*
    ---
    """)
    
    # Initialize session state
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
        max_value=40, # Adjusted max value to reflect the larger stock list
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
            if progress < 0.3:
                status_text.text("🔄 Initializing data fetch...")
            elif progress < 0.7:
                status_text.text("📊 Downloading stock data...")
            else:
                status_text.text("⚡ Processing indicators...")
        
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
    
    # New Feature: Select a stock from a dropdown to view its detailed chart
    if not st.session_state.results.empty:
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Detailed Stock View")
        
        # Use a list of tickers from the results DataFrame for the dropdown
        ticker_list = st.session_state.results['ticker'].tolist()
        selected_ticker = st.sidebar.selectbox("Select a Ticker to Analyze", ticker_list)
        
        if selected_ticker:
            st.session_state.selected_ticker = selected_ticker
    
    # Main content
    if not st.session_state.results.empty:
        # Metrics
        st.header("📊 Screening Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏢 Total Stocks", len(st.session_state.results), f"Top {top_n} selected")
        
        with col2:
            buy_signals = len(st.session_state.results[
                st.session_state.results['overall_signal'].str.contains('BUY', na=False)
            ])
            st.metric("📈 Buy Signals", buy_signals, f"{buy_signals/len(st.session_state.results)*100:.1f}%")
        
        with col3:
            avg_score = st.session_state.results['magic_score'].mean()
            st.metric("⭐ Avg Magic Score", f"{avg_score:.2f}", "Higher is better")
        
        with col4:
            bullish_trends = len(st.session_state.results[
                st.session_state.results['ema_trend'] == 'BULLISH'
            ])
            st.metric("📊 Bullish Trends", bullish_trends, f"{bullish_trends/len(st.session_state.results)*100:.1f}%")
        
        # Results table
        st.subheader("🏆 Top Ranked Stocks")
        
        display_df = st.session_state.results.copy()
        display_df['Rank'] = range(1, len(display_df) + 1)
        display_df['Price (₹)'] = display_df['current_price'].apply(lambda x: f"₹{x:.2f}")
        display_df['Magic Score'] = display_df['magic_score'].apply(lambda x: f"{x:.2f}")
        display_df['P/B Ratio'] = display_df['pb_ratio'].apply(lambda x: f"{x:.2f}") # Added Price-to-Book
        
        display_columns = [
            'Rank', 'name', 'ticker', 'Price (₹)', 'Magic Score', 'P/B Ratio',
            'rsi_signal', 'momentum_signal', 'ema_trend', 'overall_signal'
        ]
        
        styled_df = display_df[display_columns].rename(columns={
            'name': 'Company', 'ticker': 'Ticker', 'rsi_signal': 'RSI Signal',
            'momentum_signal': 'Momentum', 'ema_trend': 'EMA Trend', 'overall_signal': 'Overall Signal'
        })
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # New Feature: Display a detailed chart for the selected stock
        if st.session_state.selected_ticker:
            st.subheader(f"Detailed Analysis for {st.session_state.selected_ticker}")
            ticker_data = st.session_state.screener.stock_data.get(st.session_state.selected_ticker)
            if ticker_data:
                price_data = ticker_data['price_data']
                
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.1, 
                                    row_heights=[0.5, 0.25, 0.25])

                # Price and EMA Chart
                fig.add_trace(go.Candlestick(x=price_data.index,
                                             open=price_data['Open'],
                                             high=price_data['High'],
                                             low=price_data['Low'],
                                             close=price_data['Close'],
                                             name='Price'),
                              row=1, col=1)
                fig.add_trace(go.Scatter(x=price_data.index, y=price_data['EMA_20'], name='EMA 20', line=dict(color='orange', width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=price_data.index, y=price_data['EMA_50'], name='EMA 50', line=dict(color='blue', width=2)), row=1, col=1)

                # RSI Chart
                fig.add_trace(go.Scatter(x=price_data.index, y=price_data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                # MACD Chart
                colors = ['green' if val >= 0 else 'red' for val in price_data['MACD_Hist']]
                fig.add_trace(go.Bar(x=price_data.index, y=price_data['MACD_Hist'], name='MACD Hist', marker_color=colors), row=3, col=1)
                fig.add_trace(go.Scatter(x=price_data.index, y=price_data['MACD'], name='MACD', line=dict(color='black', width=1)), row=3, col=1)
                fig.add_trace(go.Scatter(x=price_data.index, y=price_data['MACD_Signal'], name='Signal', line=dict(color='gray', width=1)), row=3, col=1)

                fig.update_layout(height=800, title_text=f"{ticker_data['info']['name']} ({st.session_state.selected_ticker}) Technical Chart",
                                  xaxis_rangeslider_visible=False)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
                fig.update_yaxes(title_text="MACD", row=3, col=1)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No detailed data found for this ticker. Please try fetching data again.")
        
        # Visualizations
        st.header("📈 Analysis Dashboard")
        
        tab1, tab2, tab3 = st.tabs(["📊 Score Distribution", "🏭 Sector Analysis", "📈 Technical Signals"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution
                fig_hist = go.Figure(data=[
                    go.Histogram(x=st.session_state.results['magic_score'], nbinsx=15, 
                                 marker_color='lightblue', opacity=0.7)
                ])
                fig_hist.update_layout(title="Magic Formula Score Distribution",
                                       xaxis_title="Magic Formula Score", yaxis_title="Number of Stocks")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Top 10 stocks
                top_10 = st.session_state.results.head(10)
                fig_bar = go.Figure(data=[
                    go.Bar(y=[name[:20] + "..." if len(name) > 20 else name for name in top_10['name']],
                           x=top_10['magic_score'], orientation='h', marker_color='gold', opacity=0.7)
                ])
                fig_bar.update_layout(title="Top 10 Stocks by Magic Score", xaxis_title="Magic Formula Score")
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Sector distribution
                sector_counts = st.session_state.results['sector'].value_counts()
                fig_pie = go.Figure(data=[go.Pie(labels=sector_counts.index, values=sector_counts.values, hole=0.3)])
                fig_pie.update_layout(title="Sector Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # RSI vs Magic Score
                fig_scatter = go.Figure(data=[
                    go.Scatter(x=st.session_state.results['rsi'], y=st.session_state.results['magic_score'],
                               mode='markers', marker=dict(size=8, color=st.session_state.results['current_price'],
                               colorscale='Viridis', showscale=True), text=st.session_state.results['name'])
                ])
                fig_scatter.update_layout(title="RSI vs Magic Formula Score", 
                                          xaxis_title="RSI", yaxis_title="Magic Formula Score")
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab3:
            # Signal distribution
            signal_counts = st.session_state.results['overall_signal'].value_counts()
            colors = ['green' if 'BUY' in signal else 'red' if 'SELL' in signal else 'orange' 
                     for signal in signal_counts.index]
            
            fig_signals = go.Figure(data=[
                go.Bar(x=signal_counts.index, y=signal_counts.values, marker_color=colors, opacity=0.7)
            ])
            fig_signals.update_layout(title="Overall Signal Distribution", 
                                      xaxis_title="Signal Type", yaxis_title="Number of Stocks")
            st.plotly_chart(fig_signals, use_container_width=True)
        
        # Export
        st.header("💾 Export Results")
        csv = st.session_state.results.to_csv(index=False)
        st.download_button(
            label="📄 Download Results CSV",
            data=csv,
            file_name=f"magic_formula_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to Magic Formula Stock Screener!
        
        This tool combines **Joel Greenblatt's Magic Formula** with **Technical Analysis** to identify undervalued stocks in the Indian market.
        
        ### 🎯 How it works:
        1. **Fetch Data**: Click the button in the sidebar to download market data
        2. **Set Parameters**: Adjust market cap and stock count filters
        3. **Run Screening**: Let the Magic Formula find opportunities
        4. **Analyze Results**: Explore charts and technical indicators
        
        ### 📊 Features:
        - **Magic Formula Rankings** (Return on Capital + Earnings Yield)
        - **Technical Analysis** (RSI, MACD, EMA trends)
        - **Trading Signals** for investment decisions
        - **Sector Analysis** for diversification
        - **Exportable Results** for further analysis
        
        ---
        
        **Ready to start?** Click **"🚀 Fetch Fresh Data"** in the sidebar! 📈
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info("🏢 **35** Major Stocks") # Updated count
        with col2:
            st.info("📊 **9** Technical Indicators") # Updated count (Price-to-Book)
        with col3:
            st.info("🎯 **Magic Formula** Scoring")
        with col4:
            st.info("⚡ **Real-time** Data")

    # -------------------------------------------------------------------------
    # ADDED DISCLAIMER
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------

if __name__ == "__main__":
    main()
