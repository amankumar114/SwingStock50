import yfinance as yf
import pandas as pd
import smtplib
import os
import json
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import numpy as np
import talib

# === CONFIG ===
NIFTY50_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
    'INFY.NS', 'HINDUNILVR.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'ASIANPAINT.NS',
    'AXISBANK.NS', 'LT.NS', 'MARUTI.NS', 'BAJFINANCE.NS', 'WIPRO.NS',
    'ONGC.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'SUNPHARMA.NS', 'NESTLEIND.NS',
    'TECHM.NS', 'BHARTIARTL.NS', 'TATASTEEL.NS', 'POWERGRID.NS', 'NTPC.NS',
    'INDUSINDBK.NS', 'BAJAJ-AUTO.NS', 'M&M.NS', 'BRITANNIA.NS', 'HCLTECH.NS',
    'DRREDDY.NS', 'EICHERMOT.NS', 'ADANIPORTS.NS', 'JSWSTEEL.NS', 'CIPLA.NS',
    'GRASIM.NS', 'BAJAJFINSV.NS', 'HEROMOTOCO.NS', 'COALINDIA.NS', 'DIVISLAB.NS',
    'ITC.NS', 'SBILIFE.NS', 'UPL.NS', 'BPCL.NS', 'HINDALCO.NS', 'TATAMOTORS.NS',
    'APOLLOHOSP.NS', 'ADANIENT.NS', 'TATACONSUM.NS'
]

SUPPORT_EMA_DAYS = [50, 100, 200]  # Key weekly EMAs for support
RSI_OVERSOLD = 40
RSI_NEUTRAL = 50
SUPPORT_THRESHOLD = 3.0  # % distance to consider near support
MIN_WEEKS_DATA = 100  # Minimum weeks of data required (approx 2 years)

# === CONFIG ===
EMAIL_SENDER = os.environ.get('EMAIL_SENDER')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')
EMAIL_RECEIVER = os.environ.get('EMAIL_RECEIVER')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nifty_swing_trades.log'),
        logging.StreamHandler()
    ]
)

def fetch_weekly_data(ticker):
    """Fetch weekly historical data with robust error handling"""
    try:
        df = yf.download(ticker, period='max', interval='1wk', auto_adjust=True, progress=False)
        if df.empty:
            logging.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        if len(df) < MIN_WEEKS_DATA:
            logging.warning(f"Insufficient data for {ticker} - {len(df)} weeks")
            return pd.DataFrame()
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    """Calculate technical indicators with robust data handling"""
    try:
        if df.empty or len(df) < 200:  # Ensure sufficient data
            return None
            
        # Create clean price array with NaN handling
        close_prices = df['Close'].values
        if close_prices.ndim != 1:
            close_prices = close_prices.flatten()
            
        # Filter out NaN and infinite values
        valid_mask = ~np.isnan(close_prices) & ~np.isinf(close_prices)
        close_prices = close_prices[valid_mask]
        
        if len(close_prices) < 200:
            return None
            
        # Calculate EMAs
        ema_values = {}
        for period in SUPPORT_EMA_DAYS:
            if len(close_prices) < period:
                ema_values[period] = None
                continue
                
            ema = talib.EMA(close_prices.astype('float64'), timeperiod=period)
            if not np.isnan(ema[-1]):
                ema_values[period] = ema[-1]
            else:
                ema_values[period] = None
        
        # Calculate RSI
        rsi = np.nan
        if len(close_prices) >= 14:
            rsi_values = talib.RSI(close_prices.astype('float64'), timeperiod=14)
            rsi = rsi_values[-1] if not np.isnan(rsi_values[-1]) else np.nan
        
        # Calculate MACD
        macd, signal = np.nan, np.nan
        macd_bullish = False
        if len(close_prices) >= 26:
            macd_line, signal_line, _ = talib.MACD(
                close_prices.astype('float64'),
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            macd = macd_line[-1] if not np.isnan(macd_line[-1]) else np.nan
            signal = signal_line[-1] if not np.isnan(signal_line[-1]) else np.nan
            
            # Check for MACD bullish crossover
            if len(macd_line) >= 3 and not np.isnan(macd) and not np.isnan(signal):
                # Current crossover: MACD crosses above signal line
                current_crossover = (macd > signal) and (macd_line[-2] <= signal_line[-2])
                
                # Recent crossover: MACD above signal and both rising
                recent_crossover = (macd > signal) and \
                                  (macd_line[-2] > signal_line[-2]) and \
                                  (macd > macd_line[-2]) and \
                                  (signal > signal_line[-2])
                
                macd_bullish = current_crossover or recent_crossover
        
        return {
            'current_close': close_prices[-1],
            'emas': ema_values,
            'rsi': rsi,
            'macd': macd,
            'signal': signal,
            'macd_bullish': macd_bullish
        }
    except Exception as e:
        logging.error(f"Technical indicator error: {str(e)}")
        return None

def calculate_support_zone(close_price, ema_values):
    """Determine how close price is to key support EMAs"""
    closest_support = None
    min_distance = float('inf')
    
    for period, ema_value in ema_values.items():
        if ema_value is None or np.isnan(ema_value):
            continue
            
        # Calculate percentage distance to EMA
        distance_pct = abs(close_price - ema_value) / ema_value * 100
        
        # Check if within support threshold and below EMA (support from below)
        if distance_pct < min_distance and close_price < ema_value:
            min_distance = distance_pct
            closest_support = {
                'ema_period': period,
                'ema_value': ema_value,
                'distance_pct': distance_pct
            }
    
    return closest_support, min_distance

def generate_stock_report(stock_data):
    """Generate HTML report for stocks meeting criteria"""
    today = datetime.now().strftime('%d %B %Y')
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NIFTY 50 Swing Trade Opportunities</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f5f8fa;
                color: #333;
                line-height: 1.6;
                padding: 20px;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            .header {{
                background: linear-gradient(135deg, #1a2a6c, #b21f1f, #1a2a6c);
                color: white;
                text-align: center;
                padding: 25px;
                border-radius: 10px 10px 0 0;
            }}
            .header h1 {{
                margin: 0;
                font-size: 24px;
            }}
            .header p {{
                margin: 10px 0 0;
                opacity: 0.9;
            }}
            .card {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                margin: 20px;
                overflow: hidden;
                border-left: 4px solid #4CAF50;
            }}
            .card-header {{
                background-color: #f9f9f9;
                padding: 15px 20px;
                border-bottom: 1px solid #eee;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .stock-name {{
                font-weight: bold;
                font-size: 18px;
                color: #1a237e;
            }}
            .stock-rating {{
                background-color: #4CAF50;
                color: white;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 14px;
            }}
            .card-body {{
                padding: 20px;
            }}
            .data-row {{
                display: flex;
                margin-bottom: 15px;
                flex-wrap: wrap;
            }}
            .data-item {{
                flex: 1;
                min-width: 200px;
                margin-bottom: 10px;
            }}
            .data-label {{
                font-weight: 600;
                color: #666;
                font-size: 14px;
                margin-bottom: 5px;
            }}
            .data-value {{
                font-size: 16px;
                font-weight: 600;
            }}
            .support-distance {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 4px;
                background-color: #e8f5e9;
                color: #2e7d32;
                font-weight: bold;
            }}
            .rsi-value {{
                color: #c62828;
                font-weight: bold;
            }}
            .macd-bullish {{
                color: #388e3c;
                font-weight: bold;
            }}
            .reasoning {{
                background-color: #f1f8e9;
                padding: 15px;
                border-radius: 8px;
                margin-top: 15px;
                font-size: 14px;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                color: #777;
                font-size: 12px;
                border-top: 1px solid #eee;
            }}
            .no-signals {{
                text-align: center;
                padding: 40px;
                color: #777;
            }}
            .rating-strong {{
                background-color: #4CAF50;
            }}
            .rating-good {{
                background-color: #2196F3;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>NIFTY 50 Swing Trade Opportunities</h1>
                <p>Stocks near key support levels with bullish indicators</p>
                <p>{today}</p>
            </div>
    """
    
    if not stock_data:
        html += """
            <div class="no-signals">
                <h3>No Swing Trade Opportunities Found This Week</h3>
                <p>No stocks currently meet the criteria for swing trading opportunities.</p>
                <p>Check back next week for updated signals.</p>
            </div>
        """
    else:
        for stock in stock_data:
            if stock['distance_pct'] < 1.5:
                rating = "Strong Buy"
                rating_class = "stock-rating rating-strong"
            else:
                rating = "Buy"
                rating_class = "stock-rating rating-good"
            
            html += f"""
            <div class="card">
                <div class="card-header">
                    <div class="stock-name">{stock['ticker']}</div>
                    <div class="{rating_class}">{rating}</div>
                </div>
                <div class="card-body">
                    <div class="data-row">
                        <div class="data-item">
                            <div class="data-label">Current Price</div>
                            <div class="data-value">₹{stock['current_close']:.2f}</div>
                        </div>
                        <div class="data-item">
                            <div class="data-label">Nearest Support</div>
                            <div class="data-value">
                                {stock['support_ema']}-EMA: ₹{stock['support_value']:.2f}
                                <span class="support-distance">({stock['distance_pct']:.2f}% away)</span>
                            </div>
                        </div>
                        <div class="data-item">
                            <div class="data-label">RSI (14-week)</div>
                            <div class="data-value rsi-value">{stock['rsi']:.2f}</div>
                        </div>
                    </div>
                    <div class="data-row">
                        <div class="data-item">
                            <div class="data-label">MACD</div>
                            <div class="data-value">{stock['macd']:.4f}</div>
                        </div>
                        <div class="data-item">
                            <div class="data-label">Signal Line</div>
                            <div class="data-value">{stock['signal']:.4f}</div>
                        </div>
                        <div class="data-item">
                            <div class="data-label">MACD Signal</div>
                            <div class="data-value macd-bullish">Bullish</div>
                        </div>
                    </div>
                    <div class="reasoning">
                        <strong>Trading Rationale:</strong> {stock['ticker']} is trading near its {stock['support_ema']}-week EMA support level 
                        ({stock['distance_pct']:.2f}% below). The weekly RSI of {stock['rsi']:.2f} indicates oversold conditions, 
                        and the MACD shows a bullish signal. This suggests a potential buying opportunity for swing traders 
                        targeting a rebound to recent resistance levels.
                    </div>
                </div>
            </div>
            """
    
    html += """
            <div class="footer">
                <p><strong>Analysis Criteria:</strong> Stocks trading within 3% of key weekly EMAs (50, 100, 200) with RSI < 50 and bullish MACD crossover</p>
                <p><strong>Disclaimer:</strong> This is automated technical analysis. Fundamental factors and market conditions should also be considered. Past performance is not indicative of future results.</p>
                <p>Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def send_email(subject, html_body):
    """Send email with analysis report"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject
        
        msg.attach(MIMEText(html_body, 'html'))
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        logging.info("Email sent successfully")
    except Exception as e:
        logging.error(f"Error sending email: {str(e)}")

def analyze_nifty_stocks():
    """Analyze all NIFTY 50 stocks for swing trade opportunities"""
    logging.info("Starting NIFTY 50 Swing Trade Analysis")
    opportunities = []
    
    for ticker in NIFTY50_TICKERS:
        logging.info(f"Analyzing {ticker}...")
        
        # Fetch weekly data
        weekly_data = fetch_weekly_data(ticker)
        if weekly_data.empty:
            continue
        
        # Calculate indicators
        indicators = calculate_technical_indicators(weekly_data)
        if not indicators:
            continue
        
        # Check for support zone
        support, distance = calculate_support_zone(
            indicators['current_close'],
            indicators['emas']
        )
        
        # Check if all conditions are met
        if (support and 
            distance <= SUPPORT_THRESHOLD and 
            indicators['rsi'] < RSI_NEUTRAL and 
            indicators['macd_bullish'] and
            not np.isnan(indicators['rsi'])):
            
            opportunity = {
                'ticker': ticker,
                'current_close': indicators['current_close'],
                'support_ema': support['ema_period'],
                'support_value': support['ema_value'],
                'distance_pct': distance,
                'rsi': indicators['rsi'],
                'macd': indicators['macd'],
                'signal': indicators['signal']
            }
            
            opportunities.append(opportunity)
            logging.info(f"Found opportunity: {ticker} at ₹{indicators['current_close']:.2f}")
    
    return opportunities

def main():
    """Main function to run analysis and send report"""
    # Analyze stocks
    opportunities = analyze_nifty_stocks()
    
    # Generate email content
    subject = f"NIFTY 50 Swing Trade Opportunities: {len(opportunities)} Stocks Found"
    if not opportunities:
        subject = "NIFTY 50 Swing Trade Report: No Opportunities Found"
    
    html_report = generate_stock_report(opportunities)
    
    # Send email
    send_email(subject, html_report)
    logging.info("Analysis complete. Report sent.")

if __name__ == '__main__':
    main()
