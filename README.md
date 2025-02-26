from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import xgboost as xgb
import requests
from datetime import datetime, timedelta
import yfinance as yf
import json
import redis
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Configuration
REDIS_CONFIG = {'HOST': '0.0.0.0', 'PORT': 6379, 'DB': 0}
APP_CONFIG = {'CACHE_EXPIRACION': 30}

app = FastAPI(title="Stock Market AI Analysis",
              description="Advanced stock market analysis with AI")

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Redis setup
try:
    redis_client = redis.StrictRedis(host=REDIS_CONFIG['HOST'],
                                     port=REDIS_CONFIG['PORT'],
                                     db=REDIS_CONFIG['DB'],
                                     socket_timeout=5,
                                     decode_responses=True)
    redis_client.ping()
except redis.ConnectionError:
    print("Warning: Redis connection failed. Using in-memory cache.")
    redis_client = None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/technical_analysis/{symbol}")
async def technical_analysis(symbol: str):
    cache_key = f"technical_analysis_{symbol}"

    if redis_client:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)

    try:
        df = yf.download(symbol, period="1d", interval="1m")

        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for symbol")

        data = {
            "current_price": float(df['Close'][-1]),
            "day_high": float(df['High'].max()),
            "day_low": float(df['Low'].min()),
            "volume": float(df['Volume'].sum()),
            "price_change": float(df['Close'][-1] - df['Open'][0]),
            "price_change_percent": float((df['Close'][-1] - df['Open'][0]) / df['Open'][0] * 100),
            "last_updated": df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        }

        if redis_client:
            redis_client.setex(
                cache_key,
                APP_CONFIG['CACHE_EXPIRACION'],
                json.dumps(data)
            )

        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/{symbol}", response_class=HTMLResponse)
async def analysis(request: Request, symbol: str):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")

        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        hist['MA50'] = hist['Close'].rolling(window=50).mean()

        X = np.array(range(len(hist))).reshape(-1, 1)
range(len(hist))).reshape(-1, 1)
        y = hist['Close'].values

        # Enhanced ML models
        rf_model = RandomForestRegressor(n_estimators=100)
        gb_model = GradientBoostingRegressor(n_estimators=100)
        xgb_model = xgb.XGBRegressor(n_estimators=100)

        rf_model.fit(X, y)
        gb_model.fit(X, y)
        xgb_model.fit(X, y)

        prediction = (
            rf_model.predict([[len(hist)]])[0] +
            gb_model.predict([[len(hist)]])[0] +
            xgb_model.predict([[len(hist)]])[0]
        ) / 3

        data = hist.reset_index()
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

        # Sentiment analysis
        news = stock.news
        sentiment_scores = [TextBlob(item.get('title', '')).sentiment.polarity for item in news]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

        sentiment_status = "Positivo" if avg_sentiment > 0 else "Negativo" if avg_sentiment < 0 else "Neutral"
        # Calculate additional metrics
        rsi = 100 - (100 / (1 + (hist['Close'].diff(1)[hist['Close'].diff(1) > 0].mean() / 
                                -hist['Close'].diff(1)[hist['Close'].diff(1) < 0].mean())))
        
        price_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
        volume_trend = "Aumentando" if hist['Volume'].iloc[-5:].mean() > hist['Volume'].iloc[-10:-5].mean() else "Disminuyendo"
        
        ai_analysis = {
            'current_price': f"${hist['Close'].iloc[-1]:.2f}",
            'avg_price': f"${hist['Close'].mean():.2f}",
            'volatility': f"{hist['Close'].std():.2f}%",
            'momentum': "Bullish" if hist['Close'].iloc[-1] > hist['Close'].iloc[-20] else "Bearish",
            'sentiment': sentiment_status,
            'rsi': f"{rsi:.1f}",
            'price_change_percent': f"{price_change:.1f}%",
            'volume_trend': volume_trend,
            'support_level': f"${hist['Close'].rolling(20).min().iloc[-1]:.2f}",
            'resistance_level': f"${hist['Close'].rolling(20).max().iloc[-1]:.2f}",
            'ml_confidence': f"{(abs(prediction - hist['Close'].iloc[-1]) / hist['Close'].iloc[-1] * 100):.1f}%",
            'summary': f"Based on technical analysis and ML predictions, the stock shows {volume_trend.lower()} volume with RSI at {rsi:.1f} suggesting {'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'} conditions. Market sentiment is {sentiment_status.lower()} with a {price_change:.1f}% price change over the period."
        }

        return templates.TemplateResponse("analysis.html", {
            "request": request,
            "symbol": symbol,
            "data": data.to_dict('records'),
            "request": request,
            "symbol": symbol,
            "data": data.to_dict('records'),
            "ai_analysis": ai_analysis,
            "prediction": prediction
        })

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Market AI Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link href="/static/styles.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-gray-900 text-white">
    <nav class="bg-gray-800 border-b border-gray-700">
        <div class="container mx-auto px-4 py-3">
            <div class="flex justify-between items-center">
                <a href="/" class="text-2xl font-bold text-yellow-500">CryptoAnalytics AI</a>
            </div>
        </div>
    </nav>
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold mb-6 text-yellow-500">Stock Market AI Analysis</h1>
        
        <div class="bg-gray-800 rounded-lg shadow-xl p-6">
            <form id="analysisForm" class="mb-4">
                <div class="flex gap-4">
                    <input type="text" id="symbol" name="symbol" 
                           placeholder="Ingrese símbolo (ej: AAPL, MSFT)" 
                           class="flex-1 p-2 rounded bg-gray-700 text-white border border-gray-600"
                           required>
                    <button type="submit" 
                            class="bg-yellow-500 hover:bg-yellow-600 text-black font-bold py-2 px-4 rounded">
                        Analizar
                    </button>
                </div>
            </form>
        </div>
    </div>
    <script src="/static/js/main.js"></script>
</body>
</html>
/ Import Chart.js from CDN
const script = document.createElement('script');
script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
document.head.appendChild(script);

// Analysis form handling
function handleAnalysis(event) {
    event.preventDefault();
    const symbol = document.getElementById('symbol').value;
    try {/
        window.location.href = `/analysis/${symbol}`;
    } catch (error) {
        console.error('Analysis error:', error);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    try {
        // Analysis Form Handling
        const analysisForm = document.getElementById('analysisForm');
        if (analysisForm) {
            analysisForm.addEventListener('submit', handleAnalysis);
        }


        const ctx = document.getElementById('priceChart').getContext('2d');

        const dates = window.stockData.map(item => item.Date);
        const prices = window.stockData.map(item => item.Close);

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: 'Precio de Cierre',
                    data: prices,
                    borderColor: '#EAB308',
                    tension: 0.1,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: {
                            color: '#FFFFFF'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#FFFFFF'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#FFFFFF'
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error("Error in page script:", error);
    }
});

function updateAnalysis(symbol) {
    fetch(`/technical_analysis/${symbol}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById('analysis').innerHTML = `
                <div class="card">
                    <h3>Technical Analysis</h3>
                    <p>RSI: ${data.RSI?.toFixed(2) || 'N/A'}</p>
                    <p>MACD: ${data.MACD?.toFixed(2) || 'N/A'}</p>
                </div>`;
        });
}