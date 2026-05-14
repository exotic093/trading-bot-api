from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(
    title="Trading Bot API",
    description="FastAPI trading data service with Binance and OANDA adapters",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "Trading Bot API",
        "version": "1.0.0",
        "status": "online"
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/data/fetch")
async def fetch_data(request: dict):
    """
    Fetch OHLCV market data from various adapters.
    
    Request body:
    - adapter: str (binance, polygon, oanda, etc.)
    - symbol: str (BTCUSDT, AAPL, EUR_USD, etc.)
    - timeframe: str (1m, 5m, 15m, 1h, 4h, 1d, etc.)
    - lookback_bars: int (default: 500)
    
    Returns: MarketBundle JSON per blueprint §2.2
    """
    adapter = request.get("adapter")
    symbol = request.get("symbol")
    timeframe = request.get("timeframe")
    
    # Check if required API keys are configured
    required_keys = {
        "binance": ["BINANCE_API_KEY", "BINANCE_API_SECRET"],
        "polygon": ["POLYGON_API_KEY"],
        "oanda": ["OANDA_API_KEY", "OANDA_ACCOUNT_ID"],
        "fred": ["FRED_API_KEY"],
        "tradingeconomics": ["TRADINGECONOMICS_API_KEY"],
        "finnhub": ["FINNHUB_API_KEY"]
    }
    
    if adapter in required_keys:
        missing_keys = [key for key in required_keys[adapter] if not os.getenv(key)]
        if missing_keys:
            return {
                "error": "Missing API credentials",
                "adapter": adapter,
                "missing_keys": missing_keys,
                "message": f"Please set the following environment variables in Railway: {', '.join(missing_keys)}",
                "help": "See .env.example for all required keys"
            }
    
    # TODO: Implement actual data fetching logic
    # For now, return 501 Not Implemented
    return {
        "error": "Not implemented",
        "adapter": adapter,
        "symbol": symbol,
        "timeframe": timeframe,
        "message": "Data adapter not yet implemented. Awaiting API keys and implementation.",
        "status_code": 501
    }
