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
