"""
Druta Kart - Main FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
import os
import time

load_dotenv()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Druta Kart AI Support",
    description="Multilingual Agentic AI Customer Support for Quick Commerce",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "app": os.getenv("APP_NAME", "Druta Kart"),
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "timestamp": time.time()
    }


@app.get("/")
async def root():
    return {
        "message": "🚀 Druta Kart AI Support API is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("🚀 Druta Kart Backend Starting...")
    print(f"📦 App: {os.getenv('APP_NAME')}")
    print(f"🔧 Debug: {os.getenv('DEBUG')}")
    print(f"🤖 Model: {os.getenv('GROQ_TEXT_MODEL')}")
    print("=" * 50)