"""
Druta Kart - Central Configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App
    app_name: str = "Druta Kart"
    app_version: str = "1.0.0"
    debug: bool = True
    port: int = 8000

    # API Keys
    groq_api_key: str
    sarvam_api_key: str
    supabase_url: str
    supabase_anon_key: str
    hf_token: Optional[str] = None

    # Models
    groq_text_model: str = "llama-3.3-70b-versatile"
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_whisper_model: str = "whisper-large-v3"

    # Offer Safety Caps — HARDCODED, LLM cannot override
    max_wallet_credit_inr: int = 200
    max_discount_percent: int = 35
    max_free_items_per_complaint: int = 2

    # Rate Limiting
    max_requests_per_minute: int = 30
    max_image_uploads_per_session: int = 3

    # Features
    hallucination_check_sampling_rate: float = 0.5
    enable_cost_tracking: bool = True
    prompt_version: str = "v1"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()