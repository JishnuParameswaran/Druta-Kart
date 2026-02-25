"""
Pytest configuration — adds backend/ to sys.path so tests can import
backend modules directly (e.g. `from nlp.emotion_analyzer import ...`).
"""
import sys
import os

# Resolve the backend/ directory relative to this conftest file
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, os.path.abspath(BACKEND_DIR))
