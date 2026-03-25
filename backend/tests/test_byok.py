"""
BYOK (Bring Your Own Key) Tests — Two Levels

Level 1 — Unit tests
  Verify that every Groq / ChatGroq call in the pipeline receives the
  caller-supplied groq_api_key, and falls back to the server key when
  no key is provided.

Level 2 — Endpoint tests
  Verify that the FastAPI /chat and /voice endpoints accept groq_api_key
  and forward it all the way to _process_message.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure backend/ is on sys.path
# ---------------------------------------------------------------------------

_BACKEND = str(Path(__file__).parent.parent)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# ---------------------------------------------------------------------------
# Fake env vars — must be set BEFORE config.py is imported so Settings()
# validates without real API keys.
# ---------------------------------------------------------------------------

os.environ.setdefault("GROQ_API_KEY",             "gsk_fake_groq_key_for_tests")
os.environ.setdefault("SARVAM_API_KEY",           "fake_sarvam_key_for_tests")
os.environ.setdefault("SUPABASE_URL",             "https://fake.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY","fake_service_role_key_for_tests")

# ---------------------------------------------------------------------------
# sys.modules stubs — idempotent (safe to run standalone or with test_agents)
# ---------------------------------------------------------------------------

def _stub(name: str, **attrs) -> types.ModuleType:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
    else:
        # Update any missing attrs on the existing stub
        for k, v in attrs.items():
            if not hasattr(sys.modules[name], k):
                setattr(sys.modules[name], k, v)
    return sys.modules[name]


class _HumanMessage:
    type = "human"; role = "user"
    def __init__(self, content, **kw): self.content = content

class _AIMessage:
    type = "ai"; role = "assistant"
    def __init__(self, content, **kw): self.content = content

class _SystemMessage:
    type = "system"; role = "system"
    def __init__(self, content, **kw): self.content = content

def _tool_decorator(fn=None, **kw):
    def _wrap(f):
        f.name = f.__name__; f.invoke = lambda args, **kw2: f(**args); return f
    return _wrap(fn) if fn is not None else _wrap

_stub("langchain_core")
_lc_msgs = _stub("langchain_core.messages")
_lc_msgs.HumanMessage  = _HumanMessage
_lc_msgs.AIMessage     = _AIMessage
_lc_msgs.SystemMessage = _SystemMessage
_lc_tools = _stub("langchain_core.tools")
_lc_tools.tool = _tool_decorator

_stub("langgraph")
_stub("langgraph.graph", END="__end__", StateGraph=MagicMock())

_structlog = _stub("structlog")
_structlog.get_logger = MagicMock(return_value=MagicMock())
_structlog.configure  = MagicMock()
_sp = _stub("structlog.processors")
_sp.TimeStamper  = MagicMock(return_value=MagicMock())
_sp.JSONRenderer = MagicMock(return_value=MagicMock())
_structlog.processors = _sp
_ss = _stub("structlog.stdlib")
_ss.add_log_level  = MagicMock()
_ss.BoundLogger    = MagicMock()
_ss.LoggerFactory  = MagicMock(return_value=MagicMock())
_structlog.stdlib  = _ss

_stub("lingua")
_stub("transformers", pipeline=MagicMock())
_stub("torch")
_stub("sentence_transformers", SentenceTransformer=MagicMock())
_stub("supabase", create_client=MagicMock(return_value=MagicMock()))
_exifread = _stub("exifread")
_exifread.process_file = MagicMock(return_value={})
_stub("sarvamai")

_prom = _stub("prometheus_client")
_prom.Counter         = MagicMock(return_value=MagicMock())
_prom.Histogram       = MagicMock(return_value=MagicMock())
_prom.Gauge           = MagicMock(return_value=MagicMock())
_prom.generate_latest = MagicMock(return_value=b"")
_prom.CONTENT_TYPE_LATEST = "text/plain"

_slowapi = _stub("slowapi")
_slowapi.Limiter = MagicMock()
_slowapi._rate_limit_exceeded_handler = MagicMock()
_slowapi_err = _stub("slowapi.errors")
_slowapi_err.RateLimitExceeded = Exception
_slowapi_util = _stub("slowapi.util")
_slowapi_util.get_remote_address = MagicMock()

# ---------------------------------------------------------------------------
# Default stub Groq / ChatGroq classes (will be swapped in tests)
# ---------------------------------------------------------------------------

def _make_groq_resp(text: str) -> MagicMock:
    """Build a fake groq chat completion response."""
    msg = MagicMock(); msg.content = text
    choice = MagicMock(); choice.message = msg
    resp = MagicMock(); resp.choices = [choice]
    return resp


class _DefaultGroqClient:
    def __init__(self, **kw): pass
    class chat:
        class completions:
            @staticmethod
            def create(**kw):
                return _make_groq_resp("en-IN")
    class audio:
        class transcriptions:
            @staticmethod
            def create(**kw): return "hello world"


class _DefaultChatGroq:
    def __init__(self, **kw): pass
    def invoke(self, messages):
        return _AIMessage("wallet_credit")


_stub("groq",          Groq=_DefaultGroqClient)
_stub("langchain_groq", ChatGroq=_DefaultChatGroq)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BYOK_KEY   = "gsk_byok_test_key_abc123"
_SERVER_KEY = "gsk_server_demo_key_xyz"


class _TrackingGroq:
    """Records the api_key it was instantiated with."""
    last_key: str | None = None

    def __init__(self, api_key: str = "", **kw):
        _TrackingGroq.last_key = api_key

    class chat:
        class completions:
            @staticmethod
            def create(**kw):
                return _make_groq_resp("en-IN")

    class audio:
        class transcriptions:
            @staticmethod
            def create(**kw): return "hello world"


class _TrackingChatGroq:
    """Records the api_key it was instantiated with."""
    last_key: str | None = None

    def __init__(self, api_key: str = "", **kw):
        _TrackingChatGroq.last_key = api_key

    def invoke(self, messages):
        return _AIMessage("wallet_credit")


def _install_tracking_groq():
    """Replace stub Groq with spy; return restore callable."""
    orig = sys.modules["groq"].Groq
    sys.modules["groq"].Groq = _TrackingGroq
    _TrackingGroq.last_key = None
    return lambda: setattr(sys.modules["groq"], "Groq", orig)


def _install_tracking_chatgroq():
    """Replace stub ChatGroq with spy; return restore callable."""
    orig = sys.modules["langchain_groq"].ChatGroq
    sys.modules["langchain_groq"].ChatGroq = _TrackingChatGroq
    _TrackingChatGroq.last_key = None
    return lambda: setattr(sys.modules["langchain_groq"], "ChatGroq", orig)


def _base_state(**overrides) -> dict:
    return {
        "user_id": "u1", "session_id": "s1",
        "language": "en-IN", "emotion": "neutral",
        "messages": [_HumanMessage("my item is damaged")],
        "intent": "complaint", "complaint_type": "damaged",
        "order_id": "ORD-001", "image_path": None,
        "image_validation_result": None, "vision_reason": None,
        "resolved": False, "resolution_type": None, "csat_score": None,
        "offer_given": None, "fraud_flagged": False,
        "human_handoff": False, "handoff_proof": None,
        "tools_called": [], "agent_used": None, "rag_used": False,
        "response": "", "hallucination_flagged": False,
        "groq_api_key": None,
        **overrides,
    }


# ===========================================================================
# Level 1 — Unit tests: key propagation through each layer
# ===========================================================================

class TestLanguageDetectorBYOK:
    """nlp/language_detector.py — detect_language()"""

    def test_byok_key_passed_to_groq(self):
        restore = _install_tracking_groq()
        try:
            from nlp.language_detector import detect_language
            detect_language("hello", groq_api_key=_BYOK_KEY)
            assert _TrackingGroq.last_key == _BYOK_KEY
        finally:
            restore()

    def test_fallback_to_env_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", _SERVER_KEY)
        restore = _install_tracking_groq()
        try:
            from nlp.language_detector import detect_language
            detect_language("hello")
            assert _TrackingGroq.last_key == _SERVER_KEY
        finally:
            restore()

    def test_byok_overrides_env_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", _SERVER_KEY)
        restore = _install_tracking_groq()
        try:
            from nlp.language_detector import detect_language
            detect_language("hello", groq_api_key=_BYOK_KEY)
            assert _TrackingGroq.last_key == _BYOK_KEY, "BYOK key must override env key"
        finally:
            restore()


class TestSTTProcessorBYOK:
    """multimodal/stt_processor.py — transcribe_audio()"""

    def test_byok_key_passed_to_groq(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        restore = _install_tracking_groq()
        try:
            from multimodal.stt_processor import transcribe_audio
            transcribe_audio(str(audio_file), groq_api_key=_BYOK_KEY)
            assert _TrackingGroq.last_key == _BYOK_KEY
        finally:
            restore()

    def test_fallback_to_settings_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.settings.groq_api_key", _SERVER_KEY)
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        restore = _install_tracking_groq()
        try:
            from multimodal.stt_processor import transcribe_audio
            transcribe_audio(str(audio_file))
            assert _TrackingGroq.last_key == _SERVER_KEY
        finally:
            restore()


class TestImageAnalyzerBYOK:
    """multimodal/image_analyzer.py — analyze_image()"""

    def test_byok_key_passed_to_groq(self, tmp_path):
        img = tmp_path / "img.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
        restore = _install_tracking_groq()
        try:
            from multimodal.image_analyzer import analyze_image
            analyze_image(str(img), groq_api_key=_BYOK_KEY)
            assert _TrackingGroq.last_key == _BYOK_KEY
        finally:
            restore()

    def test_fallback_to_settings_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.settings.groq_api_key", _SERVER_KEY)
        img = tmp_path / "img.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
        restore = _install_tracking_groq()
        try:
            from multimodal.image_analyzer import analyze_image
            analyze_image(str(img))
            assert _TrackingGroq.last_key == _SERVER_KEY
        finally:
            restore()


class TestHallucinationGuardBYOK:
    """agents/hallucination_guard.py — check_response()"""

    def test_byok_key_reaches_chatgroq(self, monkeypatch):
        # Force sampling so the guard actually calls the LLM
        monkeypatch.setattr("agents.hallucination_guard.random.random", lambda: 0.0)
        import json
        safe_json = json.dumps({"safe": True, "issues": [], "corrected_response": None})
        class _SafeChatGroq(_TrackingChatGroq):
            def invoke(self, messages): return _AIMessage(safe_json)
        restore = _install_tracking_chatgroq()
        sys.modules["langchain_groq"].ChatGroq = _SafeChatGroq
        try:
            from agents.hallucination_guard import check_response
            check_response("All good.", groq_api_key=_BYOK_KEY)
            assert _TrackingChatGroq.last_key == _BYOK_KEY
        finally:
            restore()

    def test_fallback_to_settings_key(self, monkeypatch):
        monkeypatch.setattr("agents.hallucination_guard.random.random", lambda: 0.0)
        monkeypatch.setattr("config.settings.groq_api_key", _SERVER_KEY)
        import json
        safe_json = json.dumps({"safe": True, "issues": [], "corrected_response": None})
        class _SafeChatGroq(_TrackingChatGroq):
            def invoke(self, messages): return _AIMessage(safe_json)
        restore = _install_tracking_chatgroq()
        sys.modules["langchain_groq"].ChatGroq = _SafeChatGroq
        try:
            from agents.hallucination_guard import check_response
            check_response("All good.")
            assert _TrackingChatGroq.last_key == _SERVER_KEY
        finally:
            restore()


class TestImageValidationAgentBYOK:
    """agents/image_validation_agent.py — groq_api_key from state"""

    def test_byok_key_from_state_reaches_groq(self, tmp_path):
        img = tmp_path / "damage.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
        restore = _install_tracking_groq()
        try:
            from agents.image_validation_agent import run
            state = _base_state(image_path=str(img), groq_api_key=_BYOK_KEY)
            run(state)
            assert _TrackingGroq.last_key == _BYOK_KEY
        finally:
            restore()

    def test_no_key_in_state_uses_settings(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.settings.groq_api_key", _SERVER_KEY)
        img = tmp_path / "damage.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
        restore = _install_tracking_groq()
        try:
            from agents.image_validation_agent import run
            state = _base_state(image_path=str(img), groq_api_key=None)
            run(state)
            assert _TrackingGroq.last_key == _SERVER_KEY
        finally:
            restore()


class TestComplaintAgentBYOK:
    """agents/complaint_agent.py — groq_api_key from state"""

    def test_byok_key_from_state_reaches_chatgroq(self, monkeypatch):
        # Skip image validation and RAG to isolate LLM call
        monkeypatch.setattr(
            "agents.complaint_agent._query_policy", lambda q: "policy text"
        )
        restore = _install_tracking_chatgroq()
        try:
            from agents.complaint_agent import run
            state = _base_state(
                complaint_type="missing",   # forced resolution, no image needed
                groq_api_key=_BYOK_KEY,
            )
            run(state)
            assert _TrackingChatGroq.last_key == _BYOK_KEY
        finally:
            restore()

    def test_no_key_falls_back_to_settings(self, monkeypatch):
        monkeypatch.setattr("config.settings.groq_api_key", _SERVER_KEY)
        monkeypatch.setattr(
            "agents.complaint_agent._query_policy", lambda q: "policy text"
        )
        restore = _install_tracking_chatgroq()
        try:
            from agents.complaint_agent import run
            state = _base_state(complaint_type="missing", groq_api_key=None)
            run(state)
            assert _TrackingChatGroq.last_key == _SERVER_KEY
        finally:
            restore()


class TestOrderAgentBYOK:
    """agents/order_agent.py — groq_api_key from state"""

    def test_byok_key_from_state_reaches_chatgroq(self):
        restore = _install_tracking_chatgroq()
        try:
            from agents.order_agent import run
            # No order_id → falls through to _build_general_order_response (LLM call)
            state = _base_state(order_id=None, groq_api_key=_BYOK_KEY)
            run(state)
            assert _TrackingChatGroq.last_key == _BYOK_KEY
        finally:
            restore()


class TestDispatchAgentBYOK:
    """agents/dispatch_agent.py — groq_api_key from state"""

    def test_byok_key_from_state_reaches_chatgroq(self, monkeypatch):
        # Stub out tools so only the LLM call matters
        monkeypatch.setattr(
            "agents.dispatch_agent._detect_issues", lambda t: ["late_delivery"]
        )
        restore = _install_tracking_chatgroq()
        try:
            from agents.dispatch_agent import run
            state = _base_state(groq_api_key=_BYOK_KEY)
            run(state)
            assert _TrackingChatGroq.last_key == _BYOK_KEY
        finally:
            restore()


class TestRetentionAgentBYOK:
    """agents/retention_agent.py — groq_api_key from state"""

    def test_byok_key_from_state_reaches_chatgroq(self, monkeypatch):
        mock_offer = {
            "status": "ok",
            "offer_type": "wallet_credit",
            "offer_value": 50.0,
            "offer_description": "₹50 wallet credit",
            "risk_level": "medium",
        }
        monkeypatch.setattr(
            "agents.retention_agent.offer_generator_tool",
            MagicMock(invoke=MagicMock(return_value=mock_offer)),
            raising=False,
        )
        restore = _install_tracking_chatgroq()
        try:
            from agents.retention_agent import run
            state = _base_state(
                resolved=True,
                response="Your complaint has been resolved.",
                groq_api_key=_BYOK_KEY,
            )
            run(state)
            assert _TrackingChatGroq.last_key == _BYOK_KEY
        finally:
            restore()


class TestSupervisorBYOK:
    """agents/supervisor.py — groq_api_key injected into AgentState"""

    def test_groq_api_key_in_initial_state(self, monkeypatch):
        """run() must put groq_api_key into the AgentState passed to graph.invoke()."""
        captured_state: dict = {}

        def _fake_invoke(state):
            captured_state.update(state)
            return {**state, "response": "ok", "intent": "general", "resolved": True,
                    "tools_called": [], "hallucination_flagged": False,
                    "fraud_flagged": False, "offer_given": None,
                    "human_handoff": False, "csat_score": None,
                    "handoff_proof": None, "agent_used": "general",
                    "rag_used": False, "image_validation_result": None}

        from agents import supervisor as sup
        original_graph = sup.graph
        sup.graph = MagicMock(invoke=_fake_invoke)
        try:
            sup.run(
                user_id="u1", session_id="s1",
                message="hello", groq_api_key=_BYOK_KEY,
            )
            assert captured_state.get("groq_api_key") == _BYOK_KEY
        finally:
            sup.graph = original_graph

    def test_no_key_defaults_to_none_in_state(self):
        """When groq_api_key is omitted, initial_state should have None (settings fallback)."""
        captured_state: dict = {}

        def _fake_invoke(state):
            captured_state.update(state)
            return {**state, "response": "ok", "intent": "general", "resolved": True,
                    "tools_called": [], "hallucination_flagged": False,
                    "fraud_flagged": False, "offer_given": None,
                    "human_handoff": False, "csat_score": None,
                    "handoff_proof": None, "agent_used": "general",
                    "rag_used": False, "image_validation_result": None}

        from agents import supervisor as sup
        original_graph = sup.graph
        sup.graph = MagicMock(invoke=_fake_invoke)
        try:
            sup.run(user_id="u1", session_id="s1", message="hello")
            assert captured_state.get("groq_api_key") is None
        finally:
            sup.graph = original_graph


# ===========================================================================
# Level 2 — Endpoint tests: FastAPI /chat and /voice
# ===========================================================================

@pytest.fixture(scope="module")
def test_client():
    """Create a FastAPI TestClient with all heavy dependencies patched out."""
    from fastapi.testclient import TestClient
    import main

    # Minimal valid ChatResponse dict
    _FAKE_RESULT = {
        "response": "Test response",
        "intent": "general",
        "emotion": "neutral",
        "language": "en-IN",
        "resolved": True,
        "offer_given": None,
        "fraud_flagged": False,
        "hallucination_flagged": False,
        "tools_called": [],
        "human_handoff": False,
        "csat_score": None,
        "handoff_proof": None,
        "agent_used": "general",
        "rag_used": False,
        "image_validation_result": None,
    }

    original_pm = main._process_message
    main._CAPTURED_KEY = None  # shared slot

    def _spy_process_message(*args, **kwargs):
        main._CAPTURED_KEY = kwargs.get("groq_api_key")
        return (_FAKE_RESULT.copy(), 42.0)

    main._process_message = _spy_process_message
    client = TestClient(main.app, raise_server_exceptions=True)
    yield client
    main._process_message = original_pm


class TestChatEndpointBYOK:
    """/chat endpoint — groq_api_key in request body"""

    def test_byok_key_forwarded_to_process_message(self, test_client):
        import main
        resp = test_client.post("/chat", json={
            "user_id": "u1",
            "session_id": "s1",
            "message": "my order is late",
            "groq_api_key": _BYOK_KEY,
        })
        assert resp.status_code == 200
        assert main._CAPTURED_KEY == _BYOK_KEY

    def test_no_key_forwards_none(self, test_client):
        import main
        resp = test_client.post("/chat", json={
            "user_id": "u1",
            "session_id": "s1",
            "message": "where is my order",
        })
        assert resp.status_code == 200
        assert main._CAPTURED_KEY is None, "Absent key must forward as None (server fallback)"

    def test_response_shape_unchanged_with_byok(self, test_client):
        """BYOK must not change the response structure."""
        resp = test_client.post("/chat", json={
            "user_id": "u1",
            "session_id": "s1",
            "message": "hello",
            "groq_api_key": _BYOK_KEY,
        })
        assert resp.status_code == 200
        data = resp.json()
        for field in ("response", "intent", "emotion", "language", "resolved",
                      "fraud_flagged", "hallucination_flagged", "tools_called"):
            assert field in data, f"Missing field: {field}"


class TestVoiceEndpointBYOK:
    """/voice endpoint — groq_api_key as form field"""

    def test_byok_key_accepted_as_form_field(self, test_client, tmp_path):
        import main

        # Patch STT so it returns a transcript without hitting Groq
        original_ta = main._transcribe_audio
        main._transcribe_audio = lambda path, groq_api_key=None: "test transcript"

        audio = tmp_path / "clip.wav"
        audio.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
                          b"\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00"
                          b"\x02\x00\x10\x00data\x00\x00\x00\x00")

        try:
            resp = test_client.post(
                "/voice",
                data={"user_id": "u1", "session_id": "s1", "groq_api_key": _BYOK_KEY},
                files={"file": ("clip.wav", audio.read_bytes(), "audio/wav")},
            )
            assert resp.status_code == 200
            assert main._CAPTURED_KEY == _BYOK_KEY
        finally:
            main._transcribe_audio = original_ta

    def test_voice_no_key_forwards_none(self, test_client, tmp_path):
        import main

        original_ta = main._transcribe_audio
        main._transcribe_audio = lambda path, groq_api_key=None: "test transcript"

        audio = tmp_path / "clip.wav"
        audio.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
                          b"\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00"
                          b"\x02\x00\x10\x00data\x00\x00\x00\x00")

        try:
            resp = test_client.post(
                "/voice",
                data={"user_id": "u1", "session_id": "s1"},
                files={"file": ("clip.wav", audio.read_bytes(), "audio/wav")},
            )
            assert resp.status_code == 200
            assert main._CAPTURED_KEY is None
        finally:
            main._transcribe_audio = original_ta
