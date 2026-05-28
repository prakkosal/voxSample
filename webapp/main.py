import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import Response, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

from vastai.serverless import Serverless
from vastai.serverless.client.endpoint import Endpoint
from vastai.serverless.remote import serialization

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

VAST_API_KEY = os.getenv("VASTAI_API_KEY")
VAST_ENDPOINT_ID = os.getenv("VASTAI_ENDPOINT_ID")
VAST_ENDPOINT_NAME = os.getenv("VASTAI_ENDPOINT_NAME")
REMOTE_FUNC = os.getenv("VASTAI_REMOTE_FUNC", "generate_speech")

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@gmail.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")
SESSION_SECRET = os.getenv("SESSION_SECRET") or secrets.token_urlsafe(32)

if not VAST_API_KEY:
    raise RuntimeError("VASTAI_API_KEY is not set in .env")
if not VAST_ENDPOINT_ID and not VAST_ENDPOINT_NAME:
    raise RuntimeError("Set VASTAI_ENDPOINT_ID or VASTAI_ENDPOINT_NAME in .env")


class VastState:
    serverless: Optional[Serverless] = None
    endpoint: Optional[Endpoint] = None


state = VastState()


async def _resolve_endpoint(client: Serverless) -> Endpoint:
    endpoints = await client.get_endpoints()
    if VAST_ENDPOINT_ID:
        wanted = int(VAST_ENDPOINT_ID)
        for e in endpoints:
            if e.data.id == wanted:
                return e
        raise RuntimeError(f"No endpoint with id {wanted} found on this account")
    for e in endpoints:
        if e.name == VAST_ENDPOINT_NAME:
            return e
    raise RuntimeError(f"No endpoint named {VAST_ENDPOINT_NAME!r} found")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.serverless = Serverless(api_key=VAST_API_KEY)
    await state.serverless.__aenter__()
    try:
        state.endpoint = await _resolve_endpoint(state.serverless)
        yield
    finally:
        await state.serverless.__aexit__(None, None, None)
        state.serverless = None
        state.endpoint = None


app = FastAPI(title="VoxCPM2 TTS Web", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, max_age=60 * 60 * 24 * 7)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def require_login(request: Request) -> str:
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="not logged in")
    return user


def html_require_login(request: Request):
    if not request.session.get("user"):
        return RedirectResponse(url="/login", status_code=303)
    return None


@app.get("/login")
async def login_page(request: Request):
    if request.session.get("user"):
        return RedirectResponse(url="/", status_code=303)
    return FileResponse(STATIC_DIR / "login.html")


@app.post("/login")
async def login_submit(request: Request, email: str = Form(...), password: str = Form(...)):
    ok = secrets.compare_digest(email, ADMIN_EMAIL) and secrets.compare_digest(password, ADMIN_PASSWORD)
    if not ok:
        return Response(
            content=(STATIC_DIR / "login.html").read_text().replace(
                "<!--ERROR-->", '<div class="error">Invalid email or password</div>'
            ),
            media_type="text/html",
            status_code=401,
        )
    request.session["user"] = email
    return RedirectResponse(url="/", status_code=303)


@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/")
async def index(request: Request):
    redirect = html_require_login(request)
    if redirect is not None:
        return redirect
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/tts")
async def tts(
    text: str = Form(...),
    voice_description: str = Form(""),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(25),
    reference_wav: Optional[UploadFile] = File(None),
    _user: str = Depends(require_login),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    if state.endpoint is None:
        raise HTTPException(status_code=503, detail="Vast.ai endpoint not initialized")

    reference_wav_hex = ""
    if reference_wav is not None:
        data = await reference_wav.read()
        if data:
            reference_wav_hex = data.hex()

    kwargs = {
        "text": text,
        "reference_wav_hex": reference_wav_hex,
        "voice_description": voice_description,
        "cfg_value": float(cfg_value),
        "inference_timesteps": int(inference_timesteps),
    }
    payload = {
        "kwargs": {k: serialization.serialize(v, "") for k, v in kwargs.items()}
    }
    route = f"/remote/{REMOTE_FUNC}"

    try:
        raw = await state.endpoint.request(route=route, payload=payload)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vast.ai call failed: {e}")

    try:
        inner = raw["response"]["result"]
    except (KeyError, TypeError):
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected worker response shape: {raw!r}"[:1000],
        )

    try:
        result = serialization.deserialize_unwrap_error(inner, "", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to decode response: {e}")

    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail=f"Unexpected result type: {type(result).__name__}")
    if result.get("status") != "success":
        raise HTTPException(status_code=500, detail=result.get("error", str(result)))

    audio_hex = result.get("audio")
    if not audio_hex:
        raise HTTPException(status_code=500, detail="response missing 'audio' field")

    audio_bytes = bytes.fromhex(audio_hex)
    headers = {
        "X-Sample-Rate": str(result.get("sample_rate", "")),
        "X-Duration-Seconds": str(result.get("duration_seconds", "")),
        "X-Request-Id": str(result.get("request_id", "")),
        "Content-Disposition": 'inline; filename="output.wav"',
    }
    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
