"""Narrow OpenAI-compatible inference gateway. Tunnel THIS port (8787), never llama-server (8080) directly.

Security Invariants:
1. Strict allowlist: ONLY /healthz, /capabilities, /v1/models, /v1/chat/completions.
2. Distinct public GATEWAY_TOKEN vs internal VLLM_API_KEY.
3. Unknown routes return 404 (no catch-all proxy).
4. Unbuffered SSE streaming pass-through.
5. Strict request body limit (25 MB default).
6. Personal single-client concurrency guard (semaphore).
7. Zero internal credential or traceback leakage.
8. Request ID correlation logging without prompt content exposure.
9. Clean async lifespan management and shutdown.
"""
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import secrets
import time
from typing import AsyncGenerator

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

# Logging configuration (Request ID and metadata only, NO prompt/body contents)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("qwen_gateway")

# Configuration & secrets
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8080").rstrip("/")
GATEWAY_TOKEN = os.getenv("GATEWAY_TOKEN", "")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
MODEL = os.getenv("SERVED_MODEL_NAME", "qwen3.8-27b")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "131072"))
MULTIMODAL = os.getenv("TEXT_ONLY", "false").lower() not in {"1", "true", "yes"}
MTP_TOKENS = int(os.getenv("MTP_TOKENS", "3"))
RUN_ID = os.getenv("RUN_ID", str(int(time.time())))
MAX_BODY_BYTES = int(os.getenv("GATEWAY_MAX_BODY_BYTES", str(25 * 1024 * 1024))) # 25 MB
MAX_CONCURRENT_INFERENCE = int(os.getenv("GATEWAY_CONCURRENCY", "3"))
MAX_OUTPUT_TOKENS = int(os.getenv("GATEWAY_MAX_OUTPUT_TOKENS", "32768"))
NONSTREAM_WAIT_S = float(os.getenv("GATEWAY_NONSTREAM_WAIT_S", "60"))  # below Cloudflare's 100 s cut-off
NONSTREAM_PAD_S = 15.0

if not GATEWAY_TOKEN or not VLLM_API_KEY:
    raise RuntimeError("Both GATEWAY_TOKEN and VLLM_API_KEY environment variables are required")

# State & Lifespan
client_holder: dict[str, httpx.AsyncClient] = {}
concurrency_limit = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)

@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    timeout = httpx.Timeout(900.0, connect=10.0, read=900.0, write=30.0)
    client = httpx.AsyncClient(limits=limits, timeout=timeout)
    client_holder["client"] = client
    logger.info(f"Gateway initialized. Target upstream at {VLLM_URL}, Model: {MODEL}")
    try:
        yield
    finally:
        logger.info("Gateway shutting down, closing upstream connections...")
        await client.aclose()
        logger.info("Gateway shutdown complete.")

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

# CORS for browser clients (`npm run dev` on localhost calling the tunnel URL).
# Auth is NOT weakened: preflight OPTIONS carries no credentials, and every
# allowlisted route still requires the Bearer gateway token.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error handling {request.method} {request.url.path}: {type(exc).__name__}")
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal gateway error", "type": "gateway_error", "code": 500}},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": "http_error", "code": exc.status_code}},
    )

def auth(authorization: str | None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid authorization header")
    provided_token = authorization[7:].strip()
    if not secrets.compare_digest(provided_token, GATEWAY_TOKEN):
        raise HTTPException(401, "Unauthorized: invalid gateway token")

def internal_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {VLLM_API_KEY}",
        "Content-Type": "application/json",
    }

async def read_bounded_body(request: Request) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared = int(content_length)
        except ValueError:
            raise HTTPException(400, "Malformed Content-Length header")
        if declared > MAX_BODY_BYTES:
            raise HTTPException(413, f"Payload exceeds maximum allowed size ({MAX_BODY_BYTES} bytes)")

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BODY_BYTES:
            raise HTTPException(413, f"Payload exceeds maximum allowed size ({MAX_BODY_BYTES} bytes)")
    return bytes(body)

@app.get("/healthz")
async def healthz(authorization: str | None = Header(default=None)):
    auth(authorization)
    req_id = secrets.token_hex(4)
    client = client_holder.get("client")
    if not client:
        return JSONResponse({"ok": False, "run_id": RUN_ID, "error": "Gateway starting"}, status_code=503)

    try:
        r = await client.get(f"{VLLM_URL}/v1/models", headers=internal_headers(), timeout=8.0)
        is_ready = r.status_code == 200
        logger.info(f"[{req_id}] /healthz -> upstream {r.status_code} (ready={is_ready})")
        return JSONResponse(
            {"ok": is_ready, "run_id": RUN_ID, "model": MODEL},
            status_code=200 if is_ready else 503,
        )
    except Exception as e:
        logger.warning(f"[{req_id}] /healthz check failed: {type(e).__name__}")
        return JSONResponse(
            {"ok": False, "run_id": RUN_ID, "error": "Upstream unavailable"},
            status_code=503,
        )

@app.get("/capabilities")
async def capabilities(authorization: str | None = Header(default=None)):
    auth(authorization)
    return {
        "runId": RUN_ID,
        "model": MODEL,
        "maxModelLen": MAX_MODEL_LEN,
        "multimodal": MULTIMODAL,
        "toolCalling": True,
        "streamUsage": True,
        "reasoningLevels": ["low", "medium", "xhigh"],
        "defaultReasoning": "xhigh",
        "backend": {
            "vllm_tpu": os.getenv("VLLM_TPU_VERSION", "0.28.0"),
            "mtp_tokens": MTP_TOKENS,
        },
    }

@app.get("/v1/models")
async def models(request: Request):
    auth(request.headers.get("authorization"))
    req_id = secrets.token_hex(4)
    client = client_holder.get("client")
    if not client:
        raise HTTPException(503, "Gateway unready")

    try:
        r = await client.get(f"{VLLM_URL}/v1/models", headers=internal_headers(), timeout=10.0)
        logger.info(f"[{req_id}] GET /v1/models -> {r.status_code}")
        try:
            content = r.json() if r.content else {}
        except Exception:
            content = {"error": {"message": f"Upstream returned non-JSON (status {r.status_code})", "type": "upstream_error", "code": r.status_code}}
        return JSONResponse(status_code=r.status_code, content=content)
    except httpx.RequestError:
        raise HTTPException(503, "Upstream inference service unavailable")

@app.post("/v1/chat/completions")
async def chat(request: Request):
    auth(request.headers.get("authorization"))
    req_id = secrets.token_hex(4)
    t0 = time.time()

    body = await read_bounded_body(request)

    try:
        payload = json.loads(body)
    except Exception:
        raise HTTPException(400, "Malformed JSON request body")
    if not isinstance(payload, dict):
        raise HTTPException(400, "Request body must be a JSON object")

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "'messages' must be a non-empty array")
    payload["model"] = MODEL
    
    # Handle both max_tokens and max_completion_tokens
    max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens")
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            raise HTTPException(400, "'max_tokens' must be an integer")
        clamped_tokens = max(1, min(max_tokens, MAX_OUTPUT_TOKENS))
        payload["max_tokens"] = clamped_tokens
        if "max_completion_tokens" in payload:
            payload["max_completion_tokens"] = clamped_tokens

    stream_raw = payload.get("stream", False)
    stream = (
        stream_raw is True
        or stream_raw == 1
        or (isinstance(stream_raw, str) and stream_raw.lower() in ("1", "true"))
    )

    client = client_holder.get("client")
    if not client:
        raise HTTPException(503, "Gateway unready")

    body = json.dumps(payload).encode("utf-8")
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(413, f"Payload exceeds maximum allowed size ({MAX_BODY_BYTES} bytes)")

    logger.info(f"[{req_id}] POST /v1/chat/completions (stream={stream}, bytes={len(body)})")

    if not stream:
        try:
            await asyncio.wait_for(concurrency_limit.acquire(), timeout=30)
        except asyncio.TimeoutError:
            raise HTTPException(429, "Gateway busy: all inference slots occupied, retry shortly")

        async def call_upstream() -> tuple[int, dict]:
            try:
                r = await client.post(
                    f"{VLLM_URL}/v1/chat/completions",
                    content=body,
                    headers=internal_headers(),
                )
                logger.info(f"[{req_id}] Completed in {time.time() - t0:.2f}s (status={r.status_code})")
                try:
                    content = r.json() if r.content else {}
                except Exception:
                    content = {"error": {"message": f"Upstream returned non-JSON (status {r.status_code})", "type": "upstream_error", "code": r.status_code}}
                return r.status_code, content
            except httpx.RequestError:
                return 503, {"error": {"message": "Upstream inference service unavailable", "type": "http_error", "code": 503}}
            finally:
                concurrency_limit.release()

        task = asyncio.create_task(call_upstream())
        try:
            status, content = await asyncio.wait_for(asyncio.shield(task), timeout=NONSTREAM_WAIT_S)
            return JSONResponse(status_code=status, content=content)
        except asyncio.TimeoutError:
            pass

        # Cloudflare returns 524 when the origin sends nothing for 100 s. Past NONSTREAM_WAIT_S, send the
        # 200 header now and one space every 15 s (JSON allows leading whitespace), then the body.
        # Trade-off: an upstream error after this point arrives as an error JSON under status 200.
        logger.info(f"[{req_id}] Non-stream still running after {NONSTREAM_WAIT_S:.0f}s, keepalive padding on")

        async def padded() -> AsyncGenerator[bytes, None]:
            try:
                while not task.done():
                    done, _ = await asyncio.wait({task}, timeout=NONSTREAM_PAD_S)
                    if not done:
                        yield b" "
                yield json.dumps(task.result()[1]).encode("utf-8")
            finally:
                if not task.done():
                    task.cancel()
                    logger.info(f"[{req_id}] Downstream disconnected, aborting upstream call")

        return StreamingResponse(
            padded(),
            media_type="application/json",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def sse_generator() -> AsyncGenerator[bytes, None]:
        try:
            try:
                await asyncio.wait_for(concurrency_limit.acquire(), timeout=30)
            except asyncio.TimeoutError:
                err_payload = json.dumps({"error": {"message": "Gateway busy: all inference slots occupied, retry shortly", "type": "gateway_busy", "code": 429}})
                yield f"data: {err_payload}\n\ndata: [DONE]\n\n".encode("utf-8")
                return
            try:
                async with client.stream(
                    "POST",
                    f"{VLLM_URL}/v1/chat/completions",
                    content=body,
                    headers=internal_headers(),
                ) as r:
                    if r.status_code != 200:
                        error_bytes = await r.aread()
                        logger.warning(f"[{req_id}] Upstream error {r.status_code}")
                        try:
                            detail = json.loads(error_bytes.decode("utf-8")).get("error", {}).get(
                                "message", f"Upstream error {r.status_code}"
                            ) if error_bytes else f"Upstream error {r.status_code}"
                        except Exception:
                            detail = f"Upstream error {r.status_code}"
                        err_payload = json.dumps(
                            {"error": {"message": detail, "type": "upstream_error", "code": r.status_code}}
                        )
                        yield f"data: {err_payload}\n\ndata: [DONE]\n\n".encode("utf-8")
                        return

                    async for chunk in r.aiter_bytes():
                        if await request.is_disconnected():
                            logger.info(f"[{req_id}] Downstream disconnected, aborting upstream pull")
                            break
                        yield chunk
            finally:
                concurrency_limit.release()
        except httpx.RequestError as e:
            logger.error(f"[{req_id}] Upstream connection broken during streaming: {type(e).__name__}")
            err_payload = json.dumps({"error": {"message": "Upstream connection broken", "type": "upstream_disconnect"}})
            yield f"data: {err_payload}\n\ndata: [DONE]\n\n".encode("utf-8")
        finally:
            logger.info(f"[{req_id}] Stream closed after {time.time() - t0:.2f}s")

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
