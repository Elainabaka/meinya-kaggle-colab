# Ô 6 dùng chung cho notebook chạy vLLM (chèn bằng "#@chen chung/o6_vllm.py"). Cần từ ô 2: SPEC, SPEC_SO, DRAFTS, KV, CTX_SIZE,
# VISION, THINKING, MAX_SEQS, GPU_UTIL, VLLM_VERSION, ENGINE_PORT, ALIAS, VLLM_THEM (cờ riêng của từng loại chip); từ ô 4–5: VPY,
# model_path, spec_path. Không bắt buộc: VLLM_ENV_THEM (biến môi trường riêng của chip), VLLM_CHO_GIAY (giờ chờ nạp, mặc định 2400).
# Đặt cho các ô 7–10: ENGINE_PROC, ENGINE_TEN, ENGINE_LOG, DOC_ANH, MTP_TOKENS, SPEC_TEN, DEM_MTP.

# %%
# 6) Chạy vLLM (chỉ nghe 127.0.0.1) và chờ nạp xong. Hỏng thì lùi dần: bộ đoán riêng (DSpark/DFlash2) -> MTP -> không đoán trước;
#    KV fp8 -> bf16;
#    hết bộ nhớ thì nửa ngữ cảnh. Lần đầu mất vài phút để dịch đồ thị (torch.compile, CUDA graph).
VLLM_ENV = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_CACHE_ROOT=str(SCRATCH / "vllm-cache"),
                TORCHINDUCTOR_CACHE_DIR=str(SCRATCH / "inductor-cache"), VLLM_LOGGING_LEVEL="INFO",
                # ninja của môi trường vLLM và nvcc của máy, cho phần dịch nhân lúc nạp (xem ô 4)
                PATH=":".join([str(VPY.parent), *[d for d in ("/usr/local/cuda/bin",) if Path(d).is_dir()], os.environ.get("PATH", "")]),
                **globals().get("VLLM_ENV_THEM", {}))


def spec_json(spec):
    if spec in DRAFTS:
        return json.dumps({"model": str(spec_path), "method": DRAFTS[spec][1], "num_speculative_tokens": SPEC_SO})
    if spec == "mtp":
        return json.dumps({"method": "mtp", "num_speculative_tokens": SPEC_SO if SPEC == "mtp" else 3})
    return None


def lenh_vllm(spec, ctx, kv):
    a = [str(VPY), "-m", "vllm.entrypoints.openai.api_server", "--model", str(model_path), "--served-model-name", ALIAS,
         "--host", "127.0.0.1", "--port", str(ENGINE_PORT), "--api-key", API_KEY,
         "--max-model-len", str(ctx), "--max-num-seqs", str(MAX_SEQS), "--gpu-memory-utilization", str(GPU_UTIL),
         "--kv-cache-dtype", kv, "--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
         "--disable-uvicorn-access-log", *VLLM_THEM]
    if spec_json(spec):
        a += ["--speculative-config", spec_json(spec)]
    if not VISION:
        a += ["--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    if not THINKING:
        a += ["--default-chat-template-kwargs", json.dumps({"enable_thinking": False})]
    return a


def cho_song(proc, timeout_s):
    t0, cuoi = time.time(), None
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False, f"tiến trình thoát với mã {proc.returncode}"
        try:
            r = requests.get(f"http://127.0.0.1:{ENGINE_PORT}/health", timeout=3)
            if r.status_code == 200:
                return True, None
            cuoi = r.status_code
        except Exception as e:
            cuoi = type(e).__name__
        time.sleep(5)
    return False, f"quá {timeout_s} giây chưa lên ({cuoi})"


def DEM_MTP():
    """(số token đoán trúng, số token đã đoán) cộng dồn, đọc từ /metrics của vLLM."""
    try:
        t = requests.get(f"http://127.0.0.1:{ENGINE_PORT}/metrics", timeout=5).text
    except Exception:
        return None
    lay = lambda ten: sum(float(m) for m in re.findall(rf"^vllm:{ten}(?:{{[^}}]*}})? ([0-9.e+]+)$", t, re.M))
    return lay("spec_decode_num_accepted_tokens_total"), lay("spec_decode_num_draft_tokens_total")


if globals().get("ENGINE_PROC") is not None and ENGINE_PROC.poll() is None:     # chạy lại ô này: tắt bản cũ trước
    ENGINE_PROC.terminate()
    ENGINE_PROC.wait(timeout=120)

_specs = ["tat"] if SPEC == "tat" else list(dict.fromkeys([SPEC, "mtp", "tat"]))     # đã tắt thì không tự bật lại
_kvs = [KV] + (["auto"] if KV != "auto" else [])
_si = _ki = 0
_ctx = CTX_SIZE
ENGINE_TEN, ENGINE_LOG = f"vLLM {VLLM_VERSION}", LOG_DIR / "vllm.log"
for _lan in range(6):
    SPEC_DUNG, KV_DUNG = _specs[_si], _kvs[_ki]
    lenh = lenh_vllm(SPEC_DUNG, _ctx, KV_DUNG)
    print("Lệnh:", " ".join(a if a != API_KEY else "<khóa nội bộ>" for a in lenh))
    _f = open(ENGINE_LOG, "w")
    ENGINE_PROC = chay_nen(lenh, stdout=_f, stderr=subprocess.STDOUT, env=VLLM_ENV)
    t0 = time.time()
    ok, cuoi = cho_song(ENGINE_PROC, timeout_s=globals().get("VLLM_CHO_GIAY", 2400))
    if ok:
        CTX_SIZE = _ctx
        break
    if ENGINE_PROC.poll() is None:
        ENGINE_PROC.kill()
        ENGINE_PROC.wait(timeout=120)
    _f.close()
    _log = ENGINE_LOG.read_text(errors="ignore")
    _loi = [l for l in _log.splitlines() if re.search(r"Error|error:|out of memory|not supported|RESOURCE_EXHAUSTED", l)]
    print(f"  không nạp được (đoán trước {SPEC_DUNG}, ngữ cảnh {_ctx}, kv {KV_DUNG}): {cuoi}\n  " + "\n  ".join(l[:220] for l in _loi[-6:]))
    # Hết bộ nhớ: giữ cách đoán trước, giảm ngữ cảnh (về đúng số vLLM ước được nếu nó in ra, không thì một nửa).
    # Lỗi về kiểu KV: bỏ fp8. Lỗi khác: lùi một bậc đoán trước.
    _uoc = re.findall(r"estimated maximum model length is (\d+)", _log)
    if re.search(r"out of memory|No available memory for the cache blocks|larger than the maximum number of tokens that can be stored"
                 r"|KV cache is needed|RESOURCE_EXHAUSTED|Ran out of memory", _log, re.I) and min(
                     _ctx // 2, int(_uoc[-1]) if _uoc else _ctx) >= 16384:
        _ctx = int(_uoc[-1]) // 4096 * 4096 if _uoc and int(_uoc[-1]) < _ctx else _ctx // 2
    elif _ki + 1 < len(_kvs) and re.search(r"fp8|kv_cache_dtype|kv-cache-dtype", "\n".join(_loi), re.I):
        _ki += 1
    elif _si + 1 < len(_specs):
        _si += 1
    elif _ki + 1 < len(_kvs):
        _ki += 1
    else:
        break
if not ok:
    raise SystemExit("vLLM không lên. Cuối log:\n" + ENGINE_LOG.read_text(errors="ignore")[-5000:])
SPEC_TEN = {"dspark": "DSpark", "dflash2": "DFlash2", "mtp": "MTP"}.get(SPEC_DUNG)
DOC_ANH, MTP_TOKENS = VISION, (json.loads(spec_json(SPEC_DUNG))["num_speculative_tokens"] if SPEC_TEN else 0)
print(f"[OK] vLLM nạp xong sau {time.time() - t0:.0f} giây (PID {ENGINE_PROC.pid}, ngữ cảnh {CTX_SIZE}, kv {KV_DUNG}, "
      f"đoán trước {SPEC_TEN or 'tắt'}{f' {MTP_TOKENS} token' if SPEC_TEN else ''})")
for _dong in re.findall(r"[^\n]*(?:Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency|"
                        r"Graph capturing finished|init engine)[^\n]*", ENGINE_LOG.read_text(errors="ignore"))[:8]:
    print("  ", _dong.strip()[-170:])
for i, used, total in gpu_csv("index,memory.used,memory.total"):
    print(f"   GPU {i}: {used}/{total} MiB")
