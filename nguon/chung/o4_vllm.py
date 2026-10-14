# Ô 4 và 5 dùng chung cho notebook chạy vLLM (chèn bằng "#@chen chung/o4_vllm.py").
# Cần từ ô 2: VLLM_VERSION, VLLM_SHA256, VLLM_PIN_DATE, MODEL_REPO, SPEC_REPO (None nếu không dùng bộ đoán riêng), SCRATCH,
# MODEL_DIR. Không bắt buộc: FI_CACHE = [(địa chỉ bánh xe, sha256), ...] là nhân FlashInfer dựng sẵn cho chip của notebook.

# %%
# 4) vLLM: cài vào môi trường Python riêng (không đụng PyTorch có sẵn của Colab), kiểm mã băm, kiểm thấy GPU
VENV = SCRATCH / "vllm-venv"
VPY = VENV / "bin" / "python"


def chay(cmd, **kw):
    """Chạy lệnh, in 3.000 ký tự cuối khi hỏng; trả mã thoát."""
    r = subprocess.run([str(c) for c in cmd], text=True, capture_output=True, **kw)
    if r.returncode:
        print(f"  lệnh hỏng (mã {r.returncode}): {' '.join(map(str, cmd))[:200]}\n" + ((r.stdout or "") + (r.stderr or ""))[-3000:])
    return r.returncode


def vllm_ok():
    r = subprocess.run([str(VPY), "-c", "import torch, vllm; print(vllm.__version__, torch.__version__, torch.version.cuda, "
                        "torch.cuda.is_available(), torch.cuda.get_device_capability() if torch.cuda.is_available() else None)"],
                       text=True, capture_output=True)
    return r.returncode == 0 and f"{VLLM_VERSION} " in r.stdout and " True " in r.stdout, (r.stdout + r.stderr).strip()[-1500:]


t0 = time.time()
if not shutil.which("uv"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
ok, chu = vllm_ok() if VPY.exists() else (False, "")
if not ok:
    shutil.rmtree(VENV, ignore_errors=True)
    if chay([UV, "venv", VENV, "--python", sys.executable, "-q"]):
        raise SystemExit("Không tạo được môi trường Python riêng cho vLLM.")
    # Lấy địa chỉ bánh xe từ PyPI rồi kiểm mã băm đã ghim ở ô 2.
    _ten = f"vllm-{VLLM_VERSION}-cp38-abi3-manylinux_2_28_x86_64.whl"
    _pypi = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/vllm/{VLLM_VERSION}/json", timeout=60))
    _url = next(f["url"] for f in _pypi["urls"] if f["filename"] == _ten)
    _whl = tai(_url, SCRATCH / _ten, VLLM_SHA256)
    # Thư viện đi kèm (PyTorch, FlashInfer...) lấy đúng bản PyPI có tới ngày ghim, để lần chạy sau ra cùng một bộ.
    if chay([UV, "pip", "install", "--python", VPY, "--exclude-newer", VLLM_PIN_DATE, _whl], env=dict(os.environ, UV_HTTP_TIMEOUT="300")):
        raise SystemExit("Cài vLLM hỏng (xem chữ in ở trên).")
    _whl.unlink(missing_ok=True)
    ok, chu = vllm_ok()
# Trên chip mới vLLM dịch nhân FlashInfer ngay lúc nạp và cần lệnh ninja (Colab G4 13/10: "No such file or directory: 'ninja'").
# Ô 6 đưa thư mục bin của môi trường này lên đầu PATH để tìm thấy nó.
if ok and not (VENV / "bin" / "ninja").exists() and chay([UV, "pip", "install", "--python", VPY, "--exclude-newer", VLLM_PIN_DATE, "ninja"]):
    raise SystemExit("Cài ninja hỏng (xem chữ in ở trên).")
# Nhân FlashInfer dựng sẵn cho đúng chip (FI_CACHE ở ô 2): có thì vLLM nạp thẳng, không phải dịch lúc nạp (cần nvcc, thêm vài
# phút mỗi phiên). Cài --no-deps vì gói vỏ kéo theo bản dựng sẵn của mọi loại chip. Hỏng thì bỏ qua, vLLM tự dịch như cũ.
_fi_dau = VENV / ".fi-cache"
if ok and globals().get("FI_CACHE") and not _fi_dau.exists():
    try:
        _whls = [tai(u, SCRATCH / u.rsplit("/", 1)[1], s) for u, s in FI_CACHE]
        if chay([UV, "pip", "install", "--python", VPY, "--no-deps", *_whls]) == 0:
            _fi_dau.write_text("ok")
        for _w in _whls:
            _w.unlink(missing_ok=True)
    except Exception as e:
        print(f"  (không cài được nhân FlashInfer dựng sẵn: {type(e).__name__}: {str(e)[:200]}; vLLM sẽ tự dịch lúc nạp)")
if not ok:
    raise SystemExit("vLLM cài xong nhưng không thấy GPU (thường là driver quá cũ cho CUDA 13). Chữ in ra:\n" + chu
                     + f"\nDriver máy này: {nvidia_smi('--query-gpu=driver_version', '--format=csv,noheader')}")
print(f"[OK] vLLM sẵn sàng sau {time.time() - t0:.0f} giây: {chu}")

# %%
# 5) Model và bộ đoán trước: tải từ Hugging Face về SCRATCH (đã có thì bỏ qua)
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
if CHAY_NEN:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import snapshot_download


def lay_repo(repo):
    dst = MODEL_DIR / repo.replace("/", "__")
    t0 = time.time()
    p = Path(snapshot_download(repo, local_dir=str(dst), token=HF_TOKEN or None, max_workers=16))
    gb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
    print(f"  {repo}: {gb:.2f} GB, {time.time() - t0:.0f} giây")
    return p


trong = shutil.disk_usage(MODEL_DIR).free / 1e9
if trong < MODEL_GB + 8:
    raise SystemExit(f"Đĩa {MODEL_DIR} còn {trong:.0f} GB, không đủ cho model khoảng {MODEL_GB:.0f} GB.")
model_path = lay_repo(MODEL_REPO)
spec_path = lay_repo(SPEC_REPO) if SPEC_REPO else None
print("[OK] Model:", model_path)
