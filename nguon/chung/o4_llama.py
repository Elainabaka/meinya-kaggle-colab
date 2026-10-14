# Ô 4 dùng chung cho notebook chạy llama.cpp (chèn bằng "#@chen chung/o4_llama.py").
# Cần từ ô 1–3: LLAMA_TAG, SCRATCH, BIN_DIR, GPUS, tai. Đặt cho ô 5–6: server_bin, LLAMA_DIR, LLAMA_ENV, LLAMA_HELP.

# %%
# 4) llama.cpp: tải bản dựng sẵn chính thức (CUDA), kiểm mã băm, kiểm nó thấy đủ GPU. Không biên dịch.
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download"
LLAMA_SHA256 = {   # trường digest của API GitHub, đọc ngày 13/10/2026. Đổi LLAMA_TAG sang bản khác thì không có mã để kiểm.
    "llama-b11429-bin-ubuntu-cuda-12.8-x64.tar.gz": "b13c64b9224d3c89d9993945d70cc21177ad00178c21b8d34237e3bc139834f1",
    "cudart-llama-b11429-bin-ubuntu-cuda-12.8-x64.tar.gz": "f4ab593e31507320b121f2cb21791f4c72420a2483d3df8bf2ca1a97d04e6845",
    "llama-b11429-bin-ubuntu-cuda-13.4-x64.tar.gz": "8082b7eaa74a714c9fecca19128f751c8e32da763ee8096b8ad1e824da7621d3",
    "cudart-llama-b11429-bin-ubuntu-cuda-13.4-x64.tar.gz": "93d18648d815b2bd624d83d82f653e1db97afb478f02064305fe3cf570040a6d",
}


def giai_nen(tar_path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as t:
        t.extractall(dest, filter="data") if sys.version_info >= (3, 12) else t.extractall(dest)
    Path(tar_path).unlink()


def thu_muc_cuda():
    ds = ["/usr/local/cuda/lib64", "/usr/local/cuda/targets/x86_64-linux/lib", "/usr/local/nvidia/lib64", "/usr/lib64-nvidia"]
    for sp in site.getsitepackages():
        ds += [str(p) for p in Path(sp, "nvidia").glob("*/lib")]
    return [d for d in ds if Path(d).is_dir()]


def moi_truong(goc):
    """Biến môi trường để chạy llama-server: thêm mọi thư mục có .so dưới `goc` và thư viện CUDA của máy."""
    so_dirs = sorted({str(p.parent) for p in Path(goc).rglob("*.so*")})
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(d for d in [*so_dirs, *thu_muc_cuda(), env.get("LD_LIBRARY_PATH", "")] if d)
    # Bản dựng sẵn b11429 (CUDA 12.8) chỉ có mã máy cho sm_86, sm_89, sm_120a; T4 (sm_75) và A100 (sm_80) dịch từ PTX
    # lúc chạy (đo 13/10: lượt hỏi đầu mất khoảng 1 phút). Nới bộ đệm dịch để lần nạp lại trong phiên không dịch lại.
    env.setdefault("CUDA_CACHE_MAXSIZE", str(4 << 30))
    return env


def thu(server, goc):
    """Trả (số thiết bị CUDA mà llama-server thấy, chữ nó in ra)."""
    try:
        v = subprocess.run([str(server), "--version"], text=True, capture_output=True, timeout=120, env=moi_truong(goc))
        d = subprocess.run([str(server), "--list-devices"], text=True, capture_output=True, timeout=120, env=moi_truong(goc))
    except Exception as e:
        return 0, repr(e)
    out = (d.stdout or "") + (d.stderr or "")
    return len(re.findall(r"^\s*CUDA\d+:", out, re.M)), ((v.stdout or "") + (v.stderr or "")).strip() + "\n" + out.strip()


def co_cudart(major):
    return any(list(Path(d).glob(f"libcudart.so.{major}*")) for d in thu_muc_cuda())


t0 = time.time()
server_bin = LLAMA_DIR = None
# Kaggle đo 13/10/2026 dùng CUDA 12.8; Colab dùng CUDA 13.0. Thử bản khớp thư viện có sẵn trước.
for cuda in (["12.8", "13.4"] if (co_cudart(12) or not co_cudart(13)) else ["13.4", "12.8"]):
    goc = BIN_DIR / f"{LLAMA_TAG}-cuda{cuda}"
    if not any(goc.rglob("llama-server")):
        ten = f"llama-{LLAMA_TAG}-bin-ubuntu-cuda-{cuda}-x64.tar.gz"
        giai_nen(tai(f"{LLAMA_URL}/{LLAMA_TAG}/{ten}", SCRATCH / ten, LLAMA_SHA256.get(ten)), goc)
    sv = next(iter(goc.rglob("llama-server")), None)
    if sv is None:
        continue
    os.chmod(sv, 0o755)
    n_cuda, chu = thu(sv, goc)
    if n_cuda == 0:      # máy thiếu thư viện CUDA runtime đúng bản: lấy thêm gói cudart chính thức của cùng bản
        print(f"  bản CUDA {cuda} chưa thấy GPU, tải thêm thư viện CUDA đi kèm...")
        ten = f"cudart-llama-{LLAMA_TAG}-bin-ubuntu-cuda-{cuda}-x64.tar.gz"
        giai_nen(tai(f"{LLAMA_URL}/{LLAMA_TAG}/{ten}", SCRATCH / ten, LLAMA_SHA256.get(ten)), goc / "cudart")
        n_cuda, chu = thu(sv, goc)
    print(f"llama.cpp {LLAMA_TAG} (CUDA {cuda}): thấy {n_cuda}/{len(GPUS)} GPU")
    if n_cuda >= len(GPUS):
        server_bin, LLAMA_DIR = sv, goc
        break
if server_bin is None:
    raise SystemExit("llama-server dựng sẵn không thấy GPU trên máy này. Chữ nó in ra:\n" + chu[-2000:] +
                     "\nThử đổi LLAMA_TAG ở ô 2 sang bản mới hơn (https://github.com/ggml-org/llama.cpp/releases).")
LLAMA_ENV = moi_truong(LLAMA_DIR)
LLAMA_HELP = subprocess.run([str(server_bin), "--help"], text=True, capture_output=True, env=LLAMA_ENV).stdout
print(chu.strip()[:700])
print(f"[OK] llama-server sẵn sàng sau {time.time() - t0:.0f} giây: {server_bin}")
