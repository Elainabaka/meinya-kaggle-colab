# Ô 3 dùng chung cho mọi notebook endpoint (chèn bằng "#@chen chung/o3_khoa.py").
# Cần có từ ô 1: ON_KAGGLE, ON_COLAB. Các hàm hoi, hoi_anh đọc ENGINE_PORT, API_KEY, ALIAS lúc được gọi (ô 6 đặt).

# %%
# 3) Khóa bí mật (không bắt buộc), thư viện cho gateway, hàm dùng chung
def get_secret(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        if ON_KAGGLE:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name) or None
        if ON_COLAB:
            from google.colab import userdata
            return userdata.get(name) or None
    except Exception:
        return None
    return None


HF_TOKEN = get_secret("HF_TOKEN")
GW_TOKEN_CO_DINH = get_secret("GATEWAY_TOKEN")
CF_TOKEN = get_secret("CF_TUNNEL_TOKEN")
PUBLIC_BASE_URL = (get_secret("PUBLIC_BASE_URL") or "").rstrip("/")
API_KEY = "sk-" + secrets.token_hex(24)          # khóa nội bộ gateway -> máy chạy model, không ra ngoài
print("HF_TOKEN:", "có" if HF_TOKEN else "không (đủ cho repo công khai)")
print("GATEWAY_TOKEN:", "lấy từ Secret (cố định)" if GW_TOKEN_CO_DINH else "không có, sẽ sinh khóa mới cho phiên này")
print("CF_TUNNEL_TOKEN:", "có (tunnel có tên)" if CF_TOKEN else "không có (Quick Tunnel, địa chỉ đổi mỗi phiên)")

_thieu = [m for m in ("fastapi", "uvicorn", "httpx", "requests", "huggingface_hub") if importlib.util.find_spec(m) is None]
if _thieu:
    print("Cài thêm:", _thieu)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi>=0.116,<1", "uvicorn>=0.35,<1",
                    "httpx>=0.28,<1", "requests>=2.31", "huggingface_hub>=0.28"], check=True)
import requests

KHONG_NGHI = {"chat_template_kwargs": {"enable_thinking": False}}


def tai(url, dst, sha256=None):
    """Tải url về dst (ghi file tạm rồi đổi tên); có sha256 thì kiểm."""
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".part")
    h, n, t0 = hashlib.sha256(), 0, time.time()
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120) as r, \
            open(tmp, "wb") as f:
        while True:
            b = r.read(4 << 20)
            if not b:
                break
            f.write(b)
            h.update(b)
            n += len(b)
    if sha256 and h.hexdigest() != sha256:
        tmp.unlink()
        raise RuntimeError(f"Sai mã băm, không dùng file này: {dst.name}")
    tmp.replace(dst)
    print(f"  tải {dst.name}: {n / 2**20:.0f} MiB trong {time.time() - t0:.0f} giây"
          + (", sha256 khớp" if sha256 else ", KHÔNG có mã băm để kiểm"))
    return dst


def chay_nen(lenh, **kw):
    """subprocess.Popen cho tiến trình chạy lâu (máy chạy model, gateway, tunnel). Nhóm tiến trình riêng: bấm Stop một ô thì
    Jupyter gửi SIGINT cho cả nhóm của kernel (13/10 Colab G4: Stop ô 10 tắt luôn vLLM). PR_SET_PDEATHSIG: kernel chết
    (Restart session) thì tiến trình nhận SIGTERM, không sót lại giữ GPU và RAM."""
    import ctypes, signal
    prctl, sigterm = ctypes.CDLL("libc.so.6", use_errno=True).prctl, int(signal.SIGTERM)   # nạp sẵn ở tiến trình cha:
                                                                                          # trong con chỉ gọi hàm, không nạp gì

    def _theo_kernel():
        prctl(1, sigterm)                                                     # 1 = PR_SET_PDEATHSIG

    return subprocess.Popen(lenh, start_new_session=True, preexec_fn=_theo_kernel, **kw)


def hoi(messages, url=None, headers=None, timeout=900, **kw):
    """Một lượt chat không stream. Mặc định gọi thẳng máy chạy model trong máy (không qua gateway)."""
    r = requests.post(url or f"http://127.0.0.1:{ENGINE_PORT}/v1/chat/completions",
                      headers=headers or {"Authorization": f"Bearer {API_KEY}"},
                      json={"model": ALIAS, "messages": messages, **kw}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def anh_7391():
    """Ảnh PNG vẽ số 7391, dạng data URL, để thử phần đọc ảnh."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (448, 224), "white")
    try:
        font = ImageFont.load_default(size=110)
    except TypeError:
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((70, 50), "7391", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def hoi_anh(timeout=900):
    """Hỏi số trong ảnh 7391 qua API nội bộ, trả chữ model đáp."""
    j = hoi([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": anh_7391()}},
        {"type": "text", "text": "Trong ảnh có số nào? Chỉ trả lời bằng chữ số."}]}],
        timeout=timeout, max_tokens=32, temperature=0, **KHONG_NGHI)
    return (j["choices"][0]["message"].get("content") or "").strip()
