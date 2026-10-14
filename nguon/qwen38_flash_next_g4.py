# Nguồn của notebook "Qwen3.8-Flash-Next G4" (Colab Pro, máy G4 = RTX PRO 6000 Blackwell 96 GB). Sửa ở đây rồi chạy:
# py -3 -X utf8 tao_notebook.py qwen38_flash_next_g4.py
# Mỗi "# %%" là một ô. Notebook Qwen3.8-27B G4 là notebook riêng.
# Không có ô 9b kiểm chất lượng (bản này đã đạt 31/31 mục ở lần chạy 13/10/2026).
#@bo chung/o9b_chat_luong.py

# %% [markdown]
# # Qwen3.8-Flash-Next (180B, 6B chạy mỗi token) trên Colab G4: endpoint chuẩn OpenAI
#
# **Run All** → dò máy (phải là G4, đo RAM và đĩa) → chọn bản nén theo RAM → cài vLLM vào môi trường riêng → tải model (133–186 GB) → chạy model, bảng n-gram để ở RAM máy → tự kiểm (trả lời, nghĩ, gọi tool, đọc ảnh, tốc độ) → mở gateway và tunnel → in khung **ENDPOINT SẴN SÀNG**.
#
# | Bước | Việc cần làm |
# |---|---|
# | 1 | Runtime → Change runtime type → **G4** (bật **High-RAM** nếu có lựa chọn) |
# | 2 | **Run All**, chờ khung `ENDPOINT SẴN SÀNG` ở ô 9 (lần đầu khoảng 20–30 phút: cài vLLM, tải model, nạp bảng n-gram vào RAM) |
# | 3 | Chép `Base URL`, `API key`, `Model` vào app hay thư viện nói chuẩn OpenAI. Bấm Stop ô 10 không tắt endpoint; tắt hẳn: chạy `tat_het()` |
#
# **Vì sao chạy được trên một thẻ 96 GB:** model có 180 tỷ tham số nhưng 51 tỷ trong đó là bảng tra n-gram, mỗi token chỉ đọc vài dòng của bảng. vLLM 0.31.0 tự để bảng này ở RAM máy và tra qua PCIe, còn GPU giữ phần thân (chuyên gia MoE nén NVFP4). Mỗi token chỉ chạy 6 tỷ tham số nên nhanh hơn Qwen3.8-27B dù tổng lớn hơn nhiều lần.
#
# **Bản nén** (ô 2, `MODEL = "auto"` dùng `nvidia` khi đủ RAM và đĩa):
#
# | Bản | Tải về | RAM cần cho bảng | Ghi chú |
# |---|---|---|---|
# | `nvidia` | 133 GB | khoảng 50 GB | bản chính chủ NVIDIA, bảng n-gram nén FP8 kèm hệ số; vLLM 0.31.0 có đường riêng cho kiểu nén này. NVIDIA chỉ thử trên B200/B300, chưa ai báo chạy trên RTX PRO 6000 |
# | `primitive` | 186 GB | khoảng 100 GB | bảng n-gram BF16; nhóm làm bản này đã chạy trên một thẻ RTX PRO 6000 bằng ảnh Docker vLLM riêng (họ đo: 74 token/giây một người dùng, 484 token/giây 32 người). Config của bản này khai bảng là FP8 trong khi tệp là BF16: với vLLM 0.31.0 có thể hỏng lúc nạp, chưa ai thử |
#
# **Giấy phép:** model gốc dùng Qwen Community License 1.0 (không phải Apache), bản NVIDIA thêm NVIDIA Open Model License. Đọc trước khi dùng cho việc thương mại: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
#
# **Sơ đồ:** `127.0.0.1:8787 gateway → 127.0.0.1:8000 vLLM`. Notebook này mặc định **mở tunnel** (`TUNNEL = True` ở ô 2) để app ở máy nhà gọi vào. FAQ của Colab cấm "web service offerings not related to interactive compute" và "connecting to remote proxies"; tunnel có thuộc diện đó không thì Colab không nói rõ. Không muốn mở: `TUNNEL = False`.
#
# **Secret** (biểu tượng chìa khóa bên trái, không bắt buộc):
#
# | Secret | Tác dụng | Không đặt thì |
# |---|---|---|
# | `GATEWAY_TOKEN` | API key cố định | mỗi phiên sinh khóa mới, in ở khung |
# | `CF_TUNNEL_TOKEN` + `PUBLIC_BASE_URL` | tunnel có tên của Cloudflare, địa chỉ cố định | Quick Tunnel, địa chỉ `trycloudflare.com` đổi mỗi phiên |
# | `HF_TOKEN` | tải nhanh hơn, ít bị giới hạn lượt | tải ẩn danh |

# %%
# 1) Dò máy: nền tảng, GPU, RAM, đĩa, mạng. Notebook này chỉ chạy trên G4 (Blackwell, compute capability 12.0).
import base64, hashlib, importlib.util, io, json, os, re, secrets, shutil, site, subprocess, sys, tarfile, time
import urllib.error, urllib.request
from pathlib import Path

NOTEBOOK_VERSION = "2026-10-13"

ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
ON_COLAB = not ON_KAGGLE and importlib.util.find_spec("google.colab") is not None
CHAY_NEN = ON_KAGGLE and os.environ["KAGGLE_KERNEL_RUN_TYPE"] != "Interactive"

WORK = Path("/kaggle/working" if ON_KAGGLE else "/content" if ON_COLAB else "work").resolve()
SCRATCH = (Path("/tmp") if ON_KAGGLE else WORK) / "qwen38"
MODEL_DIR, BIN_DIR, LOG_DIR = SCRATCH / "models", SCRATCH / "bin", WORK / "qwen38_logs"
for _p in (MODEL_DIR, BIN_DIR, LOG_DIR):
    _p.mkdir(parents=True, exist_ok=True)


def nvidia_smi(*args):
    exe = shutil.which("nvidia-smi") or next(
        (p for p in ("/opt/bin/nvidia-smi", "/usr/bin/nvidia-smi", "/usr/local/nvidia/bin/nvidia-smi") if Path(p).exists()), None)
    if not exe:
        return ""
    r = subprocess.run([exe, *args], text=True, capture_output=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def gpu_csv(fields):
    out = nvidia_smi(f"--query-gpu={fields}", "--format=csv,noheader,nounits")
    return [[x.strip() for x in line.split(",")] for line in out.splitlines() if line.strip()]


def ram_gib():
    """(tổng, còn dùng được) GiB, đọc /proc/meminfo."""
    m = dict(re.findall(r"^(\w+):\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.M))
    return int(m["MemTotal"]) / 2**20, int(m["MemAvailable"]) / 2**20


GPUS = [{"index": int(i), "name": n, "gib": int(m) / 1024, "cc": cc, "driver": dv}
        for i, n, m, cc, dv in gpu_csv("index,name,memory.total,compute_cap,driver_version")]
if not GPUS or not GPUS[0]["cc"].startswith("12.") or GPUS[0]["gib"] < 90:
    raise SystemExit(
        f"Notebook này dành cho G4 (RTX PRO 6000 Blackwell 96 GB, compute capability 12.0). Máy này: "
        f"{[(g['name'], round(g['gib']), g['cc']) for g in GPUS] or 'không có GPU'}.\n"
        "  Colab: Runtime -> Change runtime type -> G4.")

try:
    urllib.request.urlopen("https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next", timeout=20).read(64)
except urllib.error.HTTPError:
    pass
except Exception as e:
    raise SystemExit(f"Chưa có mạng ({type(e).__name__}: {e}).")

RAM_TONG, RAM_TRONG = ram_gib()
DIA_TRONG = shutil.disk_usage(SCRATCH).free / 1e9
print(json.dumps({
    "notebook": NOTEBOOK_VERSION, "kaggle": ON_KAGGLE, "colab": ON_COLAB,
    "python": sys.version.split()[0], "cpu": os.cpu_count(),
    "gpu": [f"{g['name']} {g['gib']:.1f} GiB, cc {g['cc']}, driver {g['driver']}" for g in GPUS],
    "ram_gib": f"{RAM_TRONG:.0f} trống / {RAM_TONG:.0f}", "dia_trong_gb": round(DIA_TRONG, 1),
}, ensure_ascii=False, indent=1))

# %%
# 2) CẤU HÌNH: chỉ cần sửa ô này.
MODEL = "auto"           # "auto" = "nvidia" khi đủ RAM và đĩa | "nvidia" | "primitive" (bảng ở ô chữ đầu)
SPEC = "mtp"             # đoán trước bằng đầu MTP có sẵn trong model | "tat"
SPEC_N = 3               # số token đoán mỗi lượt (Qwen ghi 3 trong hướng dẫn)
CTX = "auto"             # "auto" = 131072; 262144 cùng MTP không vừa 96 GB (primitive-ai đo: KV cần 7,57 GiB, còn 5,95)
KV = "auto"              # bộ nhớ ngữ cảnh: "auto" (bf16, cách đã có người chạy trên chip này) | "fp8" (vLLM 0.31.0 mới thêm, chưa ai thử ở đây)
VISION = True            # nạp phần đọc ảnh
THINKING = True          # mặc định có nghĩ; từng lượt gọi vẫn tự bật tắt được
MAX_SEQS = 4             # số lượt hỏi chạy song song tối đa
GPU_UTIL = 0.92          # phần VRAM vLLM được giữ (primitive-ai chạy 0.92: dùng 88,8 GiB)
VLLM_THEM = []           # cờ vLLM thêm. Thử ["--moe-backend", "marlin"]: primitive-ai đo một người dùng nhanh hơn 10% (93,2 so với
                         # 84,6 token/giây, không MTP), 32 người chậm hơn 1–3%
TUNNEL = True            # mở tunnel để gọi endpoint từ máy khác (lưu ý FAQ Colab in ở ô 9
                         # khi tắt) | "auto": Colab không mở | False
KEEP_ALIVE_HOURS = 4     # ô 10 giữ phiên bấy nhiêu giờ rồi tự tắt (G4 tốn đơn vị tính toán nhanh)

VLLM_VERSION = "0.31.0"  # 05/10/2026: có Qwen3.8-Flash-Next (Qwen4Exp), bảng n-gram tự để ở RAM máy, sửa MTP + backend MoE (#54788)
VLLM_SHA256 = "39132b433967f0962d03d8a5f633b9bee6b32708f566c7d5637b79f8d0ef3433"     # bánh xe x86_64, PyPI 13/10/2026
VLLM_PIN_DATE = "2026-10-13T00:00:00Z"
VLLM_CHO_GIAY = 3600     # chờ nạp tối đa (đọc khoảng 90 GB lên GPU và đến 95 GB bảng vào RAM)
# Nhân FlashInfer dựng sẵn cho Blackwell (SM120), cùng bản 0.7.0.post1 mà vLLM 0.31.0 ghim; địa chỉ và mã băm đọc từ
# https://flashinfer.ai/whl/cu130/ ngày 13/10/2026 (gói vỏ 5 KB + gói nhân 187 MB). Thiếu thì vLLM tự dịch nhân lúc nạp.
_FI = "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.0.post1/flashinfer_jit_cache"
FI_CACHE = [(f"{_FI}-0.7.0.post1+cu130-cp39-abi3-manylinux_2_28_x86_64.whl",
             "9630eeb32396adb89ce8ffa8465fedd14549eb2e2ad3aff182d9e69be4a12075"),
            (f"{_FI}_sm120f-0.7.0.post1+cu130-cp39-abi3-manylinux_2_28_x86_64.whl",
             "1c8b5a3614fdc9ad4f0e26502c5438657e28d667d165b05ff0383ca8d208568f")]
ENGINE_PORT = 8000       # vLLM, chỉ nghe 127.0.0.1
PORT_GW = 8787           # gateway, cổng DUY NHẤT đưa ra tunnel

# (repo, GB tải về, GiB RAM trống cần cho bảng n-gram + phần còn lại của tiến trình). Cỡ đọc từ API Hugging Face 13/10/2026.
# Cả hai giữ đầu MTP và phần đọc ảnh; không có code chạy kèm (vLLM có sẵn kiến trúc này, không bật trust-remote-code).
MODELS = {
    "nvidia": ("nvidia/Qwen3.8-Flash-Next-NVFP4", 132.7, 62),             # bảng FP8 khoảng 47 GiB (modelopt MIXED_PRECISION)
    "primitive": ("primitive-ai/Qwen3.8-Flash-Next-NVFP4", 186.4, 110),   # bảng BF16 95,4 GB
}
DRAFTS = {}              # model này đoán trước bằng đầu MTP của chính nó
ALIAS = "qwen3.8-flash-next"
if MODEL == "auto":
    _r, _gb, _ram = MODELS["nvidia"]
    MODEL = "nvidia" if RAM_TRONG >= _ram and DIA_TRONG >= _gb + 8 else None
    if MODEL is None:
        raise SystemExit(
            f"Máy này còn {RAM_TRONG:.0f} GiB RAM và {DIA_TRONG:.0f} GB đĩa trống. Cần ít nhất "
            f"{MODELS['nvidia'][2]} GiB RAM (giữ bảng n-gram) và {MODELS['nvidia'][1] + 8:.0f} GB đĩa (tải model), bản nvidia.\n"
            "  Colab: Runtime -> Change runtime type -> G4, bật High-RAM nếu có; hoặc dùng notebook 'Qwen3.8-27B G4'.")
MODEL_REPO, MODEL_GB, _RAM_CAN = MODELS[MODEL]
if RAM_TRONG < _RAM_CAN:
    raise SystemExit(f"Bản '{MODEL}' cần khoảng {_RAM_CAN} GiB RAM trống, máy này còn {RAM_TRONG:.0f}. Đặt MODEL = \"auto\".")
SPEC_REPO = None
SPEC_SO = int(SPEC_N)
CTX_SIZE = 131072 if CTX == "auto" else int(CTX)
print(json.dumps({"model": MODEL_REPO, "doan_truoc": f"{SPEC} {SPEC_SO if SPEC != 'tat' else ''}".strip(), "ngu_canh": CTX_SIZE,
                  "kv": KV, "doc_anh": VISION, "nghi_mac_dinh": THINKING, "vllm": VLLM_VERSION}, ensure_ascii=False, indent=1))

#@chen chung/o3_khoa.py

#@chen chung/o4_vllm.py

#@chen chung/o6_vllm.py

#@chen chung/o7_tu_kiem.py

#@chen chung/o8_gateway.py
