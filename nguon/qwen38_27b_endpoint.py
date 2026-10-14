# Nguồn của notebook "Qwen3.8-27B endpoint". Sửa ở đây rồi chạy: py -3 -X utf8 tao_notebook.py
# Mỗi "# %%" là một ô. Luật và số đo máy: AGENTS.md, "cấu hình chung hiện tại 2026.md".

# %% [markdown]
# # Qwen3.8-27B: endpoint chuẩn OpenAI trên Kaggle 2×T4
#
# **Run All** → dò máy → lấy `llama-server` dựng sẵn (không biên dịch) → tải model GGUF → chạy model → tự kiểm (trả lời, nghĩ, gọi tool, đọc ảnh, tốc độ) → mở gateway + tunnel → in khung **ENDPOINT SẴN SÀNG**.
#
# Colab Pro dùng notebook riêng cho từng loại máy: `Qwen3.8-27B A100`, `Qwen3.8-27B G4` (thư mục `Colab/`).
#
# | Bước | Việc cần làm |
# |---|---|
# | 1 | Kaggle: cột phải → Settings → Accelerator **GPU T4 x2**, Internet **On** |
# | 2 | **Run All**, chờ khung `ENDPOINT SẴN SÀNG` ở ô 9 (lần đầu khoảng 10 phút, phần lớn là tải model) |
# | 3 | Chép `Base URL`, `API key`, `Model` vào chương trình cần dùng. |
#
# **Endpoint mở đúng 4 đường** (mọi đường đều cần `Authorization: Bearer <API key>`): `GET /healthz`, `GET /capabilities`, `GET /v1/models`, `POST /v1/chat/completions`.
#
# **Sơ đồ:** `cloudflared → 127.0.0.1:8787 gateway → 127.0.0.1:8080 llama-server`. Chỉ gateway ra ngoài; `llama-server` chỉ nghe trong máy.
#
# **Muốn địa chỉ và khóa không đổi giữa các phiên** (Kaggle: Add-ons → Secrets; Colab: biểu tượng chìa khóa):
#
# | Secret | Tác dụng | Không đặt thì |
# |---|---|---|
# | `GATEWAY_TOKEN` | API key cố định (tự đặt một chuỗi dài, khó đoán) | mỗi phiên sinh khóa mới, in ở khung |
# | `CF_TUNNEL_TOKEN` + `PUBLIC_BASE_URL` | tunnel có tên của Cloudflare, địa chỉ cố định (cần một tên miền trên Cloudflare) | Quick Tunnel, địa chỉ `trycloudflare.com` đổi mỗi phiên |
# | `HF_TOKEN` | chỉ cần khi repo model đòi đăng nhập | tải ẩn danh |

# %%
# 1) Dò máy: nền tảng, GPU, đĩa, mạng. Ô này không đoán cấu hình; các ô sau rẽ nhánh theo số đo ở đây.
import base64, hashlib, importlib.util, io, json, os, re, secrets, shutil, site, subprocess, sys, tarfile, time
import urllib.error, urllib.request
from pathlib import Path

NOTEBOOK_VERSION = "2026-10-13"

# Kiểm Kaggle TRƯỚC: máy Kaggle dựng từ image Colab nên cũng có biến COLAB_* và gói google-colab.
ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
ON_COLAB = not ON_KAGGLE and importlib.util.find_spec("google.colab") is not None
CHAY_NEN = ON_KAGGLE and os.environ["KAGGLE_KERNEL_RUN_TYPE"] != "Interactive"   # Save & Run All hoặc đẩy bằng CLI

WORK = Path("/kaggle/working" if ON_KAGGLE else "/content" if ON_COLAB else "work").resolve()
# File nặng (model, llama.cpp) để ngoài WORK: /kaggle/working chỉ 19,5 GiB và bị lưu thành Output.
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


GPUS = [{"index": int(i), "name": n, "gib": int(m) / 1024} for i, n, m in gpu_csv("index,name,memory.total")]
if not GPUS:
    raise SystemExit(
        "Không thấy GPU NVIDIA.\n"
        "  Kaggle: cột phải -> Settings -> Accelerator -> GPU T4 x2, rồi Run All lại.\n"
        "  Colab Pro: Runtime -> Change runtime type -> A100 hoặc G4.")

try:
    urllib.request.urlopen("https://huggingface.co/api/models/Qwen/Qwen3.8-27B", timeout=20).read(64)
except urllib.error.HTTPError:
    pass                                    # có trả lời HTTP là có mạng
except Exception as e:
    raise SystemExit(f"Chưa có mạng ({type(e).__name__}: {e}).\n"
                     "  Kaggle: Settings -> Internet -> On (tài khoản cần xác minh số điện thoại).")

print(json.dumps({
    "notebook": NOTEBOOK_VERSION, "kaggle": ON_KAGGLE, "colab": ON_COLAB, "chay_nen": CHAY_NEN,
    "python": sys.version.split()[0], "cpu": os.cpu_count(),
    "gpu": [f"{g['name']} {g['gib']:.1f} GiB" for g in GPUS],
    "dia_trong_gib": {str(SCRATCH): round(shutil.disk_usage(SCRATCH).free / 2**30, 1),
                      str(WORK): round(shutil.disk_usage(WORK).free / 2**30, 1)},
}, ensure_ascii=False, indent=1))

# %%
# 2) CẤU HÌNH: chỉ cần sửa ô này. Để "auto" thì notebook tự chọn theo VRAM đo được ở ô 1.
MODEL = "goc"            # "goc": Qwen3.8-27B chính gốc (Apache-2.0) | "swift15": Swift-1.5 (giấy phép riêng của tác giả)
QUANT = "auto"           # "auto" hoặc tên mức nén trong bảng MODELS: "Q4_K_M", "Q5_K_M", "Q6_K"...
CTX = "auto"             # "auto" hoặc số token ngữ cảnh: 32768, 65536, 131072, 262144
KV = "auto"              # nén bộ nhớ ngữ cảnh: "auto" (= q8_0) | "f16" | "q8_0" | "q4_0"
VISION = True            # nạp phần đọc ảnh (thêm khoảng 0,9 GiB VRAM)
THINKING = True          # mặc định có nghĩ; từng lượt gọi vẫn tự bật tắt được (xem khung ENDPOINT)
MTP = True               # đoán trước vài token bằng đầu MTP rồi kiểm (kết quả không đổi, chỉ nhanh hơn; thêm khoảng 2 GiB VRAM)
MTP_N = 3                # số token đoán trước mỗi bước
SPLIT = "auto"           # cách chia model cho 2 GPU: "auto" (thử "tensor", hỏng thì "layer") | "tensor" | "layer"
TUNNEL = "auto"          # "auto": Kaggle mở tunnel, Colab không mở (xem ô 9) | True | False
KEEP_ALIVE_HOURS = 4     # ô 10 giữ phiên bấy nhiêu giờ rồi tự tắt; thời gian chạy trừ vào hạn mức GPU
KIEM_CHAT_LUONG = False   # True: ô 9b tự kiểm chất lượng trước ô 10 (khoảng 5–10 phút; app gọi lúc đó sẽ chậm hơn)
LLAMA_TAG = "b11429"     # bản llama.cpp dựng sẵn (b11429 = bản ổn định v0.6.0, 05/10/2026)

ENGINE_PORT = 8080       # llama-server, chỉ nghe 127.0.0.1
PORT_GW = 8787           # gateway, cổng DUY NHẤT đưa ra tunnel

# Cỡ file tính bằng GB (10^9 byte), đọc từ API Hugging Face ngày 13/10/2026.
MODELS = {
    "goc": {
        "repo": "unsloth/Qwen3.8-27B-GGUF", "alias": "qwen3.8-27b",
        "file": "Qwen3.8-27B-UD-{q}.gguf", "mmproj": "mmproj-F16.gguf",
        "quants": {"Q4_K_M": 16.46, "Q5_K_M": 19.77, "Q6_K": 21.98, "Q8_K_L": 28.05},
    },
    "swift15": {
        "repo": "ukisai/Swift-1.5-Qwen3.8-27B-GGUF", "alias": "swift-1.5-qwen3.8-27b",
        "file": "Swift-1.5-Qwen3.8-27B-{q}.gguf", "mmproj": "mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf",
        "quants": {"Q4_K_M": 17.44, "Q5_K_M": 20.92, "Q6_K": 23.86, "Q8_0": 29.05},
    },
}
MMPROJ_GB = 0.93
# Đầu MTP lượng tử Q4_0 tách riêng của Unsloth (từ bản gốc; Swift-1.5 giữ nguyên đầu MTP của bản gốc theo model card).
# Đo 13/10 trên 2×T4, bản gốc Q5: code 9,8 -> 14,9 token/giây; đầu MTP nằm sẵn trong file chính chỉ được 11,6.
MTP_FILE = ("unsloth/Qwen3.8-27B-GGUF", "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", 1.37)
AUTO_MAX = "Q4_K_M"      # "auto" không chọn mức nặng hơn mức này. Đo 13/10 trên 2×T4: Q4 30 token/giây, Q5 25 (code)

# --- tự chọn mức nén và ngữ cảnh theo VRAM ---
_m = MODELS[MODEL]
ALIAS = _m["alias"]
GIB = 1e9 / 2**30
# Qwen3.8-27B chỉ có 16/64 lớp full attention giữ KV: f16 tốn 64 KiB mỗi token; q8_0 và q4_0 nhỏ hơn theo số bit.
KV_KIB = {"f16": 64.0, "q8_0": 34.0, "q4_0": 18.0}
VRAM_DUNG_DUOC = sum(g["gib"] - 0.75 for g in GPUS)      # chừa khoảng 0,75 GiB mỗi card cho CUDA


def can_gib(quant, ctx, kv):
    """Ước lượng tổng VRAM cần: trọng số + KV + bộ đệm tính toán + MTP (đo 13/10 ở ngữ cảnh 32K: khoảng 2,1 GiB)
    + phần đọc ảnh. Phần đọc ảnh (file mmproj + bộ đệm khi gặp ảnh, ước 1 GiB) nằm hết trên GPU0 nên tính hai lần:
    13/10 Swift Q5 ngữ cảnh 131K chia "tensor" dùng 14.717/15.360 MiB ở GPU0 và chết ở lượt đọc ảnh đầu tiên."""
    anh = (MMPROJ_GB * GIB + 1.0) * (2 if len(GPUS) > 1 else 1) if VISION else 0
    return _m["quants"][quant] * GIB + ctx * KV_KIB[kv] / 2**20 + 1.5 + (2.1 if MTP else 0) + anh


def chon_cau_hinh():
    kv = "q8_0" if KV == "auto" else KV
    theo_co = sorted(_m["quants"], key=_m["quants"].get, reverse=True)
    # Ưu tiên ngữ cảnh rộng nhất còn vừa, trong đó lấy mức nén nặng nhất (không quá AUTO_MAX).
    for ctx in ([int(CTX)] if CTX != "auto" else (262144, 131072, 65536, 32768, 16384)):     # 262144 = tối đa của model
        for q in ([QUANT] if QUANT != "auto" else [q for q in theo_co if _m["quants"][q] <= _m["quants"][AUTO_MAX]]):
            if can_gib(q, ctx, kv) <= VRAM_DUNG_DUOC:
                return q, ctx, kv, True
    # Không nằm trọn trong VRAM: lấy mức đã chọn (hoặc nhẹ nhất), để llama.cpp tự đẩy phần thừa sang CPU (chậm).
    return (QUANT if QUANT != "auto" else theo_co[-1]), (int(CTX) if CTX != "auto" else 16384), kv, False


QUANT_CHON, CTX_SIZE, KV_TYPE, VUA_TRON = chon_cau_hinh()
MODEL_FILE = _m["file"].format(q=QUANT_CHON)
MODEL_GB = _m["quants"][QUANT_CHON]
MODEL_REPO, MMPROJ = _m["repo"], (_m["mmproj"], MMPROJ_GB)       # ô 5–6 dùng chung đọc các biến này
LLAMA_MAU = ["--temp", "1.0", "--top-p", "0.95", "--top-k", "20"]     # mặc định theo model card (chế độ có nghĩ)
print(json.dumps({
    "model": f"{_m['repo']} / {MODEL_FILE}", "alias": ALIAS, "co_file_gb": MODEL_GB,
    "ngu_canh": CTX_SIZE, "kv": KV_TYPE, "doc_anh": VISION, "nghi_mac_dinh": THINKING, "mtp": MTP_N if MTP else 0,
    "vram_dung_duoc_gib": round(VRAM_DUNG_DUOC, 1), "vram_uoc_can_gib": round(can_gib(QUANT_CHON, CTX_SIZE, KV_TYPE), 1),
    "nam_tron_trong_gpu": VUA_TRON,
}, ensure_ascii=False, indent=1))
if not VUA_TRON:
    print("CẢNH BÁO: model không nằm trọn trong VRAM, một phần chạy bằng CPU nên sẽ chậm. "
          "Kaggle: chọn GPU T4 x2. Colab Pro: chọn A100 hoặc G4.")

#@chen chung/o3_khoa.py

#@chen chung/o4_llama.py

#@chen chung/o5_llama.py

#@chen chung/o6_llama.py

#@chen chung/o7_tu_kiem.py

#@chen chung/o8_gateway.py
