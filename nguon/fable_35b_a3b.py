# Nguồn của notebook "FableCoder 35B-A3B endpoint". Sửa ở đây rồi chạy: py -3 -X utf8 tao_notebook.py
# Mỗi "# %%" là một ô. Ô 3–10 là ô chung với notebook Qwen3.8-27B (nguon/chung/); ô 1–2 riêng.
# Luật và số đo máy: AGENTS.md, "cấu hình chung hiện tại 2026.md".

# %% [markdown]
# # FableCoder 35B-A3B: endpoint chuẩn OpenAI trên Kaggle 2×T4
#
# Model viết code `Achilles1089/fable-coder-35B-A3B` (tinh chỉnh từ Qwen3.6-35B-A3B, MoE: 35 tỷ tham số, mỗi token chạy khoảng 3 tỷ; Apache-2.0).
#
# **Run All** → dò máy → lấy `llama-server` dựng sẵn (không biên dịch) → tải model GGUF → chạy model → tự kiểm (trả lời, nghĩ, gọi tool, tốc độ) → mở gateway + tunnel → in khung **ENDPOINT SẴN SÀNG**.
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
# **Muốn địa chỉ và khóa không đổi giữa các phiên** (Kaggle: Add-ons → Secrets):
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
SCRATCH = (Path("/tmp") if ON_KAGGLE else WORK) / "fable"
MODEL_DIR, BIN_DIR, LOG_DIR = SCRATCH / "models", SCRATCH / "bin", WORK / "fable_logs"
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
        "  Kaggle: cột phải -> Settings -> Accelerator -> GPU T4 x2, rồi Run All lại.")

try:
    urllib.request.urlopen("https://huggingface.co/api/models/Achilles1089/fable-coder-35B-A3B-GGUF", timeout=20).read(64)
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
QUANT = "auto"           # "auto" hoặc tên mức nén trong bảng QUANTS: "Q4_K_M", "IQ4_XS", "Q5_K_M"
CTX = "auto"             # "auto" hoặc số token ngữ cảnh: 32768, 65536, 131072, 262144 (tối đa của model)
KV = "auto"              # nén bộ nhớ ngữ cảnh: "auto" (= q8_0) | "f16" | "q8_0" | "q4_0"
VISION = False           # True: đọc được ảnh (file mmproj của mradermacher, thêm khoảng 1 GiB VRAM ở GPU0). Để "auto" thì
                         # notebook hạ xuống mức nén IQ4_XS để giữ ngữ cảnh 262144 (đo 13/10: code 90,9 token/giây);
                         # muốn giữ Q4_K_M thì đặt QUANT = "Q4_K_M", ngữ cảnh còn 131072 (theo công thức ước, chưa chạy)
THINKING = True          # mặc định có nghĩ; từng lượt gọi vẫn tự bật tắt được (xem khung ENDPOINT)
MTP = True               # đoán trước vài token bằng lớp MTP nằm sẵn trong file model (kết quả không đổi, chỉ nhanh hơn)
MTP_N = 2                # số token đoán trước mỗi bước. Đo 13/10 trên 2×T4 (code / văn / nghĩ, token/giây): không MTP
                         # 75,0 / 75,3 / 74,9; MTP 2: 93,0 / 71,8 / 93,7; MTP 3: 91,3 / 64,6 / 94,1 -> MTP 2
SPLIT = "auto"           # cách chia model cho 2 GPU: "auto" (thử "tensor", hỏng thì "layer") | "tensor" | "layer"
                         # Đo 13/10, không MTP: "layer" 50,7 / 68,4 / 68,1; "tensor" 75,0 / 75,3 / 74,9
TUNNEL = "auto"          # "auto": Kaggle mở tunnel, Colab không mở (xem ô 9) | True | False
KEEP_ALIVE_HOURS = 4     # ô 10 giữ phiên bấy nhiêu giờ rồi tự tắt; thời gian chạy trừ vào hạn mức GPU
KIEM_CHAT_LUONG = False   # True: ô 9b tự kiểm chất lượng trước ô 10 (khoảng 5–10 phút; app gọi lúc đó sẽ chậm hơn)
LLAMA_TAG = "b11429"     # bản llama.cpp dựng sẵn (b11429 = bản ổn định v0.6.0, 05/10/2026)

ENGINE_PORT = 8080       # llama-server, chỉ nghe 127.0.0.1
PORT_GW = 8787           # gateway, cổng DUY NHẤT đưa ra tunnel
ALIAS = "fable-coder-35b-a3b"      # tên model khi gọi API

# (repo, file, cỡ GB = 10^9 byte), đọc từ API Hugging Face ngày 13/10/2026. Mọi file đều giữ lớp MTP (blk.40):
# Q4_K_M theo model card, IQ4_XS đọc phần đầu file 13/10.
QUANTS = {
    "Q4_K_M": ("Achilles1089/fable-coder-35B-A3B-GGUF", "fable-coder-35B-A3B-Q4_K_M.gguf", 21.71),
    "Q5_K_M": ("Achilles1089/fable-coder-35B-A3B-GGUF", "fable-coder-35B-A3B-Q5_K_M.gguf", 25.35),
    "IQ4_XS": ("mradermacher/fable-coder-35B-A3B-GGUF", "fable-coder-35B-A3B.IQ4_XS.gguf", 19.39),
}
MMPROJ = ("fable-coder-35B-A3B.mmproj-Q8_0.gguf", 0.61, "mradermacher/fable-coder-35B-A3B-GGUF")
MTP_FILE = None          # lớp MTP nằm sẵn trong file model, không tải file riêng
AUTO_MAX = "Q4_K_M"      # "auto" không chọn mức nặng hơn mức này
# Mức Qwen khuyên cho chế độ có nghĩ khi viết code (model card Qwen3.6-35B-A3B); từng lượt gọi vẫn tự đặt được.
LLAMA_MAU = ["--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0"]

# --- tự chọn mức nén và ngữ cảnh theo VRAM ---
GIB = 1e9 / 2**30
# KiB VRAM mỗi token ngữ cảnh. q8_0 đo 13/10 trên 2×T4 (chia "tensor", MTP 2): ngữ cảnh 32.768 dùng 2 × 11.011 MiB,
# 262.144 dùng 2 × 13.317 MiB, tức khoảng 20,5 KiB mỗi token. f16 và q4_0 suy theo số bit, chưa đo.
KV_KIB = {"f16": 38.6, "q8_0": 20.5, "q4_0": 10.9}
VRAM_DUNG_DUOC = sum(g["gib"] - 0.75 for g in GPUS)      # chừa khoảng 0,75 GiB mỗi card cho CUDA


def can_gib(quant, ctx, kv):
    """Ước lượng tổng VRAM cần: trọng số (đã gồm lớp MTP) + KV + bộ đệm tính toán + phần đọc ảnh.
    Phần đọc ảnh nằm hết trên GPU0 nên tính hai lần (bài học notebook Qwen3.8-27B, 13/10)."""
    anh = (MMPROJ[1] * GIB + 1.0) * (2 if len(GPUS) > 1 else 1) if VISION else 0
    return QUANTS[quant][2] * GIB + ctx * KV_KIB[kv] / 2**20 + 1.5 + (0.5 if MTP else 0) + anh


def chon_cau_hinh():
    kv = "q8_0" if KV == "auto" else KV
    theo_co = sorted(QUANTS, key=lambda q: QUANTS[q][2], reverse=True)
    # Ưu tiên ngữ cảnh rộng nhất còn vừa, trong đó lấy mức nén nặng nhất (không quá AUTO_MAX).
    for ctx in ([int(CTX)] if CTX != "auto" else (262144, 131072, 65536, 32768, 16384)):
        for q in ([QUANT] if QUANT != "auto" else [q for q in theo_co if QUANTS[q][2] <= QUANTS[AUTO_MAX][2]]):
            if can_gib(q, ctx, kv) <= VRAM_DUNG_DUOC:
                return q, ctx, kv, True
    # Không nằm trọn trong VRAM: lấy mức đã chọn (hoặc nhẹ nhất), để llama.cpp tự đẩy phần thừa sang CPU (chậm).
    return (QUANT if QUANT != "auto" else theo_co[-1]), (int(CTX) if CTX != "auto" else 16384), kv, False


QUANT_CHON, CTX_SIZE, KV_TYPE, VUA_TRON = chon_cau_hinh()
MODEL_REPO, MODEL_FILE, MODEL_GB = QUANTS[QUANT_CHON]       # ô 5–6 dùng chung đọc các biến này
print(json.dumps({
    "model": f"{MODEL_REPO} / {MODEL_FILE}", "alias": ALIAS, "co_file_gb": MODEL_GB,
    "ngu_canh": CTX_SIZE, "kv": KV_TYPE, "doc_anh": VISION, "nghi_mac_dinh": THINKING, "mtp": MTP_N if MTP else 0,
    "vram_dung_duoc_gib": round(VRAM_DUNG_DUOC, 1), "vram_uoc_can_gib": round(can_gib(QUANT_CHON, CTX_SIZE, KV_TYPE), 1),
    "nam_tron_trong_gpu": VUA_TRON,
}, ensure_ascii=False, indent=1))
if not VUA_TRON:
    print("CẢNH BÁO: model không nằm trọn trong VRAM, một phần chạy bằng CPU nên sẽ chậm. Kaggle: chọn GPU T4 x2.")

#@chen chung/o3_khoa.py

#@chen chung/o4_llama.py

#@chen chung/o5_llama.py

#@chen chung/o6_llama.py

#@chen chung/o7_tu_kiem.py

#@chen chung/o8_gateway.py
