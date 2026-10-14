# Nguồn của notebook "Qwen3.8-27B A100" (Colab Pro, máy A100 40 GB hoặc 80 GB). Sửa ở đây rồi chạy: py -3 -X utf8 tao_notebook.py
# Mỗi "# %%" là một ô. Bản G4 là notebook riêng (nguon/qwen38_27b_g4.py): hai kiến trúc chip khác nhau, cách nén khác nhau.

# %% [markdown]
# # Qwen3.8-27B trên Colab A100: endpoint chuẩn OpenAI, nhắm từ 100 token/giây
#
# **Run All** → dò máy (phải là A100) → cài vLLM vào môi trường riêng → tải model nén **INT4 W4A16** và bộ đoán trước **DSpark** → chạy model → tự kiểm (trả lời, nghĩ, gọi tool, đọc ảnh, tốc độ) → mở gateway → in khung **ENDPOINT SẴN SÀNG**.
#
# | Bước | Việc cần làm |
# |---|---|
# | 1 | Runtime → Change runtime type → **A100**. Máy G4 thì mở notebook `Qwen3.8-27B G4` |
# | 2 | **Run All**, chờ khung `ENDPOINT SẴN SÀNG` ở ô 9 (lần đầu khoảng 10–15 phút: cài vLLM, tải khoảng 24 GB, dịch đồ thị CUDA) |
# | 3 | Chép `Base URL`, `API key`, `Model` vào chương trình cần dùng |
#
# **Vì sao chọn INT4:** A100 (kiến trúc Ampere) không có lõi tính FP8 hay FP4 như Blackwell. Đường 4 bit nhanh nhất trên Ampere là trọng số INT4, tính bằng nhân **Marlin** của vLLM (giải nén ngay trong thanh ghi). Sinh mỗi token phải đọc hết trọng số một lần, nên trọng số nặng một phần tư thì nhanh gần gấp bốn. Bộ đoán trước DSpark (5 lớp, của Red Hat) đoán 7 token rồi model lớn kiểm một lượt; Red Hat đo trung bình được nhận khoảng 4,4 token mỗi lượt với code. Kết quả không đổi so với chạy thường, chỉ nhanh hơn.
#
# **Sơ đồ:** `127.0.0.1:8787 gateway → 127.0.0.1:8000 vLLM`. Mặc định **không mở tunnel** trên Colab (lý do ở ô 9); đặt `TUNNEL = True` ở ô 2 nếu anh muốn gọi từ ngoài vào.
#
# **Secret** (biểu tượng chìa khóa bên trái, không bắt buộc):
#
# | Secret | Tác dụng | Không đặt thì |
# |---|---|---|
# | `GATEWAY_TOKEN` | API key cố định | mỗi phiên sinh khóa mới, in ở khung |
# | `CF_TUNNEL_TOKEN` + `PUBLIC_BASE_URL` | tunnel có tên của Cloudflare, địa chỉ cố định | Quick Tunnel, địa chỉ `trycloudflare.com` đổi mỗi phiên |
# | `HF_TOKEN` | tải nhanh hơn, ít bị giới hạn lượt | tải ẩn danh |

# %%
# 1) Dò máy: nền tảng, GPU, đĩa, mạng. Notebook này chỉ chạy trên A100 (Ampere, compute capability 8.0).
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


GPUS = [{"index": int(i), "name": n, "gib": int(m) / 1024, "cc": cc, "driver": dv}
        for i, n, m, cc, dv in gpu_csv("index,name,memory.total,compute_cap,driver_version")]
if not GPUS or GPUS[0]["cc"] != "8.0":
    raise SystemExit(
        f"Notebook này dành cho A100 (Ampere, compute capability 8.0). Máy này: "
        f"{[(g['name'], g['cc']) for g in GPUS] or 'không có GPU'}.\n"
        "  Colab: Runtime -> Change runtime type -> A100. Máy G4 thì mở notebook 'Qwen3.8-27B G4'.")
A100_80 = GPUS[0]["gib"] > 60

try:
    urllib.request.urlopen("https://huggingface.co/api/models/Qwen/Qwen3.8-27B", timeout=20).read(64)
except urllib.error.HTTPError:
    pass
except Exception as e:
    raise SystemExit(f"Chưa có mạng ({type(e).__name__}: {e}).")

print(json.dumps({
    "notebook": NOTEBOOK_VERSION, "kaggle": ON_KAGGLE, "colab": ON_COLAB,
    "python": sys.version.split()[0], "cpu": os.cpu_count(),
    "gpu": [f"{g['name']} {g['gib']:.1f} GiB, cc {g['cc']}, driver {g['driver']}" for g in GPUS],
    "dia_trong_gib": round(shutil.disk_usage(SCRATCH).free / 2**30, 1),
}, ensure_ascii=False, indent=1))

# %%
# 2) CẤU HÌNH: chỉ cần sửa ô này.
MODEL = "int4"           # "int4": RedHatAI INT4 W4A16 (nhanh nhất trên A100) | "fp8": FP8 chính chủ Qwen (chỉ A100 80 GB; Ampere
                         #         chạy FP8 bằng nhân Marlin W8A16, đúng hơn chút nhưng đọc gấp rưỡi dữ liệu nên chậm hơn)
SPEC = "dspark"          # đoán trước: "dspark" (bộ đoán riêng của Red Hat) | "dflash2" (bộ đoán riêng của z-lab)
                         #           | "mtp" (đầu MTP có sẵn trong model) | "tat"
SPEC_N = "auto"          # số token đoán mỗi lượt: "auto" = 7 với dspark/dflash2, 3 với mtp
CTX = "auto"             # "auto" = 262144 trên A100 80 GB, 131072 trên A100 40 GB; hoặc một số: 65536...
KV = "auto"              # bộ nhớ ngữ cảnh: "auto" (bf16; Ampere không có lõi FP8 để đọc thẳng) | "fp8" (nhỏ một nửa, thử nếu thiếu VRAM)
VISION = True            # nạp phần đọc ảnh
THINKING = True          # mặc định có nghĩ; từng lượt gọi vẫn tự bật tắt được
MAX_SEQS = 4             # số lượt hỏi chạy song song tối đa (ít thì khởi động nhanh hơn, mỗi lượt nhanh hơn)
GPU_UTIL = 0.92          # phần VRAM vLLM được giữ
VLLM_THEM = []           # cờ vLLM thêm cho riêng máy này (để trống: vLLM tự chọn nhân Marlin cho INT4 trên SM80)
TUNNEL = "auto"          # "auto": Colab không mở (xem ô 9) | True | False
KEEP_ALIVE_HOURS = 4     # ô 10 giữ phiên bấy nhiêu giờ rồi tự tắt
KIEM_CHAT_LUONG = True    # ô 9b tự kiểm chất lượng (code, toán, tool, ảnh, ngữ cảnh dài, văn) trước ô 10, khoảng 3–10 phút

VLLM_VERSION = "0.31.0"  # 05/10/2026: sửa lỗi MTP tụt tỉ lệ đoán trúng khi dùng đồ thị CUDA đầy đủ (#59309)
VLLM_SHA256 = "39132b433967f0962d03d8a5f633b9bee6b32708f566c7d5637b79f8d0ef3433"     # bánh xe x86_64, PyPI 13/10/2026
VLLM_PIN_DATE = "2026-10-13T00:00:00Z"
# Nhân FlashInfer dựng sẵn cho Ampere (SM80), cùng bản 0.7.0.post1 mà vLLM 0.31.0 ghim; địa chỉ và mã băm đọc từ
# https://flashinfer.ai/whl/cu130/ ngày 13/10/2026 (gói vỏ 5 KB + gói nhân 116 MB). Thiếu thì vLLM tự dịch nhân khi cần.
_FI = "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.0.post1/flashinfer_jit_cache"
FI_CACHE = [(f"{_FI}-0.7.0.post1+cu130-cp39-abi3-manylinux_2_28_x86_64.whl",
             "9630eeb32396adb89ce8ffa8465fedd14549eb2e2ad3aff182d9e69be4a12075"),
            (f"{_FI}_sm80-0.7.0.post1+cu130-cp39-abi3-manylinux_2_28_x86_64.whl",
             "b142aef0b32682a80ff2fd55aa74630678820776ffe03ecc183f4a01ada1c02a")]
ENGINE_PORT = 8000       # vLLM, chỉ nghe 127.0.0.1
PORT_GW = 8787           # gateway, cổng DUY NHẤT đưa ra tunnel

# Cỡ GB đọc từ API Hugging Face 13/10/2026. Hai bản giữ đầu MTP (bf16) và phần đọc ảnh, giấy phép apache-2.0.
# RedHatAI INT4 so với bản gốc bf16 (model card): MMLU-Pro 98,8%, GPQA Diamond 98,5%, AIME 2025 98,7%.
MODELS = {
    "int4": ("RedHatAI/Qwen3.8-27B-INT4", 19.5),
    "fp8": ("Qwen/Qwen3.8-27B-FP8", 30.9),
}
# Bộ đoán riêng (bf16, khoảng 4 GB). Số token trung bình được nhận mỗi lượt (đoán 7, model card): DSpark HumanEval 4,42,
# toán 5,74, viết 3,77 (vLLM 0.29); DFlash2 HumanEval 4,39, MBPP 4,79, GSM8K 5,46 (SGLang, H200). Chưa ai so hai bộ cùng máy.
DRAFTS = {"dspark": ("RedHatAI/Qwen3.8-27B-speculator.dspark", "dspark"),
          "dflash2": ("z-lab/Qwen3.8-27B-DFlash2", "dflash")}
ALIAS = "qwen3.8-27b"
if MODEL == "fp8" and not A100_80:
    raise SystemExit("MODEL = \"fp8\" nặng 31 GB, không đủ chỗ trên A100 40 GB. Đặt MODEL = \"int4\".")
MODEL_REPO, MODEL_GB = MODELS[MODEL]
SPEC_REPO = DRAFTS[SPEC][0] if SPEC in DRAFTS else None
SPEC_SO = (7 if SPEC in DRAFTS else 3) if SPEC_N == "auto" else int(SPEC_N)
CTX_SIZE = (262144 if A100_80 else 131072) if CTX == "auto" else int(CTX)
print(json.dumps({"model": MODEL_REPO, "doan_truoc": f"{SPEC} {SPEC_SO if SPEC != 'tat' else ''}".strip(), "ngu_canh": CTX_SIZE,
                  "kv": KV, "doc_anh": VISION, "nghi_mac_dinh": THINKING, "vllm": VLLM_VERSION}, ensure_ascii=False, indent=1))

#@chen chung/o3_khoa.py

#@chen chung/o4_vllm.py

#@chen chung/o6_vllm.py

#@chen chung/o7_tu_kiem.py

#@chen chung/o8_gateway.py
