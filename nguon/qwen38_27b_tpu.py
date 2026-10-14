# Nguồn của notebook "Qwen3.8-27B TPU v6e" (Colab Pro, máy TPU v6e-1: một chip, 32 GB HBM). Sửa ở đây rồi chạy: py -3 -X utf8 tao_notebook.py
# Mỗi "# %%" là một ô. Bản A100 và G4 là notebook riêng: TPU không có CUDA, cách nén và máy chạy model khác hẳn.

# %% [markdown]
# # Qwen3.8-27B trên Colab TPU v6e-1: endpoint chuẩn OpenAI
#
# **Run All** → dò máy (phải là TPU) → cài vLLM TPU vào môi trường riêng, vá MTP → đo nhanh cách nhân trọng số 4 bit trên chính TPU này → tải model **NVFP4** → chạy model có đoán trước **MTP** → tự kiểm (trả lời, nghĩ, gọi tool, đọc ảnh, tốc độ) → mở gateway → in khung **ENDPOINT SẴN SÀNG**.
#
# | Bước | Việc cần làm |
# |---|---|
# | 1 | Runtime → Change runtime type → **v6e-1 TPU** |
# | 2 | **Run All**, chờ khung `ENDPOINT SẴN SÀNG` ở ô 9 (lần đầu khoảng 30–45 phút: cài vLLM TPU, tải khoảng 21 GB, dịch đồ thị XLA cho TPU) |
# | 3 | Chép `Base URL`, `API key`, `Model` vào chương trình cần dùng |
#
# **Cách chạy:** vLLM TPU 0.28.0 kèm bản vá của ARahim3 (port PR #3178 của tpu-inference, chưa vào bản chính thức): thiếu bản vá thì đoán trước làm hỏng chữ model trả. Cùng bộ này ARahim3 chạy bản bf16 được 130 token/giây trên v5e-8 (8 chip). v6e-1 chỉ một chip, nên model phải nén 4 bit mới vừa 32 GB: trọng số giữ 4 bit trong bộ nhớ, lúc nhân mới giãn ra bf16 (v6e không có lõi tính FP4/FP8). Sinh mỗi token phải đọc hết trọng số một lần, nên đọc ít byte thì nhanh. **Chưa đo trên v6e-1**; ước khoảng 80–110 token/giây với MTP.
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
# 1) Dò máy: nền tảng, TPU, RAM, đĩa, mạng. Ô này không mở TPU (TPU chỉ cho một tiến trình dùng; ô 4 mới kiểm kỹ).
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


def gpu_csv(fields):
    """Ô 8 gọi để in mức dùng GPU lúc giữ phiên; máy TPU không có GPU."""
    return []


TPU_DEV = sorted(str(p) for p in [*Path("/dev").glob("accel*"), *Path("/dev/vfio").glob("[0-9]*")])
TPU_LOAI = os.environ.get("TPU_ACCELERATOR_TYPE", "")
if not TPU_DEV and not TPU_LOAI:
    raise SystemExit("Máy này không có TPU. Colab: Runtime -> Change runtime type -> v6e-1 TPU.\n"
                     "  Máy A100 hoặc G4 thì mở notebook 'Qwen3.8-27B A100' hoặc 'Qwen3.8-27B G4'.")
_ram = re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text())

try:
    urllib.request.urlopen("https://huggingface.co/api/models/Qwen/Qwen3.8-27B", timeout=20).read(64)
except urllib.error.HTTPError:
    pass
except Exception as e:
    raise SystemExit(f"Chưa có mạng ({type(e).__name__}: {e}).")

print(json.dumps({
    "notebook": NOTEBOOK_VERSION, "kaggle": ON_KAGGLE, "colab": ON_COLAB,
    "python": sys.version.split()[0], "cpu": os.cpu_count(),
    "tpu": TPU_LOAI or "chưa rõ loại (ô 4 kiểm)", "thiet_bi_tpu": TPU_DEV[:8],
    "ram_gib": round(int(_ram.group(1)) / 2**20, 1) if _ram else None,
    "dia_trong_gib": round(shutil.disk_usage(SCRATCH).free / 2**30, 1),
}, ensure_ascii=False, indent=1))

# %%
# 2) CẤU HÌNH: chỉ cần sửa ô này.
MODEL = "quasar"         # "quasar": QUASAR NVFP4, mọi lớp của phần chữ 4 bit, học lại sau khi nén (GPQA Diamond 90,9 so với 91,4
                         #           bản gốc bf16) | "redhat": RedHatAI NVFP4 (MLP 4 bit, attention FP8: nặng hơn, ngữ cảnh ít hơn)
NHOM_FP4 = 16            # số trọng số dùng chung một thang tỉ lệ. 16 = giữ nguyên như lúc nén (đúng nhất). 32 hoặc 64 = tpu-inference
                         # nén lại thưa thang hơn: trọng số nhẹ khoảng 17–25%, nhanh hơn, ngữ cảnh rộng hơn, nhưng sai thêm (chưa đo)
NHAN = "auto"            # cách nhân trọng số 4 bit: "auto" = ô 4 đo trên chính TPU này rồi chọn | "xla" | "kernel"
MTP_N = 3                # số token đầu MTP (có sẵn trong model) đoán trước mỗi lượt; 0 = tắt. ARahim3 đo: 4 không khởi động được
CTX = "auto"             # "auto" = 65536 (quasar, NHOM_FP4 16), 131072 (quasar, 32 hoặc 64); hoặc một số. Thiếu HBM thì ô 6 tự hạ
VISION = True            # nạp phần đọc ảnh (tắt thì khởi động nhanh hơn vài phút: ít đồ thị phải dịch)
THINKING = True          # mặc định có nghĩ; từng lượt gọi vẫn tự bật tắt được
MAX_SEQS = 4             # số lượt hỏi chạy song song tối đa
GPU_UTIL = 0.92          # phần HBM vLLM được giữ (tên cờ của vLLM vẫn là gpu-memory-utilization)
JSON_MODE = False        # True nếu chương trình gọi vào dùng response_format / JSON schema: vllm-tpu 0.28.0 có MTP chết ở đường đó
                         # ("AttributeError: __delitem__") trừ khi tắt async scheduling (chậm hơn chút)
VLLM_CO_THEM = []        # cờ vLLM thêm nếu anh cần
TUNNEL = "auto"          # "auto": Colab không mở (xem ô 9) | True | False
KEEP_ALIVE_HOURS = 4     # ô 10 giữ phiên bấy nhiêu giờ rồi tự tắt
KIEM_CHAT_LUONG = True    # ô 9b tự kiểm chất lượng (code, toán, tool, ảnh, ngữ cảnh dài, văn) trước ô 10, khoảng 3–10 phút

VLLM_TPU_VERSION = "0.28.0"   # bản bản vá MTP nhắm tới; 0.31.0 vẫn thiếu nửa sửa ở phần chạy model (PR #3178 đóng 09/10/2026 chưa gộp)
VLLM_TPU_SHA256 = "9844a264ae4999c0ab31b96c457d4447c2637a7a3da3ebe9efd6f3bc24e5cc81"        # bánh xe cp312, PyPI 27/08/2026
TPU_INFERENCE_SHA256 = "7bf33b88ed700f43ef8fed9a77b5e7e7eb53bf5ce7197d06cbc418a19f742cce"   # tpu-inference 0.28.0, PyPI 27/08/2026
VLLM_PIN_DATE = "2026-09-09T23:59:59Z"   # thư viện đi kèm lấy bản có tới ngày này (trước khi 0.29.0 ra 10/09)
PATCH_URL = ("https://raw.githubusercontent.com/ARahim3/kaggle-tpu-lab/f3a6b48bf3e97ae082af5dee1480fa315613fd94/"
             "qwen38-27b/patches/mtp-rollback-v0280.diff")
PATCH_SHA256 = "eb7dc93795b3066af74ed19dc039f13690df010d286d2c461fa4624c422b50d2"
ENGINE_PORT = 8000       # vLLM, chỉ nghe 127.0.0.1
PORT_GW = 8787           # gateway, cổng DUY NHẤT đưa ra tunnel

# (repo, bản ghim, GB, ngữ cảnh "auto" với NHOM_FP4 16 / lớn hơn). Cỡ và bản đọc từ API Hugging Face 13/10/2026, cả hai apache-2.0,
# đều giữ đầu MTP và phần đọc ảnh ở bf16. Trong HBM, tpu-inference giữ thang tỉ lệ ở float32 (checkpoint là fp8): với nhóm 16,
# mỗi trọng số tốn 0,75 byte, nên QUASAR chiếm khoảng 25 GB trên 32 GB.
MODELS = {
    "quasar": ("QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4", "d9af17e20644c5b9a6d66dff1d14f349e2e61b05", 20.6, 65536, 131072),
    "redhat": ("RedHatAI/Qwen3.8-27B-NVFP4", "2889ddeeda282f17fc9e159a42125e535159a5e5", 24.7, 32768, 65536),
}
ALIAS = "qwen3.8-27b"
if NHOM_FP4 not in (16, 32, 64, 128):
    raise SystemExit("NHOM_FP4 phải là 16, 32, 64 hoặc 128.")
MODEL_REPO, MODEL_REV, MODEL_GB, _ctx16, _ctx_thua = MODELS[MODEL]
CTX_SIZE = (_ctx16 if NHOM_FP4 == 16 else _ctx_thua) if CTX == "auto" else int(CTX)
# Ô 6 dùng chung với notebook GPU: đoán trước chỉ có MTP, bộ nhớ ngữ cảnh bf16
SPEC, SPEC_SO, DRAFTS, SPEC_REPO, KV = ("mtp" if MTP_N else "tat"), MTP_N, {}, None, "auto"
VLLM_VERSION = f"TPU {VLLM_TPU_VERSION}"
VLLM_CHO_GIAY = 5400     # lần đầu dịch đồ thị XLA lâu
print(json.dumps({"model": f"{MODEL_REPO} @ {MODEL_REV[:8]}", "nhom_fp4": NHOM_FP4, "doan_truoc": f"mtp {MTP_N}" if MTP_N else "tắt",
                  "ngu_canh": CTX_SIZE, "doc_anh": VISION, "nghi_mac_dinh": THINKING, "vllm_tpu": VLLM_TPU_VERSION},
                 ensure_ascii=False, indent=1))

#@chen chung/o3_khoa.py

# %%
#@nhung do_nhan_tpu.py

# %%
# 4) vLLM TPU: môi trường Python 3.12 riêng (bánh xe vllm-tpu chỉ có cho 3.12), kiểm mã băm, vá MTP, kiểm TPU, rồi đo nhanh
#    cách nhân trọng số 4 bit trên chính TPU này (khoảng 1–2 phút) để chọn đường nhanh hơn.
VENV = SCRATCH / "vllm-tpu-venv"
VPY = VENV / "bin" / "python"
os.environ.pop("TPU_LIBRARY_PATH", None)     # môi trường riêng có libtpu của nó; biến này của máy sẽ trỏ sang bản khác
if globals().get("ENGINE_PROC") is not None and ENGINE_PROC.poll() is None:     # TPU chỉ một tiến trình dùng được
    ENGINE_PROC.terminate()
    ENGINE_PROC.wait(timeout=120)


def chay(cmd, **kw):
    """Chạy lệnh, in 3.000 ký tự cuối khi hỏng; trả mã thoát."""
    r = subprocess.run([str(c) for c in cmd], text=True, capture_output=True, **kw)
    if r.returncode:
        print(f"  lệnh hỏng (mã {r.returncode}): {' '.join(map(str, cmd))[:200]}\n" + ((r.stdout or "") + (r.stderr or ""))[-3000:])
    return r.returncode


def vllm_ok():
    """Kiểm bản cài, không mở TPU (chưa import vllm: nó dò nền tảng và in log ra stdout)."""
    r = subprocess.run([str(VPY), "-c", "from importlib.metadata import version as v\nimport importlib.util as u, jax, torch\n"
                        "assert u.find_spec('vllm') and u.find_spec('tpu_inference')\n"
                        "print('BAN', v('vllm-tpu'), v('tpu-inference'), 'jax', jax.__version__, 'torch', torch.__version__)"],
                       text=True, capture_output=True)
    m = re.search(r"^BAN (\S+) (\S+) (.*)$", r.stdout, re.M)
    return bool(m and m.group(1) == m.group(2) == VLLM_TPU_VERSION), (m.group(0) if m else (r.stdout + r.stderr).strip()[-1500:])


t0 = time.time()
ok, chu = vllm_ok() if VPY.exists() else (False, "")
if not ok:
    if not shutil.which("uv"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    UV = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
    shutil.rmtree(VENV, ignore_errors=True)
    if chay([UV, "venv", VENV, "--python", "3.12", "-q"]):
        raise SystemExit("Không tạo được môi trường Python 3.12 riêng cho vLLM TPU.")
    _banh = []
    for _goi, _ten, _sha in (("vllm-tpu", f"vllm_tpu-{VLLM_TPU_VERSION}-cp312-cp312-manylinux_2_24_x86_64.whl", VLLM_TPU_SHA256),
                             ("tpu-inference", f"tpu_inference-{VLLM_TPU_VERSION}-py3-none-any.whl", TPU_INFERENCE_SHA256)):
        _pypi = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{_goi}/{VLLM_TPU_VERSION}/json", timeout=60))
        _banh.append(tai(next(f["url"] for f in _pypi["urls"] if f["filename"] == _ten), SCRATCH / _ten, _sha))
    # PyTorch bản CPU như ảnh Docker chính thức của vllm-tpu (bản mặc định kéo thêm khoảng 3 GB thư viện CUDA vô dụng trên TPU).
    if chay([UV, "pip", "install", "--python", VPY, "--torch-backend=cpu", "--exclude-newer", VLLM_PIN_DATE, *_banh],
            env=dict(os.environ, UV_HTTP_TIMEOUT="300")):
        raise SystemExit("Cài vLLM TPU hỏng (xem chữ in ở trên).")
    for _b in _banh:
        _b.unlink(missing_ok=True)
    ok, chu = vllm_ok()
if not ok:
    raise SystemExit("vLLM TPU cài xong nhưng không nạp được. Chữ in ra:\n" + chu)
print(f"[OK] vLLM TPU sẵn sàng sau {time.time() - t0:.0f} giây: {chu}")

# Bản vá MTP (khôi phục trạng thái lớp attention tuyến tính khi token đoán bị loại). Không vá được thì tắt MTP.
if MTP_N:
    try:
        _va = SCRATCH / "mtp-rollback-v0280.diff"
        if not _va.exists():
            tai(PATCH_URL, _va, PATCH_SHA256)
        if not shutil.which("patch"):
            chay(["apt-get", "install", "-y", "-q", "patch"])
        _site = subprocess.check_output([str(VPY), "-c", "import importlib.util as u, os\n"
                                         "print(os.path.dirname(os.path.dirname(u.find_spec('tpu_inference').origin)))"], text=True).strip()
        _p = subprocess.run(["patch", "-p1", "-d", _site, "-i", str(_va), "--no-backup-if-mismatch", "-N"], capture_output=True, text=True)
        VA_MTP = _p.returncode == 0 or "previously applied" in _p.stdout
        print("[OK] Đã vá MTP." if VA_MTP else f"  vá MTP hỏng:\n{_p.stdout[-1500:]}{_p.stderr[-500:]}")
    except Exception as e:
        VA_MTP = False
        print(f"  vá MTP hỏng: {type(e).__name__}: {str(e)[:300]}")
    if not VA_MTP:
        print("  -> TẮT MTP: thiếu bản vá thì đoán trước làm hỏng chữ model trả. Chạy được nhưng chậm hơn.")
        MTP_N, SPEC, SPEC_SO = 0, "tat", 0

# Kiểm TPU bằng chính jax của môi trường mới: số chip, loại, HBM.
_r = subprocess.run([str(VPY), "-c", "import jax, json\nd = jax.devices()\nm = [x.memory_stats() or {} for x in d]\n"
                     "print('TPU_CHECK ' + json.dumps({'so': len(d), 'nen': d[0].platform, 'ten': d[0].device_kind,"
                     " 'hbm_gib': round(sum(x.get('bytes_limit', 0) for x in m) / 2**30, 1)}))"],
                    capture_output=True, text=True, timeout=600)
_m = re.search(r"TPU_CHECK (\{.*\})", _r.stdout)
TPU = json.loads(_m.group(1)) if _m else {}
if TPU.get("nen") != "tpu":
    raise SystemExit("jax không thấy TPU (máy chưa gắn TPU, hoặc tiến trình khác đang giữ TPU: Runtime -> Restart session "
                     "rồi chạy lại). Chữ in ra:\n" + (_r.stdout + _r.stderr)[-2000:])
if TPU["hbm_gib"] and TPU["hbm_gib"] < 28:
    raise SystemExit(f"TPU này ({TPU['so']} × {TPU['ten']}) chỉ có {TPU['hbm_gib']} GiB HBM, không đủ cho model khoảng 25 GB. "
                     "Chọn v6e-1 TPU.")
SO_CHIP = TPU["so"]
print(f"[OK] TPU: {SO_CHIP} × {TPU['ten']}, HBM {TPU['hbm_gib']} GiB"
      + ("" if "v6" in TPU["ten"] else " (notebook viết cho v6e; loại này chưa thử)"))

VLLM_ENV_THEM = {"VLLM_XLA_CACHE_PATH": str(SCRATCH / "xla-cache"), "MIN_TOKEN_BUCKET": "64", "NUM_PRECOMPILE_WORKERS": "4"}
if NHOM_FP4 != 16:
    VLLM_ENV_THEM["REQUANTIZE_COMPRESSED_TENSOR_NVFP4_BLOCK_SIZE"] = str(NHOM_FP4)

# Đo hai đường nhân trọng số 4 bit của tpu-inference: "xla" (mặc định) và "kernel" (tokamax, mặc định tắt vì từng ra NaN).
NHAN_DUNG = NHAN
if NHAN == "auto":
    try:
        _r = subprocess.run([str(VPY), str(Path("do_nhan_tpu.py").resolve())], capture_output=True, text=True, timeout=900,
                            env=dict(os.environ, DO_NHOM=str(NHOM_FP4)))
        _m = re.search(r"DO_NHAN (\{.*\})", _r.stdout)
        DO_NHAN = json.loads(_m.group(1)) if _m else {"loi": (_r.stdout + _r.stderr)[-1200:]}
    except Exception as e:
        DO_NHAN = {"loi": f"{type(e).__name__}: {e}"}
    _tot = lambda d: "ms" in d and not d.get("nan") and d.get("sai", 1) < 0.2
    _x, _k = DO_NHAN.get("xla", {}), DO_NHAN.get("kernel", {})
    NHAN_DUNG = "kernel" if _tot(_k) and (not _tot(_x) or _k["ms"] < 0.9 * _x["ms"]) else "xla"
    print("Đo phép nhân (6 cặp lớp MLP, 64 token):")
    for _ten in ("bf16", "xla", "kernel"):
        _d = DO_NHAN.get(_ten, {})
        print(f"  {_ten:<7}" + (f"{_d['ms']:>8.2f} ms  {_d['gb_s']:>5} GB/giây" if "ms" in _d else f"  {_d.get('loi', 'không đo')}")
              + (f"  sai {_d['sai']:.3f}" if "sai" in _d else "") + ("  CÓ NaN" if _d.get("nan") else ""))
    if "loi" in DO_NHAN or "tao_loi" in DO_NHAN:
        print("  đo không trọn:", (DO_NHAN.get("loi") or DO_NHAN.get("tao_loi"))[-600:])
if NHAN_DUNG == "kernel":
    VLLM_ENV_THEM["ENABLE_QUANTIZED_MATMUL_KERNEL"] = "1"
VLLM_THEM = ["--tensor-parallel-size", str(SO_CHIP), *(["--no-async-scheduling"] if JSON_MODE else []), *VLLM_CO_THEM]
print(f"[OK] Nhân trọng số 4 bit bằng đường: {NHAN_DUNG}")

# %%
# 5) Model: tải đúng bản đã ghim từ Hugging Face về SCRATCH (đã có thì bỏ qua)
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
if CHAY_NEN:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import snapshot_download

trong = shutil.disk_usage(MODEL_DIR).free / 1e9
if trong < MODEL_GB + 8:
    raise SystemExit(f"Đĩa {MODEL_DIR} còn {trong:.0f} GB, không đủ cho model khoảng {MODEL_GB:.0f} GB.")
t0 = time.time()
model_path = Path(snapshot_download(MODEL_REPO, revision=MODEL_REV, local_dir=str(MODEL_DIR / MODEL_REPO.replace("/", "__")),
                                    token=HF_TOKEN or None, max_workers=16))
spec_path = None
print(f"[OK] Model: {model_path} "
      f"({sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / 1e9:.1f} GB, {time.time() - t0:.0f} giây)")

#@chen chung/o6_vllm.py

#@chen chung/o7_tu_kiem.py

#@chen chung/o8_gateway.py
