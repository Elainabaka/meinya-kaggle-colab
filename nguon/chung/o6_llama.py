# Ô 6 dùng chung cho notebook chạy llama.cpp (chèn bằng "#@chen chung/o6_llama.py").
# Cần từ ô 1–5: GPUS, LOG_DIR, gpu_csv, API_KEY, hoi_anh, server_bin, LLAMA_TAG, LLAMA_ENV, LLAMA_HELP, model_path,
# mmproj_path, mtp_path; từ ô 2: ALIAS, ENGINE_PORT, KV_TYPE, LLAMA_MAU (cờ lấy mẫu theo model card), VUA_TRON, THINKING,
# MTP, MTP_FILE, MTP_N, SPLIT ("auto" | "tensor" | "layer"), CTX_SIZE. Tùy chọn: LLAMA_THEM (cờ thêm cho llama-server),
# ANH_MIN_TOKEN (số token ít nhất cho mỗi ảnh, mặc định 1024; 0 = để llama.cpp tự chọn).
# Đặt cho ô 7–10: ENGINE_PROC, ENGINE_TEN, ENGINE_LOG, DOC_ANH, MTP_TOKENS, CTX_SIZE (sau khi co), SPLIT_DUNG.

# %%
# 6) Chạy llama-server (chỉ nghe 127.0.0.1) và chờ nạp xong model
def co_co(flag):
    """llama-server bản này có nhận cờ `flag` không (đọc từ --help, để không chết khi cờ đổi tên)."""
    return re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", LLAMA_HELP) is not None


def cho_song(port, proc, timeout_s):
    t0, cuoi = time.time(), None
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False, f"tiến trình thoát với mã {proc.returncode}"
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
            cuoi = (r.status_code, r.text[:120])
            if r.status_code == 200:
                return True, cuoi
        except Exception as e:
            cuoi = repr(e)[:160]
        time.sleep(3)
    return False, cuoi


if globals().get("ENGINE_PROC") is not None and ENGINE_PROC.poll() is None:     # chạy lại ô này: tắt bản cũ trước
    ENGINE_PROC.terminate()
    ENGINE_PROC.wait(timeout=60)

args = [str(server_bin), "-m", str(model_path), "--alias", ALIAS,
        "--host", "127.0.0.1", "--port", str(ENGINE_PORT), "--api-key", API_KEY,
        "-np", "1", "-t", str(os.cpu_count() or 4),
        "-fa", "on", "-ctk", KV_TYPE, "-ctv", KV_TYPE, "--jinja", *LLAMA_MAU]
if VUA_TRON:
    args += ["-ngl", "99"]
if mmproj_path:
    args += ["--mmproj", str(mmproj_path)]
    # Model họ Qwen đọc sai chữ nhỏ khi ảnh ra quá ít token (llama.cpp tự cảnh báo "require at minimum 1024 image tokens").
    # Đo 13/10 với FableCoder: không cờ thì ảnh số 448×224 chỉ ra khoảng 100 token, đọc 7391 thành '71' (đúng 3/6 ảnh thử);
    # 256, 512, 1024 đều đúng 6/6. Lấy 1024 theo llama.cpp: mỗi ảnh nhỏ tốn khoảng 1.060 token ngữ cảnh, VRAM không tăng.
    _amin = globals().get("ANH_MIN_TOKEN", 1024)
    if _amin and co_co("--image-min-tokens"):
        args += ["--image-min-tokens", str(_amin)]
if not THINKING and co_co("--reasoning-budget"):
    args += ["--reasoning-budget", "0"]
# MTP_FILE None: lớp MTP nằm sẵn trong file model (không cần -md)
MTP_DUNG = bool(MTP and (mtp_path or not MTP_FILE) and co_co("--spec-type") and "draft-mtp" in LLAMA_HELP)
if MTP_DUNG:
    args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(MTP_N)] + (["-md", str(mtp_path)] if mtp_path else []) + ["-ngld", "99"]
elif MTP:
    print("  (bản llama.cpp này không có --spec-type draft-mtp: chạy không MTP)")
args += next(([f] for f in ("--no-webui", "--no-ui") if co_co(f)), [])
args += globals().get("LLAMA_THEM", [])
# Chia model cho nhiều GPU. "tensor": mỗi lớp chia cho mọi GPU, các GPU chạy cùng lúc (llama.cpp ghi EXPERIMENTAL).
# Đo 13/10 trên 2×T4, Qwen3.8-27B gốc: "layer" 9,8–10,8 token/giây; "tensor" 12,6–13,1; "tensor" + MTP: Q5 15,7–25,3, Q4 19,1–30,0.
# "row" không dùng: bản dựng sẵn báo "device CUDA0 does not support split buffers".
# Nạp hỏng (thường là hết VRAM) thì thử lại "tensor" với nửa ngữ cảnh, rồi mới lùi về "layer".
if not VUA_TRON:
    lan_thu = [(None, CTX_SIZE)]
elif len(GPUS) < 2:      # một GPU (Colab): hỏng thì thử nửa ngữ cảnh, rồi một phần tư
    lan_thu = [(None, CTX_SIZE)] + [(None, CTX_SIZE // k) for k in (2, 4) if CTX_SIZE // k >= 16384]
elif SPLIT == "auto" and re.search(r"--split-mode[^\n]*tensor", LLAMA_HELP):
    lan_thu = [("tensor", CTX_SIZE)] + ([("tensor", CTX_SIZE // 2)] if CTX_SIZE > 32768 else []) + [("layer", CTX_SIZE)]
else:
    lan_thu = [("layer" if SPLIT == "auto" else SPLIT, CTX_SIZE)]

ENGINE_TEN, ENGINE_LOG = f"llama-server {LLAMA_TAG}", LOG_DIR / "llama-server.log"
DOC_ANH_DAU = None
for SPLIT_DUNG, _ctx in lan_thu:
    lenh = args + ["-c", str(_ctx)] + (["-sm", SPLIT_DUNG, "-ts", ",".join("1" for _ in GPUS)] if SPLIT_DUNG else [])
    print("Lệnh:", " ".join(a if a != API_KEY else "<khóa nội bộ>" for a in lenh))
    SERVER_LOG_F = open(ENGINE_LOG, "w")
    ENGINE_PROC = chay_nen(lenh, stdout=SERVER_LOG_F, stderr=subprocess.STDOUT, env=LLAMA_ENV)
    t0 = time.time()
    ok, cuoi = cho_song(ENGINE_PORT, ENGINE_PROC, timeout_s=1200)
    if ok and mmproj_path:
        # Phần đọc ảnh xin thêm bộ đệm ở GPU0 khi gặp ảnh đầu tiên; 13/10 server chết đúng lúc đó vì GPU0 hết VRAM.
        # Hỏi thử một ảnh ngay đây: server chết thì coi như nạp hỏng và thử cấu hình kế tiếp.
        try:
            DOC_ANH_DAU = hoi_anh()
        except Exception as e:
            DOC_ANH_DAU = f"LỖI {type(e).__name__}: {str(e)[:160]}"
        if ENGINE_PROC.poll() is not None:
            ok, cuoi = False, "server chết ở lượt đọc ảnh đầu tiên (thường là GPU0 hết VRAM)"
    if ok:
        CTX_SIZE = _ctx
        break
    if ENGINE_PROC.poll() is None:
        ENGINE_PROC.kill()
    SERVER_LOG_F.close()
    print(f"  không nạp được (chia {SPLIT_DUNG}, ngữ cảnh {_ctx}): {cuoi}. Cuối log:\n" + ENGINE_LOG.read_text(errors="ignore")[-600:])
if not ok:
    raise SystemExit(f"llama-server không lên ({cuoi}). Cuối log:\n" + ENGINE_LOG.read_text(errors="ignore")[-4000:] +
                     "\nNếu là hết bộ nhớ (out of memory): hạ CTX hoặc chọn QUANT nhẹ hơn ở ô 2, hoặc đặt MTP = False.")
print(f"[OK] Model nạp xong sau {time.time() - t0:.0f} giây (PID {ENGINE_PROC.pid}, chia GPU: {SPLIT_DUNG or 'không'}, "
      f"ngữ cảnh {CTX_SIZE})" + (f"; đọc ảnh lần đầu: {DOC_ANH_DAU!r}" if DOC_ANH_DAU is not None else ""))
_log = ENGINE_LOG.read_text(errors="ignore")
for _dong in re.findall(r"[^\n]*(?:model buffer size|KV buffer size|compute buffer size)[^\n]*", _log)[:8]:
    print("  ", _dong.strip()[:160])
for i, used, total in gpu_csv("index,memory.used,memory.total"):
    print(f"   GPU {i}: {used}/{total} MiB")
DOC_ANH, MTP_TOKENS = mmproj_path is not None, (MTP_N if MTP_DUNG else 0)     # các ô dùng chung 7–10 đọc hai biến này
