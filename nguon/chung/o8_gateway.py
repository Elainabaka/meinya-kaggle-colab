# Ô 8, 9, 9b, 10 dùng chung (chèn bằng "#@chen chung/o8_gateway.py"; ô 9b kiểm chất lượng ở o9b_chat_luong.py). Cần từ các
# ô trước: ENGINE_PORT, ENGINE_PROC, ENGINE_TEN, DOC_ANH, MTP_TOKENS, TOC_DO, KET_QUA, CTX_SIZE, ALIAS, THINKING, TUNNEL,
# KEEP_ALIVE_HOURS, KIEM_CHAT_LUONG, PORT_GW, BIN_DIR, LOG_DIR, gpu_csv (ô 1) và tai, hoi, KHONG_NGHI, các khóa (ô 3).

# %%
#@nhung gateway.py

# %%
# 9) Gateway :8787 + tunnel -> in ENDPOINT. Chỉ gateway được đưa ra ngoài, máy chạy model thì không.
GW_TOKEN = GW_TOKEN_CO_DINH or ("gw-" + secrets.token_hex(20))
GW_FILE = Path("gateway.py").resolve()
assert GW_FILE.exists(), "Chưa chạy ô 8 (%%writefile gateway.py)."
for _ten in ("cf_proc", "gw_proc"):                                # chạy lại ô này: tắt bản cũ trước
    _p = globals().get(_ten)
    if _p is not None and _p.poll() is None:
        _p.terminate()
if ENGINE_PROC.poll() is not None:     # 13/10: bấm Stop ô 10 (bản notebook cũ) tắt luôn máy chạy model, gateway chờ vô ích
    raise SystemExit(f"{ENGINE_TEN} đã tắt (mã thoát {ENGINE_PROC.returncode}), gateway không có gì để nối. "
                     f"Chạy lại từ ô 6 (bấm vào ô 6 rồi Ctrl+F10). Log trong {LOG_DIR}")

gw_log = LOG_DIR / "gateway.log"
gw_proc = chay_nen(
    [sys.executable, "-m", "uvicorn", "gateway:app", "--host", "127.0.0.1", "--port", str(PORT_GW),
     "--workers", "1", "--log-level", "warning"],
    cwd=str(GW_FILE.parent), stdout=open(gw_log, "w"), stderr=subprocess.STDOUT,
    env=dict(os.environ, GATEWAY_TOKEN=GW_TOKEN, VLLM_API_KEY=API_KEY, VLLM_URL=f"http://127.0.0.1:{ENGINE_PORT}",
             SERVED_MODEL_NAME=ALIAS, MAX_MODEL_LEN=str(CTX_SIZE), TEXT_ONLY="false" if DOC_ANH else "true",
             GATEWAY_MAX_OUTPUT_TOKENS="32768", MTP_TOKENS=str(MTP_TOKENS), VLLM_TPU_VERSION=ENGINE_TEN,
             # số lượt gateway cho chạy cùng lúc = số lượt vLLM chạy song song (gateway.py mặc định 3: 13/10 G4 MAX_SEQS 4,
             # lượt thứ tư chờ rồi bị trả 429 "gateway busy")
             GATEWAY_CONCURRENCY=str(globals().get("MAX_SEQS", 3)),
             RUN_ID=str(int(time.time()))))
H_GW = {"Authorization": f"Bearer {GW_TOKEN}"}
LOCAL_GW = f"http://127.0.0.1:{PORT_GW}"
for _ in range(30):
    if gw_proc.poll() is not None:
        raise SystemExit("Gateway chết:\n" + gw_log.read_text(errors="ignore")[-3000:])
    try:
        if requests.get(LOCAL_GW + "/healthz", headers=H_GW, timeout=5).json().get("ok") is True:
            break
    except Exception:
        pass
    time.sleep(1)
else:
    raise SystemExit("Gateway không trả /healthz sau 30 giây:\n" + gw_log.read_text(errors="ignore")[-3000:])
print(f"gateway chạy ở 127.0.0.1:{PORT_GW} -> {ENGINE_TEN} 127.0.0.1:{ENGINE_PORT}")

MO_TUNNEL = (not ON_COLAB) if TUNNEL == "auto" else bool(TUNNEL)
BASE_URL, cf_proc = LOCAL_GW, None
if not MO_TUNNEL:
    print("KHÔNG mở tunnel: endpoint chỉ dùng được bên trong máy này.")
    if ON_COLAB:
        print('  Lý do mặc định tắt trên Colab: FAQ của Colab (đọc 13/10/2026) ghi "The following are disallowed from all managed\n'
              '  Colab runtimes", trong đó có "file hosting, media serving, or other web service offerings not related to interactive\n'
              '  compute with Colab" và "connecting to remote proxies" (https://research.google.com/colaboratory/faq.html).\n'
              "  Endpoint qua tunnel có thuộc diện đó không thì Colab không nói rõ; gói Pro không gỡ được danh sách này.\n"
              "  Anh tự quyết: đặt TUNNEL = True ở ô 2 rồi chạy lại ô 2 và ô 9.")
else:
    CF_VERSION = "2026.10.0"      # sha256 từ API GitHub ngày 13/10/2026
    CF_SHA256 = "d33ff2d14475178d2012c2c56beba87389ac5ded27649519f198a7d3134a99db"
    cf_bin = BIN_DIR / f"cloudflared-{CF_VERSION}"
    if not cf_bin.exists():
        tai(f"https://github.com/cloudflare/cloudflared/releases/download/{CF_VERSION}/cloudflared-linux-amd64", cf_bin, CF_SHA256)
        os.chmod(cf_bin, 0o755)
    cf_log = LOG_DIR / "cloudflared.log"
    if CF_TOKEN:                  # tunnel có tên: địa chỉ cố định, cấu hình trong bảng điều khiển Cloudflare trỏ về http://127.0.0.1:8787
        cf_proc = chay_nen([str(cf_bin), "tunnel", "--no-autoupdate", "run"], stdout=open(cf_log, "w"),
                           stderr=subprocess.STDOUT, env=dict(os.environ, TUNNEL_TOKEN=CF_TOKEN))
        BASE_URL = PUBLIC_BASE_URL or None
        if not BASE_URL:
            print("Tunnel có tên đã chạy nhưng thiếu Secret PUBLIC_BASE_URL (ví dụ https://qwen.tenmien.com): không tự kiểm qua tunnel được.")
    else:                         # Quick Tunnel: không cần tài khoản, địa chỉ đổi mỗi phiên
        cf_proc = chay_nen([str(cf_bin), "tunnel", "--no-autoupdate", "--url", LOCAL_GW], stdout=open(cf_log, "w"),
                           stderr=subprocess.STDOUT)
        BASE_URL, _han = None, time.time() + 120
        while time.time() < _han and not BASE_URL and cf_proc.poll() is None:
            _tim = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", cf_log.read_text(errors="ignore"))
            BASE_URL = _tim.group(0) if _tim else None
            time.sleep(1)
        if not BASE_URL:
            raise SystemExit("Không mở được Quick Tunnel:\n" + cf_log.read_text(errors="ignore")[-3000:])
    if BASE_URL:                  # kiểm đường thật từ ngoài vào: /healthz rồi một lượt chat
        _ua = {**H_GW, "User-Agent": "Mozilla/5.0"}
        for _lan in range(40):
            try:
                if requests.get(BASE_URL + "/healthz", headers=_ua, timeout=10).json().get("ok") is True:
                    break
            except Exception:
                pass
            time.sleep(3)
        else:
            raise SystemExit(f"Gateway chưa trả lời qua tunnel sau 2 phút: {BASE_URL}")
        for _lan_chat in range(6):    # tunnel mới mở còn chập chờn: 13/10 /healthz đạt rồi lượt chat liền sau báo "Network is unreachable"
            try:
                j = hoi([{"role": "user", "content": "Reply with exactly: TUNNEL_OK"}], url=BASE_URL + "/v1/chat/completions",
                        headers=_ua, max_tokens=32, temperature=0, **KHONG_NGHI)
                break
            except requests.RequestException as e:
                _loi_chat = f"{type(e).__name__}: {str(e)[:200]}"
                time.sleep(5)
        else:
            raise SystemExit(f"/healthz qua tunnel đạt nhưng lượt chat hỏng 6 lần liền ({_loi_chat}). Chạy lại ô này.")
        assert "TUNNEL_OK" in (j["choices"][0]["message"].get("content") or ""), f"Chat qua tunnel trả sai: {json.dumps(j)[:300]}"
        print(f"[OK] Đã gọi được từ ngoài vào qua tunnel (thử /healthz lần {_lan + 1}, một lượt chat đúng).")

_url = BASE_URL or "<địa chỉ tunnel có tên của anh>"
_key = "<giá trị Secret GATEWAY_TOKEN của anh>" if GW_TOKEN_CO_DINH else GW_TOKEN     # khóa cố định thì không in ra output
print("\n" + "=" * 78)
print("  ENDPOINT SẴN SÀNG (chuẩn OpenAI)")
print("=" * 78)
print(f"  Base URL : {_url}/v1")
print(f"  API key  : {_key}")
print(f"  Model    : {ALIAS}")
_anh = "không" if not DOC_ANH else "có" if KET_QUA.get("đọc ảnh", "").startswith("đạt") else "có, nhưng ô 7 đọc SAI ảnh thử"
print(f"  Ngữ cảnh : {CTX_SIZE} token | đọc ảnh: {_anh} | nghĩ mặc định: {'bật' if THINKING else 'tắt'}"
      + (f" | {globals().get('SPEC_TEN') or 'MTP'} {MTP_TOKENS} token" if MTP_TOKENS else "")
      + (f" | code {TOC_DO:.1f} token/giây" if TOC_DO else ""))
print("-" * 78)
print("  Thử nhanh bằng curl:")
print(f"    curl {_url}/v1/chat/completions -H \"Authorization: Bearer {_key}\" -H \"Content-Type: application/json\" \\")
print(f"      -d '{{\"model\":\"{ALIAS}\",\"stream\":true,\"messages\":[{{\"role\":\"user\",\"content\":\"Xin chào\"}}]}}'")
print("  Python:")
print(f"    from openai import OpenAI; c = OpenAI(base_url=\"{_url}/v1\", api_key=\"<API key>\")")
print("-" * 78)
print("  Đường được mở: GET /healthz | GET /capabilities | GET /v1/models | POST /v1/chat/completions")
print("  Tắt nghĩ cho một lượt : thêm \"chat_template_kwargs\": {\"enable_thinking\": false}")
print("  Chỉnh mức nghĩ        : \"chat_template_kwargs\": {\"reasoning_effort\": \"low\" | \"medium\" | \"xhigh\"}")
print("  Nên gọi với \"stream\": true. Lượt không stream quá 60 giây: gateway gửi khoảng trắng giữ kết nối\n"
      "  (Cloudflare cắt lượt im lặng 100 giây), nên lỗi xảy ra sau mốc đó về dưới mã 200 kèm JSON lỗi.")
if MO_TUNNEL and not CF_TOKEN:
    print("  Địa chỉ trycloudflare.com và API key đổi mỗi phiên. Muốn cố định: xem bảng Secret ở đầu notebook.")
print("=" * 78)

#@chen chung/o9b_chat_luong.py

# %%
# 10) Giữ phiên sống và báo nhịp. Bấm Stop ở ô này KHÔNG tắt endpoint; muốn tắt hẳn thì chạy tat_het().
def tat_het():
    for ten, p in (("tunnel", cf_proc), ("gateway", gw_proc), (ENGINE_TEN, ENGINE_PROC)):
        if p is not None and p.poll() is None:
            p.terminate()
            print("đã tắt", ten)


def gio_vn():
    return time.strftime("%H:%M", time.gmtime(time.time() + 7 * 3600))


_het = time.time() + KEEP_ALIVE_HOURS * 3600
print(f"{gio_vn()} (giờ VN): giữ endpoint {KEEP_ALIVE_HOURS} giờ rồi tự tắt. Sửa KEEP_ALIVE_HOURS ở ô 2 nếu cần.")
try:
    while time.time() < _het:
        _chet = [t for t, p in ((ENGINE_TEN, ENGINE_PROC), ("gateway", gw_proc), ("tunnel", cf_proc))
                 if p is not None and p.poll() is not None]
        if _chet:
            print(f"{gio_vn()} ĐÃ CHẾT: {_chet}. Xem log trong {LOG_DIR}, rồi chạy lại từ ô 6.")
            break
        _luot = gw_log.read_text(errors="ignore").count("POST /v1/chat/completions")
        _gpu = " ".join(f"GPU{i}:{u}MiB/{p_}%" for i, u, p_ in gpu_csv("index,memory.used,utilization.gpu"))
        print(f"{gio_vn()} | còn {(_het - time.time()) / 3600:.1f} giờ | {_luot} lượt chat | {_gpu}", flush=True)
        time.sleep(min(300, max(1, _het - time.time())))
    else:
        print(f"{gio_vn()} hết {KEEP_ALIVE_HOURS} giờ, tắt endpoint.")
        tat_het()
except KeyboardInterrupt:
    print("Đã dừng ô báo nhịp. Endpoint VẪN chạy. Tắt hẳn: chạy tat_het() trong một ô mới.")
