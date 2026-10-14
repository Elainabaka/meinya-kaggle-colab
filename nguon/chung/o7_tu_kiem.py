# Ô 7 dùng chung (chèn bằng "#@chen chung/o7_tu_kiem.py"). Ô 6 của từng notebook phải đặt:
#   ENGINE_PORT, ENGINE_PROC (tiến trình), ENGINE_TEN (tên in ra), ENGINE_LOG (file log), DOC_ANH (bool), MTP_TOKENS (0 = tắt)
#   và có thể đặt DEM_MTP(): trả (số token đoán trúng, số token đã đoán) cộng dồn, khi máy chạy model không trả trường timings.

# %%
# 7) Tự kiểm qua API: trả lời, tốc độ, nghĩ + stream, gọi tool, đọc ảnh. Lỗi ở hai mục đầu thì dừng.
LOCAL = f"http://127.0.0.1:{ENGINE_PORT}/v1/chat/completions"
H_LOCAL = {"Authorization": f"Bearer {API_KEY}"}
KET_QUA = {}


def do_toc_do(de, max_tokens=300):
    """Đo tốc độ sinh của một lượt stream (token/giây, không tính đoạn đọc đề). Trả (số, chữ mô tả).
    llama-server tự đo (trường timings) thì lấy số đó; máy khác (vLLM) thì tính ở đây:
    (số token - 1) / (giờ nhận token cuối - giờ nhận token đầu)."""
    t_dau = t_cuoi = None
    n_khuc, usage, tm = 0, {}, {}
    mtp0 = DEM_MTP() if globals().get("DEM_MTP") else None
    with requests.post(LOCAL, headers=H_LOCAL, stream=True, timeout=900, json={
            "model": ALIAS, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
            "temperature": 0.7, "messages": [{"role": "user", "content": de}], **KHONG_NGHI}) as s:
        s.raise_for_status()
        for line in s.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:") or "[DONE]" in line:
                continue
            try:
                c = json.loads(line[5:])
            except Exception:
                continue
            usage, tm = c.get("usage") or usage, c.get("timings") or tm
            if any((ch.get("delta") or {}).get("content") for ch in c.get("choices") or []):
                t_cuoi = time.time()
                t_dau = t_dau or t_cuoi
                n_khuc += 1
    n = usage.get("completion_tokens") or n_khuc
    tinh = (n - 1) / (t_cuoi - t_dau) if t_dau and t_cuoi > t_dau and n > 1 else None
    so = tm.get("predicted_per_second") or tinh
    trung, ten_doan = "", globals().get("SPEC_TEN") or "MTP"     # SPEC_TEN: ô 6 vLLM đặt (DSpark, DFlash2, MTP)
    if tm.get("draft_n"):
        trung = f", {ten_doan} đoán trúng {tm['draft_n_accepted']}/{tm['draft_n']}"
    elif mtp0 is not None:
        mtp1 = DEM_MTP()
        if mtp1 and mtp1[1] > mtp0[1]:
            trung = f", {ten_doan} đoán trúng {mtp1[0] - mtp0[0]:.0f}/{mtp1[1] - mtp0[1]:.0f}"
    return so, (f"{so:.1f} token/giây ({n} token{trung})" if so else "không đo được")


# (a) trả lời đúng một chuỗi, không nghĩ
j = hoi([{"role": "user", "content": "Reply with exactly: QWEN_OK"}], max_tokens=32, temperature=0, **KHONG_NGHI)
_txt = (j["choices"][0]["message"].get("content") or "").strip()
assert "QWEN_OK" in _txt, f"Trả lời sai: {_txt!r} | {json.dumps(j)[:400]}"
KET_QUA["trả lời (không nghĩ)"] = f"đạt: {_txt!r}"

# (b) tốc độ sinh chữ: một đoạn code và một đoạn văn tiếng Việt. Lượt sinh dài đầu tiên sau khi nạp còn gánh phần khởi động
# của đường đoán trước (13/10 FableCoder MTP 2: lượt code đo đầu tiên 27,6 token/giây, bài đo có làm nóng 93,0): chạy một lượt nháp.
# Có lượt nháp rồi vẫn có lúc lượt code đầu bị khựng (13/10 13:46: 48,1 trong khi văn đo liền sau 75,0, các lượt khác 91–94):
# đo code hai lần, lấy lượt nhanh hơn, và in cả lượt đầu khi nó chậm hơn từ 20%.
do_toc_do("Đếm từ 1 đến 40, mỗi số một dòng.", max_tokens=96)
_hai_lan = [do_toc_do("Viết lớp Python HangDoi (hàng đợi ưu tiên dùng heapq) có các hàm them, lay, xem, kèm docstring. Chỉ trả code.")
            for _ in range(2)]
TOC_DO, _code = max(_hai_lan, key=lambda x: x[0] or 0)
if _hai_lan[0][0] and TOC_DO and _hai_lan[0][0] < 0.8 * TOC_DO:
    _code += f", lượt đầu chỉ {_hai_lan[0][0]:.1f}"
_, _van = do_toc_do("Viết một đoạn văn khoảng 200 từ giới thiệu Việt Nam.")
KET_QUA["tốc độ sinh"] = f"code {_code} | văn {_van}"

# (c) có nghĩ + stream, mức nghĩ "low". Bật nghĩ rõ ràng:
# model Qwen nhỏ mặc định không nghĩ (13/10 bài thử Qwen3.5-2B trả "85", phần nghĩ 0 ký tự)
try:
    _nghi, _noi = [], []
    with requests.post(LOCAL, headers=H_LOCAL, stream=True, timeout=900, json={
            "model": ALIAS, "stream": True, "max_tokens": 4096,
            "messages": [{"role": "user", "content": "17 nhân 23 bằng bao nhiêu? Trả lời một con số."}],
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}) as s:
        s.raise_for_status()
        for line in s.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:") or "[DONE]" in line:
                continue
            try:
                d = (json.loads(line[5:]).get("choices") or [{}])[0].get("delta", {})
            except Exception:
                continue
            _nghi.append(d.get("reasoning_content") or d.get("reasoning") or "")
            _noi.append(d.get("content") or "")
    _nghi, _noi = "".join(_nghi), "".join(_noi)
    KET_QUA["nghĩ + stream"] = (f"đạt: trả {_noi.strip()[:40]!r}, phần nghĩ {len(_nghi)} ký tự tách riêng" if "391" in _noi
                                else f"SAI: trả {_noi.strip()[:80]!r} (đúng là 391), phần nghĩ {len(_nghi)} ký tự")
except Exception as e:
    KET_QUA["nghĩ + stream"] = f"LỖI {type(e).__name__}: {str(e)[:160]}"

# (d) gọi tool
try:
    j = hoi([{"role": "user", "content": "Thời tiết ở Hà Nội bây giờ thế nào? Hãy gọi công cụ."}], max_tokens=512, **KHONG_NGHI,
            tools=[{"type": "function", "function": {
                "name": "get_weather", "description": "Lấy thời tiết hiện tại của một thành phố",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])
    _tc = j["choices"][0]["message"].get("tool_calls") or []
    KET_QUA["gọi tool"] = (f"đạt: {_tc[0]['function']['name']}({_tc[0]['function']['arguments']})" if _tc
                           else f"KHÔNG gọi tool, trả chữ: {(j['choices'][0]['message'].get('content') or '')[:80]!r}")
except Exception as e:
    KET_QUA["gọi tool"] = f"LỖI {type(e).__name__}: {str(e)[:160]}"

# (e) đọc ảnh: ảnh vẽ số 7391
if DOC_ANH:
    try:
        _txt = hoi_anh()
        KET_QUA["đọc ảnh"] = f"đạt: đọc ra {_txt!r}" if "7391" in _txt else f"SAI: đọc ra {_txt!r} (đúng là 7391)"
    except Exception as e:
        KET_QUA["đọc ảnh"] = f"LỖI {type(e).__name__}: {str(e)[:160]}"
else:
    KET_QUA["đọc ảnh"] = "tắt (VISION = False)"

for k, v in KET_QUA.items():
    print(f"  {k:<22} {v}")
if ENGINE_PROC.poll() is not None:
    raise SystemExit(f"{ENGINE_TEN} đã chết trong lúc tự kiểm (mã {ENGINE_PROC.returncode}). Cuối log:\n"
                     + ENGINE_LOG.read_text(errors="ignore")[-3000:]
                     + "\nThấy 'out of memory': hạ CTX (ví dụ 65536) hoặc đặt VISION = False ở ô 2, rồi chạy lại từ ô 2.")
print("[OK] Model trả lời đúng qua API nội bộ.")
