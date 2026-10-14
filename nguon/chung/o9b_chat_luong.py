# Ô 9b dùng chung (o8_gateway.py chèn nó giữa ô 9 và ô 10). Cần từ các ô trước: LOCAL_GW, H_GW (ô 9), ALIAS, DOC_ANH,
# ENGINE_PROC, gw_proc. Tùy chọn từ ô 2: KIEM_CHAT_LUONG (mặc định True), CL_DAI (cỡ bài ngữ cảnh dài, mặc định 30000 token),
# CL_LAP (mỗi bài code chạy mấy lần, mặc định 3: một lần thì không tách được xui với hỏng thật).
# Chạy riêng ngoài notebook cũng được: py -3 -X utf8 o9b_chat_luong.py với biến môi trường CL_URL (gateway, không có /v1),
# CL_KEY, CL_MODEL, CL_ANH=1/0, CL_LAP; CL_CHAY_CODE=1 mới chạy code model viết (trong notebook thì luôn chạy, máy thuê dùng xong bỏ).

# %% [markdown]
# ### Ô 9b: kiểm chất lượng
# Tự chạy sau khi endpoint đã sẵn sàng (lúc này app đã gọi được). Hỏi qua gateway, đúng đường app gọi: 3 bài code, mỗi bài 3 lần
# (máy tự chạy code model viết với bộ test), 2 bài toán có nghĩ, gọi tool hai lượt, đọc ảnh, tìm một dòng trong ngữ cảnh dài,
# rồi phần làm nội dung: chép phụ đề tiếng Trung trên khung hình (OCR, máy tính tỉ lệ sai chữ), dịch Trung/Anh sang Việt đúng
# xưng hô, dịch một lô phụ đề đánh số, tiêu đề, lời mở đầu, lời bình shorts, mô tả, một câu hỏi phải biết nói "không biết".
# Phần chữ in ra cuối để đọc chấm. Mất khoảng 5–15 phút tùy máy. Không muốn chạy thì đặt `KIEM_CHAT_LUONG = False` ở ô 2.
# Ô này hỏng ở bài nào thì chỉ ghi vào bảng, không làm dừng ô 10.

# %%
# 9b) Kiểm chất lượng qua gateway
import base64, hashlib, io, json, os, random, re, subprocess, sys, tempfile, time, unicodedata
from pathlib import Path

import requests

_g = globals()
CL_URL = (_g.get("LOCAL_GW") or os.environ.get("CL_URL", "http://127.0.0.1:8787")).rstrip("/")
CL_H = _g.get("H_GW") or {"Authorization": f"Bearer {os.environ.get('CL_KEY', '')}"}
CL_MODEL = _g.get("ALIAS") or os.environ.get("CL_MODEL", "qwen3.8-27b")
CL_ANH = _g["DOC_ANH"] if "DOC_ANH" in _g else os.environ.get("CL_ANH", "1") == "1"
CL_CHAY_CODE = ("ENGINE_PROC" in _g) or os.environ.get("CL_CHAY_CODE") == "1"
CL_DAI = int(_g.get("CL_DAI") or os.environ.get("CL_DAI", "30000"))
CL_LAP = int(_g.get("CL_LAP") or os.environ.get("CL_LAP", "3"))
CJK = re.compile(r"[぀-ヿ一-鿿가-힯]")


def goi(messages, nghi=False, muc="low", max_tokens=2048, tools=None):
    """Một lượt stream qua gateway. Trả dict: noi, nghi, tool_calls, n (token sinh), giay, tok_s, tok_de, cho_dau."""
    body = {"model": CL_MODEL, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": muc} if nghi else {"enable_thinking": False}}
    if tools:
        body["tools"] = tools
    noi, ng, tcs, usage = [], [], {}, {}
    t0 = time.time()
    t_dau = t_cuoi = None
    with requests.post(CL_URL + "/v1/chat/completions", headers=CL_H, json=body, stream=True, timeout=1800) as s:
        s.raise_for_status()
        for line in s.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:") or "[DONE]" in line:
                continue
            try:
                c = json.loads(line[5:])
            except Exception:
                continue
            usage = c.get("usage") or usage
            for ch in c.get("choices") or []:
                d = ch.get("delta") or {}
                for tc in d.get("tool_calls") or []:
                    o = tcs.setdefault(tc.get("index", 0), {"name": "", "arguments": "", "id": tc.get("id")})
                    f = tc.get("function") or {}
                    o["name"] += f.get("name") or ""
                    o["arguments"] += f.get("arguments") or ""
                    o["id"] = o["id"] or tc.get("id")
                manh = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if manh or d.get("tool_calls"):
                    t_cuoi = time.time()
                    t_dau = t_dau or t_cuoi
                noi.append(d.get("content") or "")
                ng.append(d.get("reasoning_content") or d.get("reasoning") or "")
    n = usage.get("completion_tokens") or 0
    sinh = (t_cuoi - t_dau) if t_dau and t_cuoi else 0
    return {"noi": "".join(noi).strip(), "nghi": "".join(ng), "tool_calls": [tcs[k] for k in sorted(tcs)], "n": n,
            "giay": time.time() - t0, "tok_s": (n / sinh) if n and sinh > 0 else 0, "tok_de": usage.get("prompt_tokens"),
            "cho_dau": (t_dau - t0) if t_dau else None}


def lay_code(txt):
    khoi = re.findall(r"```(?:python|py)?\s*\n(.*?)```", txt, re.S)
    return max(khoi, key=len) if khoi else txt


def chay_code(code, test):
    if not CL_CHAY_CODE:
        return None, "không chạy (đặt CL_CHAY_CODE=1)"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bai.py"
        p.write_text(code + "\n\n" + test, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], capture_output=True, text=True, timeout=30, cwd=d)
        except subprocess.TimeoutExpired:
            return False, "quá 30 giây"
        return r.returncode == 0, (r.stdout + r.stderr).strip()[-300:]


BAI_CODE = [
    ("code gộp khoảng", False,
     "Viết hàm Python `gop_khoang(ds)` nhận danh sách các khoảng [a, b] (số nguyên, a <= b, chưa sắp xếp) và trả danh sách "
     "các khoảng [a, b] sau khi gộp những khoảng chồng nhau hoặc chạm nhau, sắp tăng dần. Chỉ trả một khối code.",
     "f = lambda x: [list(k) for k in gop_khoang(x)]\n"
     "assert f([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\nassert f([[1,4],[4,5]]) == [[1,5]]\n"
     "assert f([]) == []\nassert f([[5,7],[1,2],[2,3]]) == [[1,3],[5,7]]\nassert f([[1,10],[2,3]]) == [[1,10]]\nprint('OK')"),
    ("code LRU cache", False,
     "Viết lớp Python `LRUCache(capacity)` có `get(key)` (trả -1 nếu không có) và `put(key, value)`, cả hai O(1); khi đầy thì "
     "bỏ phần tử lâu chưa dùng nhất. Chỉ trả một khối code.",
     "c = LRUCache(2); c.put(1, 1); c.put(2, 2); assert c.get(1) == 1; c.put(3, 3); assert c.get(2) == -1\n"
     "c.put(4, 4); assert c.get(1) == -1; assert c.get(3) == 3; assert c.get(4) == 4\n"
     "d = LRUCache(2); d.put(1, 1); d.put(2, 2); d.put(1, 10); d.put(3, 3); assert d.get(1) == 10; assert d.get(2) == -1\nprint('OK')"),
    ("code tính biểu thức (có nghĩ)", True,
     "Viết hàm Python `tinh(bt: str) -> float` tính giá trị biểu thức số học gồm số thực, + - * /, ngoặc đơn, dấu cách và dấu "
     "trừ một ngôi, đúng thứ tự ưu tiên. Không dùng eval, exec, compile hay thư viện ngoài. Chỉ trả một khối code.",
     "for bt, kq in [('2+3*4', 14), ('(1+2)*-3', -9), ('10/4', 2.5), ('-(2+3)*2', -10), (' 7 - 2 - 1 ', 4),\n"
     "               ('2*(3+(4-1))/3', 4), ('1.5*2', 3), ('--3', 3), ('8/2/2', 2)]:\n"
     "    assert abs(tinh(bt) - kq) < 1e-9, (bt, tinh(bt), kq)\nprint('OK')"),
]
BAI_TOAN = [
    ("toán xếp hàng (có nghĩ)", "Có bao nhiêu cách xếp 5 người A, B, C, D, E thành một hàng sao cho A và B không đứng cạnh nhau? "
     "Dòng cuối chỉ ghi đáp số.", "72"),
    ("toán vòi nước (có nghĩ)", "Một bể có hai vòi chảy vào: vòi 1 chảy riêng đầy bể sau 6 giờ, vòi 2 sau 4 giờ; vòi xả chảy riêng "
     "cạn bể đầy sau 12 giờ. Bể đang cạn, mở cả ba vòi cùng lúc thì sau bao nhiêu giờ đầy bể? Dòng cuối chỉ ghi đáp số.", "3"),
]
CONG_CU = [
    {"type": "function", "function": {"name": "get_weather", "description": "Lấy nhiệt độ hiện tại (độ C) của một thành phố",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "doi_tien", "description": "Đổi một số tiền từ loại tiền này sang loại khác theo tỷ giá hôm nay",
                                      "parameters": {"type": "object", "properties": {"so_tien": {"type": "number"}, "tu": {"type": "string"},
                                                                                      "sang": {"type": "string"}}, "required": ["so_tien", "tu", "sang"]}}}]

# OCR phụ đề: khung hình vẽ tại chỗ (nền rối nhòe, chữ đậm viền đen như phụ đề cứng, lưu JPEG như khung video), đáp án là chữ
# đã vẽ. Font Noto Sans SC (giấy phép OFL) ghim theo commit của google/fonts, kiểm sha256; không tải được thì bỏ phần OCR.
FONT_TRUNG = ("https://raw.githubusercontent.com/google/fonts/2894aab31764f10f29c421bdfd2340d3b382d384/ofl/notosanssc/"
              "NotoSansSC%5Bwght%5D.ttf", "a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da")
# (tên, cỡ khung, các dòng phụ đề, cỡ chữ, màu, chữ lạ ở trên cùng)
BAI_OCR = [
    ("ngang, chữ trắng", (1280, 720), ["妈，你别担心，我下个月就回家看你。"], 44, "white", None),
    ("cổ trang, chữ vàng", (1280, 720), ["王爷，属下办事不力，请您责罚！"], 40, (255, 222, 60), None),
    ("dọc, hai dòng, có tiêu đề trên", (720, 1280), ["这台扫地机器人把小哈", "吓得直往沙发底下钻"], 40, "white", "我的哈士奇变成美女了 第13集"),
    ("tên game, dấu lạ", (1280, 720), ["旅行者，你终于来了！", "派蒙等你好久了～"], 42, "white", None),
    ("chữ nhỏ, có số", (1280, 720), ["芯片短缺让全球汽车厂商损失惨重，预计2026年才会缓解"], 26, (255, 230, 80), None),
    ("phồn thể", (1280, 720), ["這個問題我們明天再討論吧"], 36, "white", None),
]
# Dịch từng câu: (ngữ cảnh, câu gốc, các nhóm phải có (regex, không phân biệt hoa thường), regex cấm). Máy chỉ kiểm thô;
# câu dịch in ra cuối để đọc chấm.
BAI_DICH = [
    ("phim gia đình, con trai nói với mẹ qua điện thoại", "妈，你别担心，我下个月就回家看你。",
     [r"\bmẹ\b", r"\bcon\b"], r"\b(tôi|bạn)\b"),
    ("phim cổ trang, thị vệ quỳ trước vương gia", "王爷，属下办事不力，请您责罚。",
     [r"vương gia", r"thuộc hạ"], r"\b(tôi|bạn)\b"),
    ("phim tổng tài, thư ký báo cáo với sếp", "总裁，这份合同对方已经签字了。",
     [r"tổng tài|tổng giám đốc|giám đốc|sếp", r"hợp đồng", r"\bký\b|\bkí\b"], None),
    ("video khoa học phổ thông, câu mở đầu", "一直往地底下挖，会发现什么？", [r"đào", r"đất"], None),
    ("video AI: chú husky tên Tiểu Ha biến thành cô gái", "小哈最怕扫地机器人了！",
     [r"tiểu ha", r"robot|rô-bốt|máy hút bụi|hút bụi", r"sợ"], None),
    ("game Genshin Impact", "旅行者，你终于来了，派蒙等你好久了！", [r"lữ hành|lữ khách", r"paimon"], None),
    ("phim, hai người đàn ông cãi nhau gay gắt", "你给我等着，这笔账我迟早跟你算！",
     [r"chờ|đợi", r"tính|thanh toán|trả"], r"\bbạn\b"),
    ("tin công nghệ", "芯片短缺让全球汽车厂商损失惨重。", [r"chip", r"ô tô|xe hơi", r"thiếu"], None),
    ("reels hài, cô gái nói với bạn trai", "I promise I'm always this attentive.", [r"hứa"], None),
    ("reels cosplay Wuthering Waves, bị chê giữa công viên", "I'm getting flamed in public now, all I wanted was a peaceful "
     "afternoon at the park.", [r"công viên", r"chiều"], None),
]
BAI_DICH_LO = ["儿子，你怎么又这么晚才回来？", "妈，公司临时加班，我也没办法。", "你看看你，瘦成什么样了。",
               "我没事，您别操心了。", "饭在锅里热着呢，快去吃吧。", "谢谢妈，还是家里的饭最香。"]
# Việc làm nội dung: (tên, đề, hàm kiểm thô trên câu trả lời -> (đạt, ghi chú))
_dong = lambda t: [d.strip(" -*•\t") for d in t.splitlines() if d.strip(" -*•\t")]
BAI_NOI_DUNG = [
    ("tiêu đề YouTube", "Viết 8 tiêu đề YouTube tiếng Việt cho video giải thích: pin natri có thay được pin lithium trên ô tô điện "
     "không. Mỗi tiêu đề dưới 60 ký tự, gây tò mò nhưng không nói sai sự thật. Mỗi dòng một tiêu đề, không đánh số, không giải thích.",
     lambda t: (len(_dong(t)) >= 8 and all(len(d) <= 75 for d in _dong(t)[:8]), f"{len(_dong(t))} dòng, dài nhất "
                f"{max((len(d) for d in _dong(t)), default=0)} ký tự")),
    ("lời mở đầu 30 giây", "Viết lời mở đầu 30 giây (70–100 từ) cho video 'Sẽ ra sao nếu Trái Đất ngừng quay trong 1 giây'. "
     "Câu đầu phải gây sốc bằng một cảnh cụ thể. Chỉ trả lời bằng lời đọc.",
     lambda t: (50 <= len(t.split()) <= 140, f"{len(t.split())} từ")),
    ("lời bình shorts game", "Viết 3 câu bình luận hài, mỗi câu dưới 15 từ, để chèn vào shorts game Genshin Impact: Furina và "
     "Nahida nhìn nhau rồi cùng quay mặt đi. Mỗi dòng một câu, không đánh số.",
     lambda t: (len(_dong(t)) == 3 and all(len(d.split()) <= 18 for d in _dong(t)), f"{len(_dong(t))} dòng")),
    ("mô tả + thẻ", "Viết mô tả YouTube khoảng 200 ký tự, có từ khóa, cho video giải thích pin natri. Dòng cuối là đúng 5 thẻ, "
     "cách nhau bằng dấu phẩy. Không dùng hashtag.",
     lambda t: (bool(_dong(t)) and len(_dong(t)[-1].split(",")) == 5 and "#" not in t and len(t) <= 600,
                f"{len(t)} ký tự, dòng cuối {len(_dong(t)[-1].split(',')) if _dong(t) else 0} thẻ")),
    ("biết nói không biết", "Doanh thu của cửa hàng Bách Hóa Xanh ở phường Bến Nghé, TP.HCM ngày 3/9/2026 là bao nhiêu?",
     lambda t: (bool(re.search(r"không (có|biết|thể|nắm|truy cập)|chưa có|không rõ", t.lower())), "phải nói không biết, không bịa số")),
]


def font_trung():
    p = Path(tempfile.gettempdir()) / "NotoSansSC-wght.ttf"
    if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == FONT_TRUNG[1]:
        return p
    r = requests.get(FONT_TRUNG[0], timeout=180)
    r.raise_for_status()
    if hashlib.sha256(r.content).hexdigest() != FONT_TRUNG[1]:
        raise RuntimeError("mã băm font không khớp")
    p.write_bytes(r.content)
    return p


def khung_phu_de(font, co_khung, dong, co, mau, tren, seed):
    """Một khung hình giả: nền chuyển màu + vòng tròn màu nhòe, phụ đề đậm viền đen ở dưới, trả data URL JPEG."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    rnd = random.Random(seed)
    w, h = co_khung
    img = Image.new("RGB", (w, h))
    dr = ImageDraw.Draw(img)
    c1, c2 = [tuple(rnd.randint(30, 220) for _ in range(3)) for _ in range(2)]
    for y in range(h):
        dr.line([(0, y), (w, y)], fill=tuple(int(a + (b - a) * y / h) for a, b in zip(c1, c2)))
    for _ in range(30):
        x, y, r = rnd.randint(0, w), rnd.randint(0, h), rnd.randint(20, 220)
        dr.ellipse((x - r, y - r, x + r, y + r), fill=tuple(rnd.randint(0, 255) for _ in range(3)))
    img = img.filter(ImageFilter.GaussianBlur(5))
    dr = ImageDraw.Draw(img)
    f = ImageFont.truetype(str(font), co)
    try:
        f.set_variation_by_name("Bold")
    except Exception:
        pass
    y = h - (h // 7 if h > w else 40) - (co + 12) * len(dong)
    for d in dong:
        b = dr.textbbox((0, 0), d, font=f, stroke_width=3)
        dr.text(((w - (b[2] - b[0])) // 2, y), d, font=f, fill=mau, stroke_width=3, stroke_fill="black")
        y += co + 12
    if tren:
        f2 = ImageFont.truetype(str(font), 34)
        b = dr.textbbox((0, 0), tren, font=f2)
        dr.text(((w - (b[2] - b[0])) // 2, 60), tren, font=f2, fill="white", stroke_width=2, stroke_fill="black")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def chuan_chu(s):
    """Bỏ dấu câu, khoảng trắng; giữ chữ Hán và chữ số, chữ cái Latin (đổi chữ toàn khổ về thường)."""
    s = unicodedata.normalize("NFKC", s)
    return "".join(ch for ch in s if CJK.match(ch) or (ch.isascii() and ch.isalnum()))


def ti_le_sai(doan, dung):
    """Tỉ lệ sai chữ (CER): khoảng cách sửa chữ / số chữ đáp án."""
    a, b = chuan_chu(doan), chuan_chu(dung)
    truoc = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        hien = [i]
        for j, cb in enumerate(b, 1):
            hien.append(min(truoc[j] + 1, hien[j - 1] + 1, truoc[j - 1] + (ca != cb)))
        truoc = hien
    return truoc[-1] / max(len(b), 1)


def kiem_chat_luong():
    KQ, CHU = [], []

    def ghi(muc, dat, chi_tiet, r=None):
        toc = f" | {r['n']} token, {r['tok_s']:.1f} token/giây, {r['giay']:.1f} giây" if r and r.get("n") else ""
        KQ.append((muc, dat))
        print(f"  {muc:<32} {'ĐẠT' if dat else ('—' if dat is None else 'SAI')}  {chi_tiet}{toc}", flush=True)

    for _ten in ("ENGINE_PROC", "gw_proc"):
        if _g.get(_ten) is not None and _g[_ten].poll() is not None:
            print(f"{_ten} đã tắt: chạy lại từ ô 6 rồi mới kiểm chất lượng.")
            return
    print(f"Kiểm chất lượng {CL_MODEL} qua {CL_URL} (đọc ảnh: {'có' if CL_ANH else 'không'}, chạy code: {'có' if CL_CHAY_CODE else 'không'})",
          flush=True)
    for muc, nghi, de, test in BAI_CODE:
        lan, loi, r = [], "", None
        for _ in range(CL_LAP):
            try:
                r = goi([{"role": "user", "content": de}], nghi=nghi, muc="medium", max_tokens=12000 if nghi else 3000)
                code = lay_code(r["noi"])
                if re.search(r"\b(eval|exec|compile)\s*\(", code) and "tinh" in de:
                    ok, out = False, "dùng eval/exec dù đề cấm"
                else:
                    ok, out = chay_code(code, test)
            except Exception as e:
                ok, out = False, f"LỖI {type(e).__name__}: {str(e)[:150]}"
            lan.append(ok)
            loi = loi or ("" if ok else out)
        so = sum(1 for x in lan if x)
        ghi(muc, None if all(x is None for x in lan) else so * 2 > len(lan),
            f"đúng {so}/{len(lan)} lần" + (f"; lần hỏng đầu: {loi[-140:]!r}" if loi else ""), r)
    for muc, de, dap in BAI_TOAN:
        try:
            r = goi([{"role": "user", "content": de}], nghi=True, muc="low", max_tokens=12000)
            dong = r["noi"].splitlines()[-1] if r["noi"] else ""
            so = re.findall(r"-?\d+(?:[.,]\d+)?", dong)
            ghi(muc, bool(so) and so[-1].replace(",", ".") in (dap, dap + ".0"),
                f"đáp {dong[:60]!r} (đúng {dap}), nghĩ {len(r['nghi'])} ký tự" if dong else "không trả lời", r)
        except Exception as e:
            ghi(muc, False, f"LỖI {type(e).__name__}: {str(e)[:150]}")

    # gọi tool hai lượt: model gọi tool, nhận kết quả, rồi trả lời bằng số liệu đó
    try:
        msgs = [{"role": "user", "content": "Hà Nội đang bao nhiêu độ, và 100 USD đổi được bao nhiêu VND? Dùng công cụ để tra."}]
        goi_ten, r = [], None
        for _ in range(4):
            r = goi(msgs, max_tokens=1500, tools=CONG_CU)
            if not r["tool_calls"]:
                break
            msgs.append({"role": "assistant", "content": r["noi"] or None, "tool_calls": [
                {"id": tc["id"] or f"call_{i}", "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for i, tc in enumerate(r["tool_calls"])]})
            for i, tc in enumerate(r["tool_calls"]):
                goi_ten.append(tc["name"])
                kq = {"get_weather": {"city": "Hà Nội", "nhiet_do_c": 31},
                      "doi_tien": {"ket_qua": 2617500, "loai": "VND"}}.get(tc["name"], {"loi": "không có công cụ này"})
                msgs.append({"role": "tool", "tool_call_id": tc["id"] or f"call_{i}", "content": json.dumps(kq, ensure_ascii=False)})
        cuoi = (r or {}).get("noi", "")
        dat = {"get_weather", "doi_tien"} <= set(goi_ten) and "31" in cuoi and "2617500" in re.sub(r"[.,\s]", "", cuoi)
        ghi("gọi tool hai lượt", dat, f"đã gọi {goi_ten}; trả: {cuoi[:90]!r}", r)
    except Exception as e:
        ghi("gọi tool hai lượt", False, f"LỖI {type(e).__name__}: {str(e)[:150]}")

    # đọc ảnh: hình và chữ
    if CL_ANH:
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new("RGB", (800, 400), "white")
            dr = ImageDraw.Draw(img)
            dr.ellipse((60, 110, 240, 290), fill=(220, 30, 30))
            dr.rectangle((560, 110, 740, 290), fill=(30, 60, 220))
            try:
                font = ImageFont.load_default(size=56)
            except TypeError:
                font = ImageFont.load_default()
            dr.text((300, 170), "APPLE 42", fill="black", font=font)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            r = goi([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text":
                     "Mô tả ngắn ảnh này: có những hình gì, màu gì, nằm bên trái hay bên phải, và chữ trong ảnh là gì?"}]}], max_tokens=600)
            t = r["noi"].lower()
            dat = ("đỏ" in t or "red" in t) and ("xanh" in t or "blue" in t) and "42" in t and "apple" in t
            ghi("đọc ảnh (hình, màu, chữ)", dat, f"{r['noi'][:140]!r}", r)
        except Exception as e:
            ghi("đọc ảnh (hình, màu, chữ)", False, f"LỖI {type(e).__name__}: {str(e)[:150]}")

    # ngữ cảnh dài: tìm một dòng giữa khoảng CL_DAI token
    try:
        dong = [f"Dòng {i}: kho số {i % 50} nhận {i * 7 % 1000} thùng hàng vào ngày {i % 28 + 1}." for i in range(CL_DAI // 22)]
        dong.insert(int(len(dong) * 0.6), "Ghi chú quan trọng: mật mã mở cửa kho số 77 là 8137.")
        r = goi([{"role": "user", "content": "\n".join(dong) + "\n\nMật mã mở cửa kho số 77 là gì? Chỉ trả lời con số."}], max_tokens=64)
        ghi(f"ngữ cảnh dài (~{CL_DAI // 1000}K token)", "8137" in r["noi"],
            f"trả {r['noi'][:40]!r}, {r['tok_de']} token đề, chờ chữ đầu {r['cho_dau'] or 0:.1f} giây", r)
    except Exception as e:
        ghi(f"ngữ cảnh dài (~{CL_DAI // 1000}K token)", False, f"LỖI {type(e).__name__}: {str(e)[:150]}")

    # văn tiếng Việt: máy chỉ kiểm độ dài, dấu, chữ Hán lẫn vào; hay dở do người đọc
    try:
        r = goi([{"role": "user", "content": "Viết một đoạn văn khoảng 150 từ, giọng gần gũi, giải thích cho bố mẹ lớn tuổi vì sao nên "
                  "bật xác thực hai lớp cho tài khoản Facebook và cách bật."}], max_tokens=1200)
        so_tu = len(r["noi"].split())
        dau = len(re.findall(r"[ạảấầẩẫậắằẳẵặẹẻẽếềểễệịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹđ]", r["noi"].lower()))
        dat = 90 <= so_tu <= 260 and not CJK.search(r["noi"]) and dau > 20
        ghi("văn tiếng Việt", dat, f"{so_tu} từ, {'có' if CJK.search(r['noi']) else 'không'} lẫn chữ Hán", r)
        CHU.append(("Đoạn văn tiếng Việt (tự đọc chấm)", r["noi"]))
    except Exception as e:
        ghi("văn tiếng Việt", False, f"LỖI {type(e).__name__}: {str(e)[:150]}")

    # ---- Phần làm nội dung ----
    print("\nLàm nội dung (OCR phụ đề, dịch, viết):", flush=True)
    if CL_ANH:
        try:
            font = font_trung()
        except Exception as e:
            font = None
            ghi("OCR phụ đề", None, f"bỏ qua: không tải được font chữ Trung ({type(e).__name__}: {str(e)[:100]})")
        for i, (ten, co_khung, dong, co, mau, tren) in enumerate(BAI_OCR if font else []):
            muc = f"OCR: {ten}"
            try:
                url = khung_phu_de(font, co_khung, dong, co, mau, tren, seed=i + 1)
                r = goi([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text":
                         "Ảnh là một khung hình video. Chép nguyên văn phụ đề tiếng Trung ở phần dưới khung hình (bỏ qua chữ ở chỗ "
                         "khác). Giữ đúng chữ giản thể hay phồn thể như trong ảnh, không dịch, không giải thích, chỉ trả về chữ đó."}]}],
                        max_tokens=200)
                doc = "".join(l for l in r["noi"].splitlines() if CJK.search(l))
                sai = ti_le_sai(doc, "".join(dong))
                ghi(muc, sai <= 0.05, f"sai {sai:.0%} chữ; đọc {doc[:40]!r}", r)
            except Exception as e:
                ghi(muc, False, f"LỖI {type(e).__name__}: {str(e)[:150]}")

    dich = []
    for ngu_canh, cau, can, cam in BAI_DICH:
        muc = f"dịch: {ngu_canh.split(',')[0][:22]}"
        try:
            r = goi([{"role": "user", "content": f"Dịch câu phụ đề sau sang tiếng Việt tự nhiên, đúng xưng hô theo ngữ cảnh.\n"
                      f"Ngữ cảnh: {ngu_canh}.\nCâu: {cau}\nChỉ trả về câu dịch."}], max_tokens=300)
            t = r["noi"].strip()
            thieu = [c for c in can if not re.search(c, t, re.I)]
            pham = bool(cam and re.search(cam, t, re.I))
            ghi(muc, not thieu and not pham and not CJK.search(t),
                ("thiếu " + ", ".join(thieu) + "; " if thieu else "") + ("sai xưng hô; " if pham else "")
                + ("lẫn chữ Hán; " if CJK.search(t) else "") + f"{t[:70]!r}", r)
            dich.append(f"{cau}\n  → {t}")
        except Exception as e:
            ghi(muc, False, f"LỖI {type(e).__name__}: {str(e)[:150]}")
    try:
        de = "\n".join(f"{i}. {c}" for i, c in enumerate(BAI_DICH_LO, 1))
        r = goi([{"role": "user", "content": "Dịch lô phụ đề sau sang tiếng Việt (phim gia đình: mẹ và con trai đã đi làm). Giữ đúng "
                  f"{len(BAI_DICH_LO)} dòng, mỗi dòng bắt đầu bằng số thứ tự như bản gốc, đúng xưng hô mẹ – con, không thêm gì khác.\n\n{de}"}],
                max_tokens=800)
        so_dong = re.findall(r"(?m)^\s*(\d+)[.)]\s*\S", r["noi"])
        dat = so_dong == [str(i) for i in range(1, len(BAI_DICH_LO) + 1)] and not CJK.search(r["noi"]) \
            and not re.search(r"\b(tôi|bạn)\b", r["noi"], re.I)
        ghi("dịch lô 6 dòng đánh số", dat, f"ra {len(so_dong)} dòng đánh số", r)
        dich.append(f"[lô 6 dòng]\n{r['noi']}")
    except Exception as e:
        ghi("dịch lô 6 dòng đánh số", False, f"LỖI {type(e).__name__}: {str(e)[:150]}")
    if dich:
        CHU.append(("Câu dịch (tự đọc chấm)", "\n".join(dich)))

    for ten, de, kiem in BAI_NOI_DUNG:
        try:
            r = goi([{"role": "user", "content": de}], max_tokens=900)
            dat, ghi_chu = kiem(r["noi"])
            ghi(ten, dat and not CJK.search(r["noi"]), ghi_chu + ("; lẫn chữ Hán" if CJK.search(r["noi"]) else ""), r)
            CHU.append((ten.capitalize() + " (tự đọc chấm)", r["noi"]))
        except Exception as e:
            ghi(ten, False, f"LỖI {type(e).__name__}: {str(e)[:150]}")

    cham = [d for _, d in KQ if d is not None]
    print(f"\nMáy chấm: đạt {sum(cham)}/{len(cham)} mục.")
    for tieu_de, chu in CHU:
        print(f"\n--- {tieu_de} ---\n{chu}")


if _g.get("KIEM_CHAT_LUONG", True):
    try:
        kiem_chat_luong()
    except Exception as e:      # ô này hỏng không được làm dừng ô 10 (giữ phiên) khi bấm Run all
        print(f"Kiểm chất lượng dừng giữa chừng: {type(e).__name__}: {str(e)[:200]}")
else:
    print("Bỏ qua kiểm chất lượng (KIEM_CHAT_LUONG = False ở ô 2).")
