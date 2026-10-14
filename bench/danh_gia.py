"""Đo một model qua endpoint chuẩn OpenAI (khung "ENDPOINT SẴN SÀNG" của các notebook trong thư mục này).

Dùng (từ thư mục cha của bench/):
    py -3 -X utf8 bench/danh_gia.py --endpoint <file.json> --may "<mô tả máy>" toc_do
    py -3 -X utf8 bench/danh_gia.py --endpoint <file.json> --may "<mô tả máy>" gsm8k [--gioi-han 200] [--nghi]
    py -3 -X utf8 bench/danh_gia.py --endpoint <file.json> --may "<mô tả máy>" humaneval [--nghi] [--chay-bang wsl]
File endpoint: {"baseUrl": "https://....trycloudflare.com", "token": "gw-...", "model": "qwen3.8-flash-next"}. Đặt tên file
có chữ "secret" và để ngoài repo. Kết quả ghi ở bench/ket_qua/<ngày>_<model>/: <bài>.json (tổng) + <bài>.jsonl (từng câu),
không chứa địa chỉ hay khóa.

Dữ liệu tải lần đầu vào bench/du_lieu/: GSM8K test (openai/grade-school-math, MIT), HumanEval (openai/human-eval, MIT).
HumanEval chạy code do model viết: nên chạy trong WSL hay máy ảo (--chay-bang wsl, hoặc một lệnh python khác).
Chỉ dùng thư viện chuẩn.
"""
import argparse
import concurrent.futures as cf
import gzip
import json
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

GOC = Path(__file__).resolve().parent
DU_LIEU = GOC / "du_lieu"
NGUON_DU_LIEU = {
    "gsm8k_test.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "HumanEval.jsonl.gz": "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
}


class Ban(Exception):
    """Gateway báo hết chỗ (429 gateway_busy): chờ rồi gửi lại."""


def goi(ep, messages, max_tokens, nghi=False, temperature=0.0, timeout=900, lan_thu=3, cho_ban=60):
    """Một lượt chat có stream. Trả chữ, phần nghĩ, số token vào/ra, giây chờ chữ đầu, tốc độ giải mã (token/giây).
    Gateway bận (mọi chỗ chạy đều có lượt, ví dụ app đang dùng chung endpoint) thì chờ 10 giây rồi gửi lại, tối đa cho_ban lần."""
    body = {"model": ep["model"], "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": nghi}}
    req = urllib.request.Request(ep["baseUrl"] + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Authorization": "Bearer " + ep["token"], "Content-Type": "application/json",
                                          "User-Agent": "Mozilla/5.0"})
    ban = lan = 0
    while True:
        try:
            t0 = time.perf_counter()
            dau = cuoi = None
            chu, nghi_txt, usage = [], [], {}
            with urllib.request.urlopen(req, timeout=timeout) as r:
                for dong in r:
                    dong = dong.strip()
                    if not dong.startswith(b"data:") or dong == b"data: [DONE]":
                        continue
                    o = json.loads(dong[5:])
                    if o.get("error"):
                        e = o["error"] if isinstance(o["error"], dict) else {}
                        if e.get("code") == 429 or e.get("type") == "gateway_busy":
                            raise Ban()
                        raise RuntimeError(f"endpoint báo lỗi: {str(o['error'])[:300]}")
                    usage = o.get("usage") or usage
                    for c in o.get("choices") or []:
                        d = c.get("delta") or {}
                        r_ = d.get("reasoning_content") or d.get("reasoning")
                        if r_ or d.get("content"):
                            cuoi = time.perf_counter()
                            dau = dau or cuoi
                        if r_:
                            nghi_txt.append(r_)
                        if d.get("content"):
                            chu.append(d["content"])
            ra = usage.get("completion_tokens", 0)
            return {"chu": "".join(chu), "nghi": "".join(nghi_txt), "vao": usage.get("prompt_tokens", 0), "ra": ra,
                    "cho_dau": (dau - t0) if dau else None, "tong": time.perf_counter() - t0,
                    "giai_ma": (ra - 1) / (cuoi - dau) if dau and cuoi > dau and ra > 1 else None}
        except (Ban, urllib.error.HTTPError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code != 429:
                raise
            ban += 1
            if ban > cho_ban:
                raise RuntimeError("gateway bận quá lâu") from e
            time.sleep(10)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            lan += 1
            if lan >= lan_thu:
                raise
            print(f"  lỗi mạng ({type(e).__name__}), thử lại sau 5 giây", flush=True)
            time.sleep(5)


def tai_du_lieu(ten):
    p = DU_LIEU / ten
    if not p.exists():
        DU_LIEU.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(NGUON_DU_LIEU[ten], headers={"User-Agent": "Mozilla/5.0"}),
                                    timeout=120) as r:
            p.write_bytes(r.read())
    data = p.read_bytes()
    if ten.endswith(".gz"):
        data = gzip.decompress(data)
    return [json.loads(l) for l in data.decode("utf-8").splitlines() if l.strip()]


def trung_vi(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


# ---------------------------------------------------------------- tốc độ
VIEC = {
    "code (Python)": "Write a Python implementation of an LRU cache class with get/put, type hints and docstrings, "
                     "followed by 6 unittest test cases. Code only.",
    "văn tiếng Việt": "Viết một bài khoảng 600 chữ tiếng Việt giải thích cho học sinh cấp 3 vì sao pin natri có thể "
                      "thay pin lithium trong xe điện giá rẻ.",
    "văn tiếng Anh": "Write about 600 words explaining to high-school students how a CPU cache works.",
    "dịch Trung → Việt": "Dịch sang tiếng Việt tự nhiên, giữ nghĩa:\n" + "芯片短缺让全球汽车厂商损失惨重，预计要到明年下半年才会"
                         "缓解。业内人士表示，新工厂的建设周期长，短期内难以弥补产能缺口。与此同时，部分厂商开始重新设计电路板，"
                         "以使用更容易买到的芯片。" * 3,
}


def de_dai(so_token, nonce):
    """Đề dài khoảng so_token token: các dòng sổ kho, giấu một mã ở giữa. nonce ở đầu để không trúng bộ nhớ đệm đề."""
    n = max(1, so_token // 36)
    dong = [f"Phiên {nonce}."] + [f"Mục {i}: kho số {i * 7 % 1000} nhận {(i * 37) % 900 + 100} thùng hàng, "
                                  f"người trực số {(i * 13) % 50}." for i in range(n)]
    dong.insert(len(dong) // 2, "Ghi chú đặc biệt: mã mở két của kho trung tâm là 50417.")
    return [{"role": "user", "content": "\n".join(dong) + "\n\nMã mở két của kho trung tâm là bao nhiêu? Chỉ trả lời con số."}]


def toc_do(ep, a):
    kq = {"viec": {}, "de_dai": [], "cung_luc": []}
    goi(ep, [{"role": "user", "content": "Xin chào"}], 16)                       # lượt nháp
    print("Tốc độ theo loại việc (không nghĩ, 3 lần, lấy trung vị):")
    for ten, de in VIEC.items():
        lan = [goi(ep, [{"role": "user", "content": de}], 1024, temperature=0.7) for _ in range(3)]
        kq["viec"][ten] = {"giai_ma": trung_vi([l["giai_ma"] for l in lan]), "cho_dau": trung_vi([l["cho_dau"] for l in lan]),
                           "ra": trung_vi([l["ra"] for l in lan])}
        print(f"  {ten:20s} {kq['viec'][ten]['giai_ma']} token/giây, chờ chữ đầu {kq['viec'][ten]['cho_dau']} giây")
    lan = [goi(ep, [{"role": "user", "content": "Một bể nước có hai vòi: vòi A chảy đầy bể trong 6 giờ, vòi B trong 9 giờ, "
                                                "vòi C tháo cạn bể trong 12 giờ. Mở cả ba cùng lúc thì bao lâu đầy bể?"}],
               4096, nghi=True, temperature=0.6) for _ in range(3)]
    kq["viec"]["toán (có nghĩ)"] = {"giai_ma": trung_vi([l["giai_ma"] for l in lan]), "cho_dau": trung_vi([l["cho_dau"] for l in lan]),
                                    "ra": trung_vi([l["ra"] for l in lan])}
    print(f"  {'toán (có nghĩ)':20s} {kq['viec']['toán (có nghĩ)']['giai_ma']} token/giây")

    print("Đề dài (lần đầu, rồi gửi lại y hệt để đo bộ nhớ đệm đề):")
    for n in a.de_dai:
        msgs = de_dai(n, random.randrange(10**9))
        r1 = goi(ep, msgs, 16)
        r2 = goi(ep, msgs, 16)
        dong = {"token_de": r1["vao"], "cho_dau": round(r1["cho_dau"], 2), "nap_de": round(r1["vao"] / r1["cho_dau"]),
                "cho_dau_lan_2": round(r2["cho_dau"], 2), "dung": "50417" in r1["chu"]}
        kq["de_dai"].append(dong)
        print(f"  {dong['token_de']:>7} token đề: chờ chữ đầu {dong['cho_dau']} giây ({dong['nap_de']} token/giây nạp đề), "
              f"gửi lại {dong['cho_dau_lan_2']} giây, tìm đúng mã: {dong['dung']}")

    print("Nhiều lượt cùng lúc (mỗi lượt khoảng 600 chữ, không nghĩ):")
    for n in a.cung_luc:
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(n) as ex:
            lan = list(ex.map(lambda i: goi(ep, [{"role": "user", "content": f"Viết khoảng 500 chữ tiếng Việt về chủ đề số {i}: "
                                                 f"{['pin natri', 'chip 2nm', 'xe điện', 'robot hút bụi', 'tàu vũ trụ', 'mạng 6G', 'pin mặt trời', 'máy in 3D'][i % 8]}."}],
                                            700, temperature=0.7), range(n)))
        tg = time.perf_counter() - t0
        dong = {"so_luot": n, "moi_luot": trung_vi([l["giai_ma"] for l in lan]), "tong_ca_may": round(sum(l["ra"] for l in lan) / tg),
                "giay": round(tg, 1)}
        kq["cung_luc"].append(dong)
        print(f"  {n} lượt: mỗi lượt {dong['moi_luot']} token/giây, cả máy {dong['tong_ca_may']} token/giây ({dong['giay']} giây)")
    return kq


# ---------------------------------------------------------------- GSM8K
def so_cuoi(text):
    m = re.search(r"####\s*([-−]?[\d,]*\.?\d+)", text)
    if not m:
        nums = re.findall(r"[-−]?\d[\d,]*\.?\d*", text)
        if not nums:
            return None
        s = nums[-1]
    else:
        s = m.group(1)
    try:
        return float(s.replace(",", "").replace("−", "-").rstrip("."))
    except ValueError:
        return None


def gsm8k(ep, a, ghi):
    bai = tai_du_lieu("gsm8k_test.jsonl")[: a.gioi_han or None]

    def mot(i_b):
        i, b = i_b
        de = b["question"] + "\n\nSolve step by step. Write the final answer on the last line as: #### <number>"
        r = goi(ep, [{"role": "user", "content": de}], 8192 if a.nghi else 1024, nghi=a.nghi)
        dap = float(b["answer"].split("####")[-1].strip().replace(",", ""))
        tra = so_cuoi(r["chu"])
        return {"i": i, "dung": tra is not None and abs(tra - dap) < 1e-6, "dap": dap, "tra": tra, "ra": r["ra"],
                "chu": r["chu"][-400:]}

    return chay_lo(bai, mot, a, ghi, "GSM8K")


# ---------------------------------------------------------------- HumanEval
def lay_code(text):
    khoi = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(khoi, key=len) if khoi else text


def humaneval(ep, a, ghi):
    bai = tai_du_lieu("HumanEval.jsonl.gz")[: a.gioi_han or None]
    thu_muc = Path(tempfile.mkdtemp(prefix="humaneval_"))
    lenh = ["wsl.exe", "-e", "python3"] if a.chay_bang == "wsl" else a.chay_bang.split()

    def duong(p):
        if a.chay_bang != "wsl":
            return str(p)
        s = str(p).replace("\\", "/")
        return f"/mnt/{s[0].lower()}{s[2:]}"

    def mot(i_b):
        i, b = i_b
        de = ("Complete the following Python function. Reply with the complete function, including the signature and any "
              "imports it needs, in one ```python code block. No explanation.\n\n```python\n" + b["prompt"] + "```")
        r = goi(ep, [{"role": "user", "content": de}], 16384 if a.nghi else 2048, nghi=a.nghi)
        code = lay_code(r["chu"])
        # Đề (import, hàm phụ, chữ ký + docstring: hợp lệ dù chưa có thân) đặt trước; hàm model viết định nghĩa lại hàm cần
        # làm. Bỏ đề thì hỏng các bài đề cho sẵn hàm phụ (HumanEval/38: model chỉ trả decode_cyclic, thiếu encode_cyclic).
        p = thu_muc / f"bai_{i:03d}.py"
        p.write_text(f"{b['prompt']}\n{code}\n\n{b['test']}\n\ncheck({b['entry_point']})\n", encoding="utf-8")
        try:
            kq = subprocess.run(lenh + [duong(p)], capture_output=True, timeout=30)
            dung, loi = kq.returncode == 0, kq.stderr.decode("utf-8", "replace")[-300:]
        except subprocess.TimeoutExpired:
            dung, loi = False, "quá 30 giây"
        return {"i": i, "task": b["task_id"], "dung": dung, "ra": r["ra"], "loi": loi, "code": code}

    return chay_lo(bai, mot, a, ghi, "HumanEval")


def chay_lo(bai, mot, a, ghi, ten):
    t0 = time.perf_counter()
    dong = []
    with cf.ThreadPoolExecutor(a.song_song) as ex, open(ghi, "w", encoding="utf-8") as f:
        for k in ex.map(mot, enumerate(bai)):
            dong.append(k)
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
            if len(dong) % 20 == 0 or len(dong) == len(bai):
                d = sum(x["dung"] for x in dong)
                print(f"  {ten}: {len(dong)}/{len(bai)} bài, đúng {d} ({100 * d / len(dong):.1f}%), "
                      f"{time.perf_counter() - t0:.0f} giây", flush=True)
    d = sum(x["dung"] for x in dong)
    return {"so_bai": len(dong), "dung": d, "diem": round(100 * d / len(dong), 1), "nghi": a.nghi,
            "token_ra_trung_vi": trung_vi([x["ra"] for x in dong]), "giay": round(time.perf_counter() - t0)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("bai", choices=["toc_do", "gsm8k", "humaneval"])
    p.add_argument("--endpoint", required=True, help="file JSON {baseUrl, token, model}")
    p.add_argument("--may", default="", help="mô tả máy và cách chạy, ghi vào kết quả")
    p.add_argument("--nghi", action="store_true", help="bật nghĩ (mặc định tắt)")
    p.add_argument("--song-song", type=int, default=4, help="số lượt gửi cùng lúc (GSM8K, HumanEval)")
    p.add_argument("--gioi-han", type=int, default=0, help="chỉ chạy N bài đầu")
    p.add_argument("--chay-bang", default=sys.executable, help='lệnh chạy code HumanEval; "wsl" = python3 trong WSL')
    p.add_argument("--de-dai", type=int, nargs="+", default=[1000, 8000, 32000, 64000, 120000])
    p.add_argument("--cung-luc", type=int, nargs="+", default=[1, 2, 4, 8])
    a = p.parse_args()
    ep = json.loads(Path(a.endpoint).read_text(encoding="utf-8"))
    ep["baseUrl"] = ep["baseUrl"].rstrip("/").removesuffix("/v1")
    thu_muc = GOC / "ket_qua" / f"{datetime.now():%Y-%m-%d}_{ep['model']}"
    thu_muc.mkdir(parents=True, exist_ok=True)
    ten = a.bai + ("_nghi" if a.nghi and a.bai != "toc_do" else "")
    t0 = time.time()
    if a.bai == "toc_do":
        kq = toc_do(ep, a)
    else:
        kq = (gsm8k if a.bai == "gsm8k" else humaneval)(ep, a, thu_muc / f"{ten}.jsonl")
        print(f"{a.bai}: {kq['dung']}/{kq['so_bai']} = {kq['diem']}%")
    kq = {"model": ep["model"], "may": a.may, "bat_dau": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
          "qua_tunnel": "trycloudflare.com" in ep["baseUrl"], **kq}
    (thu_muc / f"{ten}.json").write_text(json.dumps(kq, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("đã ghi", (thu_muc / f"{ten}.json").relative_to(GOC.parent))


if __name__ == "__main__":
    main()
