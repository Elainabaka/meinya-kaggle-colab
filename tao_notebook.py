"""Đổi file nguồn trong nguon/ thành notebook .ipynb.

Dùng:  py -3 -X utf8 tao_notebook.py            (dựng mọi notebook khai trong NOTEBOOKS)
       py -3 -X utf8 tao_notebook.py --kiem     (chỉ kiểm cú pháp từng ô, không ghi file)
       py -3 -X utf8 tao_notebook.py <nguồn>.py (chỉ dựng notebook của các nguồn kể tên, các notebook khác giữ nguyên)

File nguồn là Python thường, chia ô bằng dòng "# %%":
    # %% [markdown]      ô chữ; mỗi dòng bắt đầu bằng "# " (bỏ tiền tố khi dựng)
    # %%                 ô code
    #@nhung <file>       trong ô code: thay cả ô bằng "%%writefile <file>" + nội dung nguon/<file>
    #@chen <file>        đứng riêng một dòng: chèn nguyên nguon/<file> (các ô dùng chung cho nhiều notebook)
    #@bo <file>          ở đầu file nguồn: bỏ mọi "#@chen <file>" đó, kể cả chèn lồng (một notebook bỏ một ô chung)
Chỉ dùng thư viện chuẩn. Notebook dựng ra không có output.
"""
import ast
import json
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parent
NGUON = GOC / "nguon"

# nguồn -> các nơi đặt notebook (cùng một nội dung), kèm loại máy Colab ghi vào metadata (gpuType)
NOTEBOOKS = {
    "qwen38_27b_endpoint.py": [(GOC / "Kaggle" / "Qwen3.8-27B endpoint.ipynb", "T4")],     # llama.cpp, Kaggle 2×T4
    "qwen38_27b_a100.py": [(GOC / "Colab" / "Qwen3.8-27B A100.ipynb", "A100")],          # vLLM INT4 + DSpark
    "qwen38_27b_g4.py": [(GOC / "Colab" / "Qwen3.8-27B G4.ipynb", "G4")],                # vLLM NVFP4 + DSpark
    "qwen38_flash_next_g4.py": [(GOC / "Colab" / "Qwen3.8-Flash-Next G4.ipynb", "G4")],  # vLLM NVFP4 + MTP, bảng n-gram ở RAM
    "qwen38_27b_tpu.py": [(GOC / "Colab" / "Qwen3.8-27B TPU v6e.ipynb", "V6E1")],        # vLLM TPU NVFP4 + MTP
    "fable_35b_a3b.py": [(GOC / "Kaggle" / "FableCoder 35B-A3B endpoint.ipynb", "T4")],  # llama.cpp, Kaggle 2×T4
}


def chen(text, sau=0, bo=frozenset()):
    """Thay mỗi dòng '#@chen <file>' bằng nội dung nguon/<file> từ dòng '# %%' đầu tiên trở đi
    (phần chú thích ở đầu file chèn không vào notebook); file nằm trong bo thì bỏ. Chèn lồng tối đa 3 tầng."""
    if sau > 3:
        raise SystemExit("#@chen lồng quá 3 tầng")
    out = []
    for line in text.splitlines():
        if line.startswith("#@chen ") and line.split(None, 1)[1].strip() in bo:
            continue
        if line.startswith("#@chen "):
            con = chen((NGUON / line.split(None, 1)[1].strip()).read_text(encoding="utf-8"), sau + 1, bo).splitlines()
            out += con[next((i for i, l in enumerate(con) if l.startswith("# %%")), len(con)):]
        else:
            out.append(line)
    return "\n".join(out)


def tach_o(text):
    """Trả danh sách (loại, nguồn) theo dấu '# %%'."""
    cells, kind, buf = [], None, []
    bo = frozenset(l.split(None, 1)[1].strip() for l in text.splitlines() if l.startswith("#@bo "))
    for line in chen(text, bo=bo).splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, buf))
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        elif kind is not None:
            buf.append(line)
    if kind is not None:
        cells.append((kind, buf))
    out = []
    for kind, buf in cells:
        while buf and not buf[-1].strip():
            buf.pop()
        while buf and not buf[0].strip():
            buf.pop(0)
        if kind == "markdown":
            buf = [l[2:] if l.startswith("# ") else ("" if l == "#" else l) for l in buf]
        elif buf and buf[0].startswith("#@nhung "):
            name = buf[0].split(None, 1)[1].strip()
            buf = [f"%%writefile {name}"] + (NGUON / name).read_text(encoding="utf-8").rstrip("\n").split("\n")
        out.append((kind, "\n".join(buf)))
    return out


def kiem_cu_phap(cells, ten):
    for i, (kind, src) in enumerate(cells):
        if kind != "code":
            continue
        body = src.split("\n", 1)[1] if src.startswith("%%writefile ") else src
        try:
            ast.parse(body)
        except SyntaxError as e:
            raise SystemExit(f"{ten}: ô {i} sai cú pháp, dòng {e.lineno}: {e.msg}")


def dung(cells, colab_gpu):
    nb_cells = []
    for kind, src in cells:
        lines = src.split("\n")
        cell = {"cell_type": kind, "metadata": {},
                "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    tpu = colab_gpu.startswith("V")       # Colab ghi loại TPU dạng "V6E1", "V5E1"
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            **({} if tpu else {"kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True}}),
            "accelerator": "TPU" if tpu else "GPU",
            "colab": {"provenance": [], "gpuType": colab_gpu},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    chi_kiem = "--kiem" in sys.argv
    chon = [a for a in sys.argv[1:] if not a.startswith("--")]
    if thieu := [a for a in chon if a not in NOTEBOOKS]:
        raise SystemExit(f"Không có trong NOTEBOOKS: {thieu}")
    for ten, dich in NOTEBOOKS.items():
        if chon and ten not in chon:
            continue
        cells = tach_o((NGUON / ten).read_text(encoding="utf-8"))
        kiem_cu_phap(cells, ten)
        if chi_kiem:
            print(f"{ten}: {len(cells)} ô, cú pháp đạt")
            continue
        for p, colab_gpu in dich:
            data = json.dumps(dung(cells, colab_gpu), ensure_ascii=False, indent=1) + "\n"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(data, encoding="utf-8", newline="\n")
            print(f"đã ghi {p.relative_to(GOC)} ({len(cells)} ô, {len(data) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
