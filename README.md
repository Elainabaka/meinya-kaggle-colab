# Meinya Kaggle & Colab: model lớn trên GPU đám mây, ra endpoint chuẩn OpenAI

*English: Jupyter notebooks that turn a free Kaggle 2×T4 session or a Colab Pro G4 into a private, OpenAI-compatible LLM
endpoint (gateway with API key, Cloudflare tunnel, vision, tool calling). Headline: **Qwen3.8-Flash-Next (180B MoE, 6B
active) on a single Colab G4 (RTX PRO 6000, 96 GB): 191 tok/s writing code, 129 tok/s writing Vietnamese, 131K context,
GSM8K 96.4%, HumanEval 97.0%**, measured from a home connection through the tunnel. Notebook text is in Vietnamese.*

Mỗi notebook chỉ cần **Run all**: tự dò máy, cài engine, tải model, chạy model, tự kiểm rồi in khung **ENDPOINT SẴN SÀNG**
có Base URL, API key và tên model. Dán ba thứ đó vào app hay thư viện nào nói chuẩn OpenAI là dùng được.

| Mở ngay | Máy | Model | Đo thật |
|---|---|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Elainabaka/meinya-kaggle-colab/blob/main/Colab/Qwen3.8-Flash-Next%20G4.ipynb) | Colab Pro G4 | Qwen3.8-Flash-Next 180B | code 191, văn Việt 129 token/giây |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Elainabaka/meinya-kaggle-colab/blob/main/Colab/Qwen3.8-27B%20G4.ipynb) | Colab Pro G4 | Qwen3.8-27B | code 168, văn Việt 58 token/giây |
| [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/Elainabaka/meinya-kaggle-colab/blob/main/Kaggle/FableCoder%2035B-A3B%20endpoint.ipynb) | Kaggle 2×T4 (miễn phí) | FableCoder 35B-A3B | code khoảng 94, văn 73 token/giây |
| [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/Elainabaka/meinya-kaggle-colab/blob/main/Kaggle/Qwen3.8-27B%20endpoint.ipynb) | Kaggle 2×T4 (miễn phí) | Qwen3.8-27B | code khoảng 30 token/giây |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Elainabaka/meinya-kaggle-colab/blob/main/Colab/Qwen3.8-27B%20A100.ipynb) | Colab Pro A100 | Qwen3.8-27B | chưa thử |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Elainabaka/meinya-kaggle-colab/blob/main/Colab/Qwen3.8-27B%20TPU%20v6e.ipynb) | Colab TPU v6e-1 | Qwen3.8-27B | chưa thử |

Kaggle cho 30 giờ GPU mỗi tuần, miễn phí (đọc bằng `kaggle quota` ngày 13/10/2026). Máy G4, A100 và TPU của Colab cần gói
trả phí (Colab Pro trở lên; FAQ của Colab: "premium GPUs subject to availability").

## Số đo nổi bật: Qwen3.8-Flash-Next 180B trên một máy Colab G4

Đo ngày 13/10/2026 từ một máy ở Việt Nam, gọi qua Cloudflare Quick Tunnel (số đo gồm cả đường mạng). Máy: Colab Pro G4
(RTX PRO 6000 Blackwell 96 GB), vLLM 0.31.0, `nvidia/Qwen3.8-Flash-Next-NVFP4`, đoán trước MTP 3 token, ngữ cảnh 131.072.
Model có 180 tỷ tham số nhưng mỗi token chỉ chạy 6 tỷ; bảng n-gram 47 GiB nằm ở RAM máy, GPU giữ phần còn lại.

| Việc (một lượt, không nghĩ) | Tốc độ viết | Chờ chữ đầu |
|---|---|---|
| Viết code Python | **191 token/giây** | 0,4 giây |
| Văn tiếng Việt | **129 token/giây** | 0,4 giây |
| Văn tiếng Anh | 132 token/giây | 0,3 giây |
| Dịch Trung → Việt | 150 token/giây | 0,4 giây |
| Toán, có nghĩ | 181 token/giây | 0,4 giây |

| Độ dài đề | Chờ chữ đầu | Gửi lại đúng đề đó (bộ nhớ đệm) | Tìm đúng mã giấu giữa đề |
|---|---|---|---|
| 6.314 token | 0,79 giây | 0,58 giây | đúng |
| 25.419 token | 2,31 giây | 0,82 giây | đúng |
| 51.704 token | 4,15 giây | 0,98 giây | đúng |
| 97.898 token | 7,53 giây | 1,17 giây | đúng |

| Chất lượng | Kết quả | Cách chấm |
|---|---|---|
| GSM8K (1.319 bài toán) | **96,4%** (1.271 bài) | không nghĩ, nhiệt độ 0, so số cuối với đáp án |
| HumanEval (164 bài code) | **97,0%** (159 bài) | không nghĩ, nhiệt độ 0, chạy test của bài |
| HumanEval, có nghĩ | **97,0%** (159 bài) | như trên, bật nghĩ, trần 16.384 token; cả 5 bài sai đều nghĩ chưa xong khi chạm trần |
| Bộ kiểm nội dung tiếng Việt (31 mục) | **31/31** | code, toán, gọi tool, đọc ảnh, OCR chữ Trung, dịch phim, viết tiêu đề, lời bình, "biết nói không biết" |

So với Qwen3.8-27B trên cùng máy G4: viết code ngang nhau (168 token/giây), văn tiếng Việt nhanh gần gấp đôi (58 → 129).
Kết quả từng câu: [`bench/ket_qua/2026-10-13_qwen3.8-flash-next/`](bench/ket_qua/2026-10-13_qwen3.8-flash-next/).

## Notebook có sẵn

| Notebook | Máy | Model, engine | Đo thật |
|---|---|---|---|
| `Colab/Qwen3.8-Flash-Next G4.ipynb` | Colab Pro G4 | Qwen3.8-Flash-Next 180B NVFP4, vLLM 0.31.0, MTP 3, 131K | code 191, văn Việt 129 token/giây (13/10/2026) |
| `Colab/Qwen3.8-27B G4.ipynb` | Colab Pro G4 | Qwen3.8-27B NVFP4, vLLM 0.31.0, DSpark, 262K | code 168, văn Việt 58 token/giây (13/10/2026) |
| `Kaggle/FableCoder 35B-A3B endpoint.ipynb` | Kaggle 2×T4 (miễn phí) | FableCoder 35B-A3B Q4_K_M, llama.cpp, MTP 2, 262K | code khoảng 94, văn 73 token/giây (13/10/2026) |
| `Kaggle/Qwen3.8-27B endpoint.ipynb` | Kaggle 2×T4 (miễn phí) | Qwen3.8-27B Q4_K_M, llama.cpp, MTP 3 | code khoảng 30 token/giây (13/10/2026) |
| `Colab/Qwen3.8-27B A100.ipynb` | Colab Pro A100 | Qwen3.8-27B INT4, vLLM 0.31.0, DSpark | chưa thử |
| `Colab/Qwen3.8-27B TPU v6e.ipynb` | Colab TPU v6e-1 | Qwen3.8-27B NVFP4, vLLM TPU 0.28.0, MTP 3 | chưa thử |

Mỗi loại chip một notebook riêng vì cách nén và cờ chạy khác nhau; phần cài engine, gateway, tunnel và giữ phiên dùng chung.

## Cách dùng

1. Bấm nút **Open in Colab** / **Open in Kaggle** ở bảng trên, chọn đúng máy (Kaggle: GPU T4 x2, bật Internet; Colab:
   Runtime → Change runtime type → G4).
2. **Run all**. Lần đầu mất khoảng 20–30 phút (cài engine, tải model; bản Flash-Next tải 133 GB).
3. Chép `Base URL`, `API key`, `Model` trong khung **ENDPOINT SẴN SÀNG**. Ví dụ:

```python
from openai import OpenAI
c = OpenAI(base_url="https://<địa-chỉ>.trycloudflare.com/v1", api_key="<API key>")
print(c.chat.completions.create(model="qwen3.8-flash-next", messages=[{"role": "user", "content": "Xin chào"}]).choices[0].message.content)
```

Tắt nghĩ cho một lượt: thêm `"chat_template_kwargs": {"enable_thinking": false}`. Nên gọi với `"stream": true`.

**Secret** (không bắt buộc, đặt ở bảng Secrets của Kaggle/Colab, notebook không in ra): `HF_TOKEN` (tải nhanh hơn),
`GATEWAY_TOKEN` (API key cố định), `CF_TUNNEL_TOKEN` + `PUBLIC_BASE_URL` (tunnel có tên, địa chỉ cố định). Không đặt thì mỗi
phiên sinh khóa mới và dùng Quick Tunnel, địa chỉ đổi mỗi phiên.

## Có gì trong repo

| Đường dẫn | Là gì |
|---|---|
| `Kaggle/`, `Colab/` | Notebook dựng sẵn, không có output |
| `nguon/` | Nguồn của notebook: file Python thường, chia ô bằng `# %%`; `nguon/chung/` là các ô dùng chung |
| `tao_notebook.py` | Dựng lại notebook từ `nguon/` (`--kiem` chỉ kiểm cú pháp từng ô) |
| `bench/danh_gia.py`, `bench/ket_qua/` | Bộ đo tốc độ, GSM8K, HumanEval và kết quả gốc của bảng trên |
| `do_moi_truong.py` | Dò cấu hình máy (GPU, RAM, đĩa, thư viện); không đọc gì riêng tư, không in khóa |
| `cấu hình chung hiện tại 2026.md` | Cấu hình Kaggle đo thật ngày 13/10/2026 và cấu hình Colab theo nguồn chính thức |

Sửa notebook thì sửa trong `nguon/` rồi chạy `python3 tao_notebook.py` (Windows: `py -3 -X utf8 tao_notebook.py`).

## Tự đo lại

```
python3 bench/danh_gia.py --endpoint <file.json> --may "<mô tả máy>" toc_do
python3 bench/danh_gia.py --endpoint <file.json> --may "<mô tả máy>" gsm8k
python3 bench/danh_gia.py --endpoint <file.json> --may "<mô tả máy>" humaneval --chay-bang wsl [--nghi]
```

`<file.json>` là `{"baseUrl": "...", "token": "...", "model": "..."}` lấy từ khung ENDPOINT (giữ file này cho riêng mình,
trong đó có API key). Script chỉ dùng thư viện chuẩn Python, tự tải GSM8K test và HumanEval (cả hai MIT, của OpenAI) vào
`bench/du_lieu/`, ghi kết quả từng câu vào `bench/ket_qua/`. HumanEval chạy code do model viết: nên chạy trong WSL hay máy ảo
(`--chay-bang wsl`).

**Giới hạn của số đo:** mỗi bài đo một lần; GSM8K và HumanEval là bộ đề cũ, model mới làm gần hết và có thể đã thấy đề lúc
học, nên chỉ cho thấy bản nén NVFP4 không bị hỏng chất lượng, không dùng để so model; HumanEval chấm bằng cách riêng của script
(ghép đề với code model trả, chạy test gốc), không phải evalplus. Số lượt cùng lúc trong lần đo này bị gateway giới hạn ở 3
(đã sửa trong notebook sau lần đo), nên chưa đưa vào bảng.

## Giấy phép và lưu ý

- Notebook, script, tài liệu trong repo: MIT (`LICENSE`).
- Model có giấy phép riêng: Qwen3.8-Flash-Next dùng Qwen Community License 1.0, bản NVIDIA thêm NVIDIA Open Model License;
  Qwen3.8-27B dùng Apache-2.0; các model khác xem trang của model trên Hugging Face. Đọc trước khi dùng cho việc thương mại.
- Tunnel: FAQ của Colab không cho "web service offerings not related to interactive compute" và "connecting to remote proxies";
  tunnel có thuộc diện đó không thì Colab không nói rõ. Notebook in lưu ý này; bạn tự quyết.
- Endpoint chỉ đưa gateway ra ngoài (có API key, chỉ mở `/healthz`, `/capabilities`, `/v1/models`, `/v1/chat/completions`);
  engine chạy model chỉ nghe `127.0.0.1`.

Làm bởi [Elainabaka](https://elainabaka.com). Thấy lỗi hay muốn thêm máy, model: mở issue hoặc pull request.
