# Cấu hình Kaggle và Google Colab (đo và kiểm ngày 13/10/2026)

> **Dùng để:** AI agent đọc trước khi viết notebook (`.ipynb`) hoặc script chạy trên Kaggle và Colab.
> **Phần Kaggle là số đo thật** trên máy Kaggle ngày 13/10/2026 (07:45–08:04 giờ Việt Nam) bằng [do_moi_truong.py](do_moi_truong.py). **Phần Colab lấy từ nguồn chính thức, chưa đo** (Colab CLI chưa chạy trên Windows).

## 0. Cách đọc

| Nhãn | Nghĩa |
|---|---|
| `[đo]` | chạy thật trên máy của nền tảng, ngày ghi ở trên |
| `[nguồn]` | đọc nguyên văn ở trang chính thức ngày 13/10/2026 (link ở mục 6) |
| `[thứ cấp]` | bài bên thứ ba, hoặc chỉ đọc được tiêu đề trang |
| `[suy luận]` | tự tính từ số đã đo hoặc đã đọc, chưa chạy thử |
| `[chưa kiểm]` | không tìm được nguồn; đừng coi là sự thật |

**Luật số 1:** hai nền tảng đổi cấu hình thường xuyên (Colab viết: "Colab does not publish these limits, in part because they can vary over time"). Notebook không viết cứng cấu hình: ô đầu tiên dò máy rồi rẽ nhánh (mục 4.1). Số trong file này dùng để lên kế hoạch.

**Đo lại:** sửa `id` trong `kernel-metadata.json` (mẫu ở mục 6) rồi `kaggle kernels push -p <thư mục> --accelerator NvidiaTeslaT4`, xong `kaggle kernels output <id> -p out`. Trên Colab: dán nội dung `do_moi_truong.py` vào một ô và chạy.

## 1. Kaggle Notebooks

### 1.1 Bộ tăng tốc

| Lựa chọn | Tình trạng 13/10/2026 | Bằng chứng |
|---|---|---|
| **GPU T4 x2** | Có. 2 × Tesla T4, mỗi cái 15.360 MiB (PyTorch thấy 14,56 GiB), compute capability 7.5 | `[đo]` |
| GPU P100 | **Đã bỏ.** Yêu cầu `NvidiaTeslaP100` qua CLI vẫn nhận 2 × T4 | `[đo]`; bài của Kaggle "Sunsetting the NVIDIA Tesla P100 GPU on September 15, 2026" `[thứ cấp: chỉ đọc được tiêu đề]` |
| A100, L4, L4 x1, H100, RTX Pro 6000 | Có tên trong danh sách của CLI ("Accelerators available as of Sep 2026") nhưng "Some of these are only available to participants of specific competitions, and some are only available to Kaggle admins." Tài khoản thường: coi như không có | `[nguồn]` |
| TPU | Danh sách CLI ghi `TpuV5E8` (TPU v5e-8) và `TpuV6E8`. TPU v3-8 cũ: yêu cầu `Tpu1VmV38` nằm hàng chờ 14 phút không chạy | `[nguồn]` cho danh sách; `[đo]` cho hàng chờ; bài "New TPU v5e-8s! Phasing out v3-8s" `[thứ cấp: tiêu đề]`. TPU v5e-8, v6e-8: chưa đo |

### 1.2 Máy `[đo]`

| | Phiên GPU T4 x2 | Phiên CPU |
|---|---|---|
| CPU | 4 vCPU (Xeon 2,00 GHz; 2 nhân × 2 luồng) | 4 vCPU (Xeon 2,20 GHz) |
| RAM | 31 GiB | 31 GiB |
| `/dev/shm` | 14 GiB | 14 GiB |
| GPU | 2 × Tesla T4, driver 580.178.04, CUDA toolkit 12.8 | không |

### 1.3 Ổ đĩa và đường dẫn `[đo]`

| Đường dẫn | Sự thật |
|---|---|
| `/kaggle/working` | thư mục làm việc mặc định; ghi được; **19,5 GiB**; là phần được lưu thành Output |
| `/kaggle/input` | **chỉ đọc** (ghi vào báo `OSError 30 Read-only file system`). Bố cục thư mục con khi gắn dataset: `[chưa kiểm]`, nên dò bằng `os.listdir` hoặc `kagglehub`, đừng viết cứng |
| `/kaggle/temp`, `/kaggle/tmp` | **không tồn tại** |
| `/tmp` | ghi được, mất khi phiên tắt. Lúc đo còn trống 1.070 GiB trên ổ chung của máy chủ; Kaggle không cam kết con số này |
| `/root` | thư mục HOME |

Model nặng (từ khoảng 15 GB) tải vào `/tmp`, không tải vào `/kaggle/working`.

### 1.4 Phần mềm `[đo]`

Image `docker-python` build ngày 01/10/2026, dựng trên image của Colab `release-colab-external-images_20260917`.

| Mục | Giá trị |
|---|---|
| Hệ điều hành | Ubuntu 24.04.1 LTS (image GPU), 24.04.5 LTS (image CPU); kernel 6.18 |
| Python | 3.13.15 |
| User | `root` (uid 0); `sudo` và `apt` dùng được |
| PyTorch | 2.11.0+cu128 (hỗ trợ `sm_75` tới `sm_120`), cuDNN 9.19; torchvision 0.26.0; torchaudio 2.11.0 |
| TensorFlow, Keras, JAX | 2.20.0; 3.13.2; 0.11.1 (JAX thấy cả 2 GPU) |
| Hugging Face | transformers 5.16.1; accelerate 1.14.0; peft 0.20.0; datasets 4.8.5; huggingface_hub 1.29.0; diffusers 0.40.0 |
| Khác | numpy 2.1.3; pandas 2.3.3; kagglehub 1.0.2; `uv` 0.12.9; pip 24.1.2; ffmpeg 6.1.1; gcc 13.3; cmake; ninja; mamba; tmux; git |
| **Chưa cài sẵn** | `bitsandbytes`, `flash-attn`, `xformers`, `vllm`, `llama-cpp-python`, `onnxruntime`, `optimum`, `trl`, `unsloth`, `autoawq`, `gptqmodel`, `git-lfs`, `aria2c`, `conda` |

`flash-attn` trên PyPI với môi trường này chỉ có bản mã nguồn (`flash_attn-2.8.3.post1.tar.gz`): cài là phải biên dịch.

### 1.5 Mạng

- Đẩy bằng CLI với `"enable_internet": true` thì vào được huggingface.co, pypi.org, github.com `[đo]`. Không khai thì mặc định tắt: "If not specified, will be `false`" `[nguồn]`.
- Mặc định trên giao diện web và điều kiện xác minh số điện thoại: `[chưa kiểm]`. Notebook tự kiểm mạng ở ô đầu (mục 4.1).

### 1.6 Hạn mức

| Mục | Giá trị | Nhãn |
|---|---|---|
| GPU mỗi tuần | 30 giờ | `[đo]` bằng `kaggle quota` trên một tài khoản; lần làm mới kế tiếp ghi `2026-10-17T00:00:00` (thứ Bảy) |
| TPU mỗi tuần | 20 giờ | `[đo]` như trên |
| Thời lượng tối đa một phiên | 12 giờ (CPU, GPU), 9 giờ (TPU) | `[thứ cấp]`; trang tài liệu Kaggle cần JavaScript nên chưa đọc được |
| Thời gian chờ khi để máy không làm gì | | `[chưa kiểm]` |
| Số phiên GPU cùng lúc | Coi là 1: đẩy hai kernel GPU cùng lúc thì cả hai kẹt `QUEUED` hơn 20 phút | `[đo]` 11/10/2026 |

Máy T4 x2 có 2 GPU: nhiều việc thì chạy chung một phiên, mỗi việc một GPU bằng `CUDA_VISIBLE_DEVICES`.

### 1.7 T4 làm được gì `[đo]`

| Việc | Kết quả |
|---|---|
| `torch.float16` | chạy thẳng trên phần cứng |
| `torch.bfloat16` | `torch.cuda.is_bf16_supported()` trả `True` nhưng `is_bf16_supported(including_emulation=False)` trả `False`: chạy bằng giả lập. Attention (SDPA) 0,391 s so với 0,049 s của fp16 (chậm 8 lần); nhân ma trận 1,117 s so với 0,453 s (chậm 2,5 lần). **Trên T4 dùng `torch.float16`** |
| `attn_implementation="sdpa"` | chạy tốt (backend flash của PyTorch bật sẵn) |
| Gói `flash-attn` | README ghi FlashAttention-2 hỗ trợ "Ampere, Ada, or Hopper GPUs"; T4 (Turing) phải dùng repo riêng `flash-attention-turing` `[nguồn]`. **Trên T4 không dùng `flash_attention_2`** |
| Trọng số FP8, NVFP4 | `[chưa kiểm]` trên T4; đừng lên kế hoạch dựa vào |

## 2. Google Colab (chưa đo)

### 2.1 Bộ tăng tốc

Danh sách hiện có `[nguồn: README của google-colab-cli]`: "Spin up CPU, GPU (T4, L4, G4, H100, A100), or TPU (v5e1, v6e1) runtimes in seconds." TPU v2-8, V100, P100 không có trong danh sách này.

| Loại | Bộ nhớ | Nhãn |
|---|---|---|
| T4 | 16 GB (15.360 MiB, đo trên cùng loại card ở Kaggle) | `[đo]` ở Kaggle |
| L4 | 24 GB | `[nguồn]` tài liệu GPU của Google Cloud, không phải trang Colab |
| A100 | 40 GB hoặc 80 GB; Colab cấp loại nào: `[chưa kiểm]` | |
| H100 | 80 GB | `[nguồn]` Google Cloud |
| G4 | 96 GB (NVIDIA RTX PRO 6000 Blackwell) | `[nguồn]` Google Cloud cho máy G4; "G4 của Colab là chip này": `[thứ cấp]` |
| TPU v5e-1 / v6e-1 | 16 GB / 32 GB HBM mỗi chip | `[nguồn]` tài liệu Cloud TPU |

- Bản miễn phí: "access to expensive resources like GPUs is heavily restricted"; GPU cao cấp: "You can access premium GPUs subject to availability by purchasing one of our paid plans" `[nguồn: FAQ]`. Bản miễn phí thường được T4 `[thứ cấp]`.
- Cờ `--high-mem` cần gói Pro hoặc Pro+; "L4 and TPU runtimes ignore this flag because they only offer one shape" `[nguồn]`.

### 2.2 RAM, vCPU, ổ đĩa

Colab không công bố `[nguồn: FAQ]`. Các số hay gặp trên mạng (12,7 GB RAM; 51–53 GB High-RAM; 100–225 GB đĩa) đều `[chưa kiểm]`. Ô dò môi trường in số thật.

### 2.3 Phần mềm `[nguồn: repo googlecolab/backend-info, commit mới nhất 03/10/2026]`

README của repo ghi các file này có thể lệch bản đang chạy một hai ngày.

| | CPU | GPU | TPU |
|---|---|---|---|
| Ubuntu | 24.04.5 LTS | 24.04.4 LTS | 24.04.5 LTS |
| Python | 3.13.16 | 3.13.15 | 3.13.16 |
| PyTorch | 2.11.0+cpu | **2.11.0+cu130** (CUDA 13.0 từ 30/09/2026) | 2.9.0+cpu kèm torch-xla 2.9.0 |
| TensorFlow | 2.21.0 | 2.21.0 | không có |
| JAX | 0.11.1 | 0.11.1 | 0.7.2 |
| transformers / accelerate | 5.18.0 / 1.15.0 | 5.18.0 / 1.15.0 | 5.18.0 / 1.15.0 |

- Không có trong danh sách gói cài sẵn: `bitsandbytes`, `flash-attn`, `vllm`, `xformers`, `llama-cpp-python`.
- Ghim bản runtime cũ được ("past runtime versions are available for 1 year after their release"). Bản 2026.07: Ubuntu 22.04.5, Python 3.12.13, PyTorch 2.11.0, JAX 0.7.2, TensorFlow 2.20.0 `[nguồn]`.
- **Kaggle dùng CUDA 12.8 (torch `cu128`), Colab dùng CUDA 13.0 (torch `cu130`).** Wheel biên dịch theo CUDA không dùng chung cho hai bên.

### 2.4 Giới hạn `[nguồn: FAQ]`

| Mục | Nguyên văn |
|---|---|
| Bản miễn phí | "notebooks can run for at most 12 hours, depending on availability and your usage patterns" |
| Pro+ | "continuous code execution for up to 24 hours if you have sufficient compute units" |
| Để máy không làm gì | "Runtimes will time out if you are idle." Số phút không công bố |
| Compute unit | "Google AI Pro members get 200 CCUs." `[nguồn: trang hỗ trợ Google One]`. Mức đốt mỗi giờ theo loại GPU: `[chưa kiểm]`, xem trong giao diện Colab |
| Bị ngắt không báo trước ở bản miễn phí, hoặc bản trả tiền đã hết compute unit | "remote control such as SSH shells, remote desktops"; "bypassing the notebook UI to interact primarily via a web UI"; "running distributed computing workers" |
| Cấm ở mọi gói | đào coin, torrent, "connecting to remote proxies", "using multiple accounts to work around access or resource usage restrictions" |

### 2.5 Đường dẫn và khóa bí mật

| Mục | Giá trị | Nhãn |
|---|---|---|
| Thư mục làm việc | `/content` | `[thứ cấp]` |
| Google Drive | `from google.colab import drive; drive.mount('/content/drive')` rồi dùng `/content/drive/MyDrive` | chữ ký hàm `[nguồn: colabtools]` |
| Khóa bí mật Colab | `from google.colab import userdata; userdata.get("TEN")` | `[nguồn: colabtools]` |
| Khóa bí mật Kaggle | `from kaggle_secrets import UserSecretsClient; UserSecretsClient().get_secret("TEN")`; tạo ở web: Add-ons → Secrets | `[nguồn: tài liệu kaggle-cli]` |
| User | `root` | `[chưa kiểm]` trên Colab; đã đo trên image Kaggle dựng từ image Colab |

## 3. So sánh nhanh

| Tiêu chí | Kaggle | Colab |
|---|---|---|
| GPU cho tài khoản thường | 2 × T4 (2 × 14,56 GiB dùng được) | miễn phí: thường T4; trả tiền: L4, A100, H100, G4 |
| CPU / RAM | 4 vCPU / 31 GiB `[đo]` | `[chưa kiểm]` |
| Python / Ubuntu | 3.13.15 / 24.04 | 3.13.15–3.13.16 / 24.04 |
| PyTorch | 2.11.0+cu128 | 2.11.0+cu130 |
| transformers | 5.16.1 | 5.18.0 |
| Thư mục làm việc | `/kaggle/working` (19,5 GiB) | `/content` |
| Chỗ để model nặng | `/tmp` | `/content` |
| Dữ liệu đầu vào | `/kaggle/input` (chỉ đọc) | Drive hoặc tải về `/content` |
| Mạng | phải bật (CLI: `enable_internet`) | bật sẵn `[thứ cấp]` |
| Lưu lâu dài | Output của kernel, Dataset | Google Drive |
| bfloat16 phần cứng | không (T4) | có từ L4 trở lên `[suy luận]` từ compute capability ≥ 8.0 |
| Hạn mức | 30 giờ GPU mỗi tuần `[đo]` | theo compute unit; miễn phí tối đa 12 giờ mỗi phiên |

## 4. Luật viết notebook cho AI agent

### 4.1 Ô đầu tiên: dò máy, không đoán

```python
import os, urllib.request
import torch

# Kiểm Kaggle TRƯỚC: image Kaggle dựng từ image Colab nên trên Kaggle cũng có biến COLAB_*,
# gói google-colab và thư mục /content [đo]. Kiểm Colab trước sẽ nhận nhầm.
ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
ON_COLAB = not ON_KAGGLE and "COLAB_RELEASE_TAG" in os.environ
WORK = "/kaggle/working" if ON_KAGGLE else "/content"
SCRATCH = "/tmp" if ON_KAGGLE else "/content"      # chỗ để model nặng
os.environ.setdefault("HF_HOME", f"{SCRATCH}/hf")

N_GPU = torch.cuda.device_count()
CAPS = [torch.cuda.get_device_capability(i) for i in range(N_GPU)]
VRAM = [torch.cuda.get_device_properties(i).total_memory / 2**30 for i in range(N_GPU)]
NATIVE_BF16 = N_GPU > 0 and torch.cuda.is_bf16_supported(including_emulation=False)
DTYPE = torch.bfloat16 if NATIVE_BF16 else torch.float16
ATTN = "sdpa"                                      # mặc định an toàn trên mọi GPU

try:
    urllib.request.urlopen("https://huggingface.co", timeout=10)
except Exception as e:
    raise SystemExit(f"Chưa có mạng ({e}). Kaggle: Settings -> Internet on, hoặc enable_internet trong kernel-metadata.json")

print(dict(kaggle=ON_KAGGLE, colab=ON_COLAB, gpus=N_GPU, caps=CAPS, vram_gib=[round(v, 1) for v in VRAM],
           dtype=str(DTYPE), torch=torch.__version__))
```

`COLAB_RELEASE_TAG` trên Colab thật: `[chưa đo]`.

### 4.2 Luật

1. Đổi thư mục bằng `%cd`, không dùng `!cd` (lệnh `!` chạy ở shell con, không giữ thư mục).
2. Gói chưa cài sẵn (mục 1.4, 2.3) thì cài ở đầu notebook: `%pip install -q bitsandbytes`. Python là 3.13: trước khi chọn một gói biên dịch sẵn, kiểm nó có wheel cho `cp313`.
3. Không cài lại `torch` (`pip install torch==...`): sẽ kéo bản CUDA khác và phá image. Gói nào đòi hạ torch thì chọn cách khác.
4. Kiểu số và attention lấy từ ô dò máy (`DTYPE`, `ATTN`). Chỉ dùng `flash_attention_2` khi compute capability ≥ 8.0 và đã cài được gói.
5. Kaggle: model nặng vào `/tmp` (`HF_HOME` đã đặt ở ô đầu), kết quả cần giữ ghi vào `/kaggle/working`, không ghi vào `/kaggle/input`.
6. Hai GPU: `device_map="auto"` kèm `max_memory={0: "13GiB", 1: "14GiB"}` để chừa chỗ cho ngữ cảnh `[suy luận]`.
7. Kaggle chạy nền không cần mở trình duyệt `[đo]`: `kaggle kernels push -p <thư mục> --accelerator NvidiaTeslaT4`, theo dõi bằng `kaggle kernels status <id>`, lấy kết quả bằng `kaggle kernels output <id> -p <thư mục>`. Trên Windows đặt `PYTHONUTF8=1` trước khi gọi `kaggle`.
8. Khóa bí mật lấy qua API ở mục 2.5, không dán vào notebook.
9. Colab: không mở SSH, tunnel hay web UI trên bản miễn phí (mục 2.4).

## 5. Qwen3.8-27B: tên đúng và bản nào vừa máy nào

| Mục | Giá trị | Nhãn |
|---|---|---|
| Repo | **`Qwen/Qwen3.8-27B`**, tạo 05/08/2026, Apache-2.0, 27,78 tỷ tham số BF16 | `[nguồn: API Hugging Face]` |
| Tên sai hay gặp | `Qwen/Qwen2.5-27B-Instruct` không có (API trả 401); Qwen2.5 không có cỡ 27B | `[nguồn]` |
| Cách nạp theo card | `AutoProcessor` + `AutoModelForMultimodalLM`, `device_map="auto"`; card không nhắc flash attention | `[nguồn: model card]` |
| Kiến trúc | `model_type: qwen3_5`; 64 lớp = 48 linear attention + 16 full attention; có phần nhìn ảnh; `config.json` ghi `transformers_version 5.8.0.dev0` (Kaggle 5.16.1 và Colab 5.18.0 mới hơn) | `[nguồn: config.json]`; chưa chạy thử nạp |
| Ngữ cảnh | 262.144 token | `[nguồn]` |
| Thinking | bật mặc định; tắt bằng `chat_template_kwargs: {"enable_thinking": False}` | `[nguồn]` |
| Sampling | thinking: T 1,0, top_p 0,95, top_k 20; không thinking: T 0,7, top_p 0,8, top_k 20, presence_penalty 1,5 | `[nguồn]` |

Cỡ file `[nguồn: API Hugging Face]`; cột "vừa" là `[suy luận]` từ cỡ file, chưa chạy, và còn phải trừ chỗ cho ngữ cảnh (KV cache):

| Bản | Cỡ | Kaggle T4 x2 (29,1 GiB) | L4 24 GB | A100 40 GB | A100 80 / H100 80 / G4 96 GB |
|---|---|---|---|---|---|
| BF16 gốc `Qwen/Qwen3.8-27B` | 55,56 GB | không | không | không | vừa |
| FP8 chính thức `Qwen/Qwen3.8-27B-FP8` | 30,87 GB | không | không | vừa | vừa |
| `nvidia/Qwen3.8-27B-NVFP4` | 21,92 GB | `[chưa kiểm]` | sát | vừa | vừa |
| `cyankiwi/Qwen3.8-27B-AWQ-INT4` | 21,02 GB | vừa nếu chia 2 GPU | sát | vừa | vừa |
| `RedHatAI/Qwen3.8-27B-INT4` | 19,45 GB | vừa nếu chia 2 GPU | sát | vừa | vừa |
| GGUF `unsloth/Qwen3.8-27B-GGUF` Q4_K_M | 16,46 GB (thêm `mmproj` 0,93 GB nếu cần ảnh) | vừa nếu chia 2 GPU | vừa | vừa | vừa |
| GGUF Q5_K_M / Q6_K / Q8_0 | 19,77 / 21,98 / 29,05 GB | vừa / vừa / không (chia 2 GPU) | sát / rất sát / không | vừa | vừa |

Đường TPU v6e-1 (32 GB): notebook `Colab/Qwen3.8-27B TPU v6e.ipynb`.

## 6. Nguồn và mẫu đo lại

| Nội dung | Nguồn |
|---|---|
| Danh sách bộ tăng tốc Kaggle, `kernel-metadata.json`, secrets | https://github.com/Kaggle/kaggle-cli (`docs/kernels.md`, `docs/kernels_metadata.md`) |
| Bỏ P100 | https://www.kaggle.com/discussions/product-announcements/735239 |
| TPU v5e-8 | https://www.kaggle.com/discussions/product-announcements/607202 |
| Image Kaggle | https://github.com/Kaggle/docker-python |
| Gói và hệ điều hành của Colab | https://github.com/googlecolab/backend-info |
| Giới hạn và điều cấm của Colab | https://research.google.com/colaboratory/faq.html |
| Ghim runtime Colab | https://research.google.com/colaboratory/runtime-version-faq.html |
| Loại GPU, TPU của Colab | https://github.com/googlecolab/google-colab-cli |
| Compute unit của Google AI Pro | https://support.google.com/googleone/answer/14534406 |
| Bộ nhớ GPU, TPU | https://docs.cloud.google.com/compute/docs/gpus, https://docs.cloud.google.com/tpu/docs/v6e |
| FlashAttention | https://github.com/Dao-AILab/flash-attention |
| Qwen3.8-27B | https://huggingface.co/Qwen/Qwen3.8-27B |

`kernel-metadata.json` dùng để đo (đặt cạnh `do_moi_truong.py`):

```json
{
  "id": "<tên-tài-khoản>/do-moi-truong",
  "title": "do moi truong",
  "code_file": "do_moi_truong.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```
