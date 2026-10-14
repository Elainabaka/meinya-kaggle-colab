# Environment probe: prints facts about the machine. Reads nothing private, prints no token values.
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request


def sh(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e!r}"


def show(key: str, value) -> None:
    print(f"@@ {key} = {value}", flush=True)


show("python", sys.version.replace("\n", " "))
show("executable", sys.executable)
show("platform", platform.platform())
show("os_release", sh("grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release"))
show("kernel", sh("uname -r"))
show("whoami", sh("whoami"))
show("id", sh("id"))
show("sudo_present", shutil.which("sudo"))
show("sudo_n_true", sh("sudo -n true && echo OK"))
show("apt_writable", os.access("/var/lib/dpkg", os.W_OK))
show("cwd", os.getcwd())
show("home", os.path.expanduser("~"))
show("nproc", os.cpu_count())
show("cpu_model", sh("lscpu | grep -E 'Model name|Thread|Core|Socket'"))
show("mem", sh("free -g | head -2"))
show("shm", sh("df -h /dev/shm | tail -1"))

for p in ["/", "/kaggle/working", "/kaggle/temp", "/kaggle/tmp", "/tmp", "/kaggle/input", "/content", "/root"]:
    if os.path.exists(p):
        u = shutil.disk_usage(p)
        show(f"disk[{p}]", f"total={u.total/2**30:.1f}GiB free={u.free/2**30:.1f}GiB writable={os.access(p, os.W_OK)}")
    else:
        show(f"disk[{p}]", "MISSING")
show("df", sh("df -h | grep -vE 'tmpfs|overlay.*docker' | head -30"))
show("ls_kaggle", sh("ls -la /kaggle"))
show("ls_kaggle_input", sh("ls -la /kaggle/input | head -20"))
try:
    with open("/kaggle/input/_probe_write_test", "w") as f:
        f.write("x")
    show("input_write", "WRITABLE")
    os.remove("/kaggle/input/_probe_write_test")
except Exception as e:  # noqa: BLE001
    show("input_write", repr(e))

# Environment variable NAMES only; values only for a fixed harmless list.
SAFE = {"KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_DOCKER_IMAGE", "KAGGLE_URL_BASE", "KAGGLE_GCP_ZONE",
        "KAGGLE_CONTAINER_NAME", "KAGGLE_KERNEL_INTEGRATIONS", "CUDA_VERSION", "NVIDIA_VISIBLE_DEVICES",
        "PYTHONPATH", "LD_LIBRARY_PATH", "TPU_NAME", "TPU_ACCELERATOR_TYPE", "TPU_WORKER_ID",
        "COLAB_RELEASE_TAG", "COLAB_GPU", "COLAB_TPU_1VM", "HF_HOME", "TRANSFORMERS_CACHE", "TMPDIR",
        "BUILD_DATE", "GIT_COMMIT", "KAGGLE_DATA_PROXY_URL", "MPLBACKEND", "TESSERACT_PATH"}
show("env_names", sorted(k for k in os.environ if any(t in k for t in ("KAGGLE", "COLAB", "CUDA", "NVIDIA", "TPU", "HF_", "XLA", "JAX"))))
for k in sorted(SAFE):
    if k in os.environ:
        show(f"env[{k}]", os.environ[k])

show("nvidia_smi_query", sh("nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap --format=csv"))
show("nvidia_smi", sh("nvidia-smi | head -25"))
show("nvcc", sh("nvcc --version | tail -2"))
show("cuda_dirs", sh("ls -d /usr/local/cuda* 2>/dev/null"))

for exe in ["uv", "git", "git-lfs", "ffmpeg", "aria2c", "wget", "curl", "gcc", "cmake", "ninja", "conda", "mamba", "pip", "pip3", "node", "ollama", "nvtop", "tmux"]:
    show(f"which[{exe}]", shutil.which(exe))
show("gcc_version", sh("gcc --version | head -1"))
show("ffmpeg_version", sh("ffmpeg -version | head -1"))
show("uv_version", sh("uv --version"))
show("pip_version", sh("pip --version"))

PKGS = ["torch", "torchvision", "torchaudio", "tensorflow", "keras", "jax", "jaxlib", "flax", "numpy", "pandas",
        "transformers", "tokenizers", "accelerate", "bitsandbytes", "peft", "datasets", "huggingface_hub", "hf_xet",
        "safetensors", "sentencepiece", "flash_attn", "xformers", "vllm", "triton", "onnxruntime", "onnxruntime-gpu",
        "tensorrt", "llama_cpp_python", "optimum", "auto_gptq", "autoawq", "gptqmodel", "llmcompressor", "unsloth",
        "trl", "diffusers", "kagglehub", "kaggle", "google-colab", "gradio", "scikit-learn", "scipy", "cupy-cuda12x",
        "nvidia-cudnn-cu12", "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12", "einops", "timm", "openai", "google-genai",
        "torch_xla", "torchao", "sentence-transformers", "faiss-gpu", "faiss-cpu", "librosa", "soundfile", "opencv-python",
        "ipykernel", "notebook", "jupyterlab", "ipywidgets", "papermill", "pyarrow", "polars", "xgboost", "lightgbm", "catboost"]
vers = {}
for name in PKGS:
    try:
        vers[name] = md.version(name)
    except Exception:  # noqa: BLE001
        vers[name] = None
show("pkgs", json.dumps(vers))
show("pkg_count", len(list(md.distributions())))

try:
    import torch

    show("torch", torch.__version__)
    show("torch_cuda_built", torch.version.cuda)
    show("torch_cudnn", torch.backends.cudnn.version())
    show("cuda_available", torch.cuda.is_available())
    show("arch_list", torch.cuda.get_arch_list() if torch.cuda.is_available() else None)
    n = torch.cuda.device_count()
    show("device_count", n)
    for i in range(n):
        pr = torch.cuda.get_device_properties(i)
        show(f"gpu[{i}]", f"{pr.name} total={pr.total_memory/2**30:.2f}GiB cap={torch.cuda.get_device_capability(i)}")
    if n:
        show("is_bf16_supported", torch.cuda.is_bf16_supported())
        try:
            show("is_bf16_supported_no_emulation", torch.cuda.is_bf16_supported(including_emulation=False))
        except Exception as e:  # noqa: BLE001
            show("is_bf16_supported_no_emulation", repr(e))
        import time

        import torch.nn.functional as F

        for dt in (torch.float16, torch.bfloat16, torch.float32):
            try:
                q = torch.randn(1, 8, 2048, 64, device="cuda", dtype=dt)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(20):
                    o = F.scaled_dot_product_attention(q, q, q, is_causal=True)
                torch.cuda.synchronize()
                a = torch.randn(4096, 4096, device="cuda", dtype=dt)
                t1 = time.time()
                for _ in range(20):
                    b = a @ a
                torch.cuda.synchronize()
                show(f"bench[{dt}]", f"sdpa20={t1-t0:.3f}s matmul20={time.time()-t1:.3f}s finite={bool(torch.isfinite(o).all())}")
            except Exception as e:  # noqa: BLE001
                show(f"bench[{dt}]", repr(e))
        show("sdp_backends", f"flash={torch.backends.cuda.flash_sdp_enabled()} mem_eff={torch.backends.cuda.mem_efficient_sdp_enabled()} math={torch.backends.cuda.math_sdp_enabled()}")
except Exception as e:  # noqa: BLE001
    show("torch_error", repr(e))

try:
    import jax

    show("jax", jax.__version__)
    show("jax_devices", [str(d) for d in jax.devices()])
    show("jax_device_count", jax.device_count())
except Exception as e:  # noqa: BLE001
    show("jax_error", repr(e)[:300])

try:
    import tensorflow as tf

    show("tf", tf.__version__)
    show("tf_gpus", [d.name for d in tf.config.list_physical_devices("GPU")])
except Exception as e:  # noqa: BLE001
    show("tf_error", repr(e)[:300])

for url in ["https://huggingface.co", "https://pypi.org/simple/pip/", "https://github.com", "https://storage.googleapis.com"]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=15) as r:
            show(f"net[{url}]", r.status)
    except Exception as e:  # noqa: BLE001
        show(f"net[{url}]", repr(e)[:200])

show("flash_attn_wheel_dry_run", sh("pip download flash-attn --no-deps --no-build-isolation -d /tmp/_fa 2>&1 | tail -3"))
show("done", True)
