# Đo phép nhân trọng số NVFP4 trên TPU này, bằng đúng hai đường của tpu-inference 0.28.0, so với bf16.
# Chạy trong môi trường vLLM (ô 4 gọi), giữ TPU vài chục giây rồi trả lại. In một dòng "DO_NHAN {json}".
#   xla:    tpu_inference.layers.common.linear.xla_quantized_matmul (mặc định của tpu-inference)
#   kernel: tokamax gmm_v2 (bật bằng ENABLE_QUANTIZED_MATMUL_KERNEL=1; mặc định tắt vì từng ra NaN)
# Cỡ ma trận như MLP của Qwen3.8-27B, 64 token một lượt (bucket nhỏ nhất của vLLM TPU), 6 cặp lớp nối nhau.
# Biến môi trường DO_NHOM = số trọng số chung một thang (16 như checkpoint; 32, 64 khi ô 2 đặt NHOM_FP4 lớn hơn).
import json
import os
import time

import jax
import jax.numpy as jnp

M, K, N, L = (int(x) for x in os.environ.get("DO_CO", "64,5120,17408,6").split(","))     # DO_CO nhỏ: chỉ để thử trên CPU
B = int(os.environ.get("DO_NHOM", "16"))
kq = {"tpu": jax.devices()[0].device_kind, "so_chip": jax.device_count(), "nhom": B}


def luong_tu(w):
    """[K, N] f32 -> (fp4 [K, N], thang f32 [K/B, N]), cách tpu-inference giữ NVFP4 sau khi nạp."""
    wb = w.reshape(w.shape[0] // B, B, w.shape[1])
    s = jnp.maximum(jnp.max(jnp.abs(wb), axis=1), 1e-12) / 6.0
    return (wb / s[:, None, :]).astype(jnp.float4_e2m1fn).reshape(w.shape), s


def tao(seed, k, n):
    w = jax.random.normal(jax.random.PRNGKey(seed), (k, n), jnp.float32) * 0.02
    q, s = luong_tu(w)
    return q, s, (q.astype(jnp.float32).reshape(k // B, B, n) * s[:, None, :]).reshape(k, n).astype(jnp.bfloat16)


def chuan(x):
    x = x.astype(jnp.float32)
    return (x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)).astype(jnp.bfloat16)


def do(f, *a, lan=20):
    f(*a).block_until_ready()
    t = time.perf_counter()
    for _ in range(lan):
        r = f(*a)
    r.block_until_ready()
    return (time.perf_counter() - t) / lan


try:
    lop = [(tao(2 * i, K, N), tao(2 * i + 1, N, K)) for i in range(L)]
    QU, SU, WU = (jnp.stack([l[0][j] for l in lop]) for j in range(3))
    QD, SD, WD = (jnp.stack([l[1][j] for l in lop]) for j in range(3))
    del lop
except Exception as e:      # chip không đổi được sang fp4: vLLM cũng sẽ hỏng ở bước nạp NVFP4
    kq["tao_loi"] = f"{type(e).__name__}: {str(e)[:300]}"
    print("DO_NHAN " + json.dumps(kq, ensure_ascii=False), flush=True)
    raise SystemExit(0)
x0 = jax.random.normal(jax.random.PRNGKey(99), (M, K), jnp.bfloat16)
so_phan_tu = 2 * L * K * N
byte_moi_so = 0.5 + 4 / B      # fp4 + thang f32 chia cho cả nhóm


def chuoi(nhan):
    def f(x, *w):
        def buoc(x, wl):
            u, d = wl[: len(wl) // 2], wl[len(wl) // 2:]
            return chuan(nhan(chuan(nhan(x, *u)), *d)), None
        return jax.lax.scan(buoc, x, w)[0]
    return jax.jit(f)


bf16 = lambda x, w: jnp.dot(x, w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
t = do(chuoi(bf16), x0, WU, WD)
kq["bf16"] = {"ms": round(t * 1e3, 3), "gb_s": round(so_phan_tu * 2 / t / 1e9)}
ref = jnp.dot(x0.astype(jnp.float32), WU[0].astype(jnp.float32))

try:
    from tpu_inference.layers.common.linear import xla_quantized_matmul
    xla = lambda x, q, s: xla_quantized_matmul(x, q, s, quantize_activation=True)
    t = do(chuoi(xla), x0, QU, SU, QD, SD)
    y = xla(x0, QU[0], SU[0]).astype(jnp.float32)
    kq["xla"] = {"ms": round(t * 1e3, 3), "gb_s": round(so_phan_tu * byte_moi_so / t / 1e9),
                 "sai": float(jnp.max(jnp.abs(y - ref)) / jnp.max(jnp.abs(ref))), "nan": bool(jnp.isnan(y).any())}
except Exception as e:
    kq["xla"] = {"loi": f"{type(e).__name__}: {str(e)[:300]}"}

try:
    from tokamax._src.ops.experimental.gmm_v2.gmm_v2 import gmm_v2

    def ker(x, q, s):
        return gmm_v2(lhs=x, rhs=q[None], group_sizes=jnp.array([x.shape[0]], jnp.int32), rhs_scale=s[None, :, None, :],
                      rhs_bias=None, group_offset=jnp.array([0], jnp.int32), zero_initialize=False,
                      preferred_element_type=x.dtype, maybe_quantize_lhs=True)
    t = do(chuoi(ker), x0, QU, SU, QD, SD)
    y = jax.jit(ker)(x0, QU[0], SU[0]).astype(jnp.float32)
    kq["kernel"] = {"ms": round(t * 1e3, 3), "gb_s": round(so_phan_tu * byte_moi_so / t / 1e9),
                    "sai": float(jnp.max(jnp.abs(y - ref)) / jnp.max(jnp.abs(ref))), "nan": bool(jnp.isnan(y).any())}
except Exception as e:
    kq["kernel"] = {"loi": f"{type(e).__name__}: {str(e)[:300]}"}

print("DO_NHAN " + json.dumps(kq, ensure_ascii=False), flush=True)
