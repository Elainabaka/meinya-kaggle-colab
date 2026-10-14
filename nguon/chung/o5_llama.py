# Ô 5 dùng chung cho notebook chạy llama.cpp (chèn bằng "#@chen chung/o5_llama.py").
# Cần từ ô 1–3: MODEL_DIR, CHAY_NEN, HF_TOKEN; từ ô 2: MODEL_REPO, MODEL_FILE, MODEL_GB, VISION, MMPROJ (None hoặc
# (tên file, GB) hoặc (tên file, GB, repo)), MTP, MTP_FILE (None khi MTP nằm sẵn trong file model, hoặc (repo, tên, GB)).
# Đặt cho ô 6: model_path, mmproj_path, mtp_path.

# %%
# 5) Model: dùng file đã gắn ở /kaggle/input nếu có, không thì tải từ Hugging Face về SCRATCH
def lay_file(ten, gb, repo=None):
    inp = Path("/kaggle/input")
    co_san = next((p for p in inp.rglob(Path(ten).name) if p.is_file()), None) if inp.exists() else None
    if co_san:
        print(f"  dùng file đã gắn: {co_san} ({co_san.stat().st_size / 1e9:.2f} GB)")
        return co_san
    dst = MODEL_DIR / ten
    if dst.exists() and dst.stat().st_size >= gb * 1e9 * 0.98:
        print(f"  đã có: {dst}")
        return dst
    trong = shutil.disk_usage(MODEL_DIR).free / 1e9
    if trong < gb + 2:
        raise SystemExit(f"Đĩa {MODEL_DIR} còn {trong:.1f} GB, không đủ cho file {gb} GB. Chọn QUANT nhẹ hơn ở ô 2.")
    if CHAY_NEN:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"      # chạy nền thì thanh tiến độ chỉ làm rối log
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    p = Path(hf_hub_download(repo or MODEL_REPO, ten, local_dir=str(MODEL_DIR), token=HF_TOKEN or None))
    dt = time.time() - t0
    print(f"  tải {ten}: {p.stat().st_size / 1e9:.2f} GB trong {dt:.0f} giây ({p.stat().st_size / 1e6 / max(dt, 1):.0f} MB/giây)")
    return p


model_path = lay_file(MODEL_FILE, MODEL_GB)
mmproj_path = lay_file(*MMPROJ) if VISION and MMPROJ else None
mtp_path = lay_file(MTP_FILE[1], MTP_FILE[2], repo=MTP_FILE[0]) if MTP and MTP_FILE else None
print("[OK] Model:", model_path)
