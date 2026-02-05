import os

# =========================
# CHANGE THIS ONLY
# =========================
DATA_ROOT = r"C:\Users\Shaik\Documents\temp_conv\Conv-TasNet\Conv_TasNet_Pytorch\wsj0-mix\2speakers\wav8k\min"
OUT_DIR = "scp"

os.makedirs(OUT_DIR, exist_ok=True)

def write_scp(folder, scp_path):
    with open(scp_path, "w") as f:
        for root, _, files in os.walk(folder):
            files.sort()
            for file in files:
                if file.endswith(".wav"):
                    full_path = os.path.join(root, file)
                    f.write(f"{file} {full_path}\n")

# =========================
# TRAIN
# =========================
write_scp(os.path.join(DATA_ROOT, "tr", "mix"), os.path.join(OUT_DIR, "tr_mix.scp"))
write_scp(os.path.join(DATA_ROOT, "tr", "s1"),  os.path.join(OUT_DIR, "tr_s1.scp"))
write_scp(os.path.join(DATA_ROOT, "tr", "s2"),  os.path.join(OUT_DIR, "tr_s2.scp"))

# =========================
# VALIDATION
# =========================
write_scp(os.path.join(DATA_ROOT, "cv", "mix"), os.path.join(OUT_DIR, "cv_mix.scp"))
write_scp(os.path.join(DATA_ROOT, "cv", "s1"),  os.path.join(OUT_DIR, "cv_s1.scp"))
write_scp(os.path.join(DATA_ROOT, "cv", "s2"),  os.path.join(OUT_DIR, "cv_s2.scp"))

# =========================
# TEST
# =========================
write_scp(os.path.join(DATA_ROOT, "tt", "mix"), os.path.join(OUT_DIR, "tt_mix.scp"))
write_scp(os.path.join(DATA_ROOT, "tt", "s1"),  os.path.join(OUT_DIR, "tt_s1.scp"))
write_scp(os.path.join(DATA_ROOT, "tt", "s2"),  os.path.join(OUT_DIR, "tt_s2.scp"))

print("✅ SCP files generated successfully")
