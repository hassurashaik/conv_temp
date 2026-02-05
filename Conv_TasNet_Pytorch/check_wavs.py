import torchaudio
from pathlib import Path

bad = []
root = Path("wsj0-mix/2speakers/wav8k/min")

for wav in root.rglob("*.wav"):
    try:
        torchaudio.load(str(wav))
    except Exception:
        bad.append(wav)

print("Broken:", len(bad))
for b in bad[:10]:
    print(b)
