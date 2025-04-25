import os
from PIL import Image
import matplotlib.pyplot as plt

# 1) Point this at your downloaded dataset folder
dataset_dir = "data/dataset_low_res"

# 2) Lists to collect stats
widths = []
heights = []
mismatches = []        # will store tuples (path, width, height)
mismatch_paths = []    # just the filepaths

# 3) Walk through every subfolder & file
for root, _, files in os.walk(dataset_dir):
    for fname in files:
        # only consider common image extensions
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
            continue

        path = os.path.join(root, fname)
        try:
            w, h = Image.open(path).size
        except Exception as e:
            print(f"⚠️  Skipping unreadable file: {path}")
            continue

        # record for distributions
        widths.append(w)
        heights.append(h)

        # if it isn’t 640×512, note it
        if (w, h) != (640, 512):
            mismatches.append((path, w, h))
            mismatch_paths.append(path)

# 4) Print a summary
total = len(widths)
good  = total - len(mismatches)
bad   = len(mismatches)
print(f"\nChecked {total} images total.")
print(f"   • Exactly 640×512: {good}  ({100*good/total:.1f}%)")
print(f"   • Mismatches:      {bad}  ({100*bad/total:.1f}%)")

if bad:
    print("\nSample of files NOT 640×512:")
    for p, w, h in mismatches[:10]:
        print(f"   • {w}×{h} → {p}")

# 5) Plot width/height histograms
plt.figure(figsize=(6,3))
plt.hist(widths,  bins=50)
plt.title("Image Width Distribution")
plt.xlabel("Width (px)")
plt.ylabel("Count")
plt.tight_layout()

plt.figure(figsize=(6,3))
plt.hist(heights, bins=50)
plt.title("Image Height Distribution")
plt.xlabel("Height (px)")
plt.ylabel("Count")
plt.tight_layout()

plt.show()

# Now: 
#  • `mismatches` holds [(path, w, h), …]
#  • `mismatch_paths` holds just [path, …]
