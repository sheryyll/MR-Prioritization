# Cell 1 - Environment check
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# %% Cell 2 - Clean clone and install
%cd /kaggle/working
!rm -rf MR-Prioritization
!git clone https://github.com/sheryyll/MR-Prioritization.git
%cd MR-Prioritization
!pip install -e . --no-deps -q

import sys
src_path = "/kaggle/working/MR-Prioritization/src"
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from mrrank import config
print("LABEL_CORRUPTION_EPOCHS:", config.LABEL_CORRUPTION_EPOCHS)
print("MUTANT_ACC_MIN/MAX:", config.MUTANT_ACC_MIN, config.MUTANT_ACC_MAX)

# %% Cell 3 - Sanity check: confirm this is the latest code
!grep -n "LABEL_CORRUPTION_EPOCHS" src/mrrank/config.py
!grep -n "generate_label_corruption_mutants" src/mrrank -r

# %% Cell 4 - Generate all 20 label-corruption mutants (~4-5 hours total)
!python -m mrrank.generate_label_corruption_mutants --epochs 20 --num-workers 2

# %% Cell 5 - Package mutants + manifest for download
!mkdir -p /kaggle/working/label_mutants_export
!cp mutants/LC_*.pth /kaggle/working/label_mutants_export/
!cp outputs/label_corruption_manifest.json /kaggle/working/label_mutants_export/
!cd /kaggle/working/label_mutants_export && zip -r /kaggle/working/label_mutants.zip .
print("Zipped. Download via Cell 6.")

# %% Cell 6 - Download the zip
import base64
from IPython.display import HTML

def make_download_link(filepath, filename):
    with open(filepath, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    size_mb = len(data) / (1024 * 1024)
    html = f'''<a download="{filename}" href="data:application/octet-stream;base64,{b64}" 
    style="font-size:16px; padding:8px; background:#20beff; color:white; text-decoration:none; border-radius:4px;">
    Download {filename} ({size_mb:.1f} MB)</a>'''
    return HTML(html)

display(make_download_link('/kaggle/working/label_mutants.zip', 'label_mutants.zip'))