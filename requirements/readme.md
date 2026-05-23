
# TDV Environment Setup for GH200 (CUDA 12.6, Aarch64)

Follow these steps to set up a reproducible environment for training TDV models on NVIDIA GH200 nodes.

---

### 1. Create Conda Environment
```bash
conda create -n tdv_reproduce python=3.11 -y
conda activate tdv_reproduce
```

---

### 2. Install Dependencies from `requirements/gh200.txt`
```bash
pip install -r requirements/gh200.txt --no-deps
```
**Note:** `--no-deps` ensures pip does not reinstall or rebuild dependencies already handled by Conda. This avoids conflicts and prevents heavy C/C++ libraries from being compiled from source on Aarch64.

---

### 3. Install PyTorch Nightly (CUDA 12.6)
```bash
pip install --pre torch torchaudio torchvision --index-url https://download.pytorch.org/whl/nightly/cu126
```

---

### 4. Install aiohttp using conda-forge (since normal pip install fails)
```bash
conda install -c conda-forge aiohttp
```

---

### 5. Install ffprobe and libiconv (if not present in the environment already)
```bash
conda install -c conda-forge libiconv ffmpeg
```

---

✅ After setup, verify installation:
```bash
python -c "import torch, aiohttp, cv2; print('Environment OK')"
ffprobe -version
```
