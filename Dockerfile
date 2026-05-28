FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG MODEL_NAME=openbmb/VoxCPM2

# System deps required by soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HOME=/root/.cache/huggingface
ENV VOXCPM_MODEL=${MODEL_NAME}

# Python deps. hf_transfer gives us the Rust-based HF downloader.
RUN python3 -m pip install --no-cache-dir \
    voxcpm \
    soundfile \
    numpy \
    hf_transfer \
    huggingface_hub

# Bake the model weights into the image so workers don't have to download
# them at boot. Build and push this image, then set VAST_IMAGE to its URI.
RUN python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['VOXCPM_MODEL'])"
