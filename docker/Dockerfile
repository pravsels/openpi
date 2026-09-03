# openpi training container 
FROM nvcr.io/nvidia/pytorch:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_LFS_SKIP_SMUDGE=1 \
    UV_PYTHON=3.11 \
    UV_PROJECT_ENVIRONMENT=/.venv

# System deps + uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git git-lfs ffmpeg libgl1 build-essential cmake pkg-config \
        libavcodec-dev libavformat-dev libswscale-dev libavfilter-dev libavdevice-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv

# Install a Python 3.11 runtime for wheels that are missing on 3.12 (e.g., mujoco).
RUN uv python install 3.11


WORKDIR /workspace/repo

# Install dependencies at build time.
# RCW task-index remap (de4e4eb) is only on git main, not PyPI. Both report
# 0.2.18, so a stale lock or cache can leave the image on the PyPI wheel.
# Force git main after uv sync. RLT also uv-syncs a mounted repo .venv at
# launch; this keeps image /.venv safe for scripts that still use it.
COPY . /workspace/repo
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --group dev && \
    uv pip install -e . && \
    uv pip install "robocandywrapper @ git+https://github.com/villekuosmanen/RoboCandyWrapper.git@main"

# Install decord into the uv environment (source for arm64, pip for amd64).
# Arm64 fix: patch FFmpeg includes + AVCodec constness to match newer FFmpeg headers.
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        git clone --recursive https://github.com/dmlc/decord.git && \
        cd decord && \
        sed -i 's/#include <libavcodec\/avcodec.h>/#include <libavcodec\/avcodec.h>\n#include <libavcodec\/bsf.h>/' src/video/ffmpeg/ffmpeg_common.h && \
        sed -i 's/AVCodec \*dec/const AVCodec \*dec/g' src/video/video_reader.cc && \
        mkdir build && cd build && \
        cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release && \
        make -j$(nproc) && \
        cd ../python && uv pip install . && \
        cd ../.. && rm -rf decord; \
    else \
        uv pip install --prerelease=allow decord; \
    fi

CMD ["/bin/bash"]

