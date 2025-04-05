FROM debian:trixie-slim
ENV DEBIAN_FRONTEND=noninteractive
ENV FFMPEG_VERSION=5.1.4

# Install Debian Packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    autoconf \
    automake \
    libtool \
    pkg-config \
    git \
    wget \
    curl \
    python3-dev \
    python3-pip \
    python3-venv \
    cython3 \
    libzimg2 \
    libzimg-dev \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    ninja-build \
    meson \
    cmake \
    libboost-all-dev \
    nasm \
    yasm \
    make \
    gcc \
    g++ \
    libsdl1.2-dev \
    libass-dev \
    libfreetype6-dev \
    libtheora-dev \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    texinfo \
    zlib1g-dev \
    libx264-dev \
    libmp3lame-dev \
    libopus-dev \
    libvpx-dev \
    libx265-dev \
    libnuma-dev \
    libasound2 \
    libass9 \
    libvdpau1 \
    libva-x11-2 \
    libva-drm2 \
    libxcb-shape0 \
    libvorbisenc2 \
    libtheora0 \
    libaribb24-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

 RUN echo "deb http://http.us.debian.org/debian trixie main contrib non-free" | tee -a /etc/apt/sources.list && \
     apt-get update && \
     apt-get install -y libfdk-aac-dev

# Initialize Python Environment
WORKDIR /app
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

## Install ffmpeg
RUN mkdir /tmp/ffmpeg_sources && \
    cd /tmp/ffmpeg_sources && \
    curl -fsSL http://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.bz2 | tar -xj --strip-components=1 && \
    ./configure \
    --prefix=/usr/local \
    --disable-shared \
    --pkg-config-flags=--static \
    --enable-gpl \
    --enable-libass \
    --enable-libfreetype \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libtheora \
    --enable-libvorbis \
    --enable-libvpx \
    --enable-libx264 \
    --enable-libx265 \
    --enable-version3 \
    --enable-libaribb24 \
    --enable-nonfree \
    --disable-debug \
    --disable-doc \
    --enable-vaapi \
    --enable-libfdk-aac \
    && \
    make -j$(nproc) && \
    make install

# Install VapourSynth
RUN git clone https://github.com/vapoursynth/vapoursynth.git /opt/vapoursynth && \
    cd /opt/vapoursynth && \
    ./autogen.sh && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Install L-SMASH
RUN cd /tmp && \
    git clone https://github.com/l-smash/l-smash.git && \
    cd l-smash && \
    ./configure --enable-shared && \
    make -j$(nproc) && \
    make install

# Install L-SMASH-Works
RUN cd /tmp && \
    git clone -b 20220505 https://github.com/plife18/L-SMASH-Works.git && \
    cd L-SMASH-Works/VapourSynth && \
    LDFLAGS="-Wl,-Bsymbolic" meson build && \
    cd build && \
    ninja -v && \
    ninja install

# Copy and Install Python Packages
COPY app/requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Application Code
COPY . /app/

# Run FastAPI
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
