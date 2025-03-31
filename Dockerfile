FROM debian:trixie-slim
ENV DEBIAN_FRONTEND=noninteractive

# Install Debian Packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    autoconf \
    automake \
    libtool \
    pkg-config \
    git \
    wget \
    python3-dev \
    python3-pip \
    python3-venv \
    cython3 \
    libzimg2 \
    libzimg-dev \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install VapourSynth
RUN git clone https://github.com/vapoursynth/vapoursynth.git /usr/src/vapoursynth && \
    cd /usr/src/vapoursynth && \
    ./autogen.sh && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Initialize Python environment
WORKDIR /app
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY ./app /app
RUN pip install --no-cache-dir -r requirements.txt

# Run FastAPI
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]