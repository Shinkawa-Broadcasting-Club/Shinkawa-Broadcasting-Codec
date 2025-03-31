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
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Initialize Python Environment
WORKDIR /app
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install VapourSynth
RUN git clone https://github.com/vapoursynth/vapoursynth.git /opt/vapoursynth && \
    cd /opt/vapoursynth && \
    ./autogen.sh && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Copy and Install Python Packages
COPY app/requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Application Code
COPY . /app/

# Run FastAPI
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
