FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install Packages
RUN apt-get update && apt-get install -y \
    build-essential \
    autoconf \
    automake \
    libtool \
    pkg-config \
    curl \
    git \
    xz-utils \
    nasm \
    yasm \
    cmake \
    libevent-dev \
    libjpeg-dev \
    libgif-dev \
    libpng-dev \
    libwebp-dev \
    libmagickcore-6.q16-6-extra \
    libmagickwand-6.q16-6 \
    libmemcached-dev \
    zlib1g-dev \
    libopencv-dev \
    ocl-icd-libopencl1 \
    opencl-headers \
    libboost-filesystem-dev \
    libboost-system-dev \
    python3-dev \
    cython3 \
    wget \
    ffmpeg \
    software-properties-common \
    && apt-get clean

# Install Latest libstdc++6
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
apt-get update && \
apt-get install -y libstdc++6

# Install zimg
RUN wget http://ftp.jp.debian.org/debian/pool/main/z/zimg/libzimg2_3.0.5+ds1-1+b2_amd64.deb && \
    dpkg -i libzimg2_3.0.5+ds1-1+b2_amd64.deb && \
    rm libzimg2_3.0.5+ds1-1+b2_amd64.deb

RUN wget http://ftp.jp.debian.org/debian/pool/main/z/zimg/libzimg-dev_3.0.5+ds1-1+b2_amd64.deb && \
    dpkg -i libzimg-dev_3.0.5+ds1-1+b2_amd64.deb && \
    rm libzimg-dev_3.0.5+ds1-1+b2_amd64.deb

# Install Vapoursynth
RUN git clone https://github.com/vapoursynth/vapoursynth.git /usr/src/vapoursynth && \
    cd /usr/src/vapoursynth && \
    ./autogen.sh && \
    ./configure && \
    make -j4 && \
    make install && \
    ldconfig && \
    python3 ./setup.py build && \
    python3 ./setup.py install

# Initialize Python
WORKDIR /app
COPY ./app /app
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Run Flask
CMD ["python3", "api.py"]