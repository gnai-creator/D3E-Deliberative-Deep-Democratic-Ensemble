FROM tensorflow/tensorflow:2.15.0-gpu

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt update && \
    apt install -y sudo git build-essential cmake curl wget \
    python3-dev python3-distutils libopenblas-dev ninja-build && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    ln -s /usr/bin/python3 /usr/local/bin/python && \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install scikit-learn pandas matplotlib seaborn numpy transformers tensorflow==2.15.0 tf_keras && \
    python -m pip install llama-cpp-python --extra-index-url=https://abetlen.github.io/llama-cpp-python/cu118
