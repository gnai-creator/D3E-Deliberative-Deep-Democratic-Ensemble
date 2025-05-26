FROM nvcr.io/nvidia/tensorflow:23.12-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive

# Atualiza e instala dependências básicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    sudo \
    && rm -rf /var/lib/apt/lists/*
    
RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN apt-get update && apt-get install -y git-lfs && git lfs install

RUN apt-get install -y locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


# Configura usuário vscode, porque o VSCode acha que é especial
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd \
    && chmod 0440 /etc/sudoers.d/nopasswd

# Cria symlink (só pra manter a paz com o VSCode)
RUN ln -s /usr/bin/python3 /usr/local/bin/python

# Atualiza pip e instala libs básicas
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copia requirements.txt e instala dependências do projeto
# COPY requirements.txt /tmp/requirements.txt
# RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Define diretório de trabalho
WORKDIR /workspace

# Muda pro usuário civilizado
USER $USERNAME
