FROM tensorflow/tensorflow:2.15.0-gpu

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# Instala dependências básicas do sistema
RUN apt update && \
    apt install -y sudo git build-essential cmake curl wget \
    python3-dev python3-distutils libopenblas-dev ninja-build && \
    # Criação do usuário
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    # Atualiza o pip e cria o symlink do python
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    ln -s /usr/bin/python3 /usr/local/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

# Copia requirements.txt antes de mudar de usuário (evita permissão boba)
COPY requirements.txt /tmp/requirements.txt

# Instala os pacotes Python
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Define diretório de trabalho
WORKDIR /workspace

# Muda pro usuário padrão (menos perigoso que root)
USER $USERNAME
