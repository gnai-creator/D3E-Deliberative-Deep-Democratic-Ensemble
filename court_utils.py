# court_logic.py — Sistema de julgamento simbólico com modelos cooperativos e adversariais
import numpy as np
import tensorflow as tf
from metrics_utils import salvar_voto_visual, preparar_voto_para_visualizacao
from runtime_utils import log
from metrics_utils import ensure_numpy
from models_loader import load_model

# Converte logits para rótulos se necessário
def normalizar_y_para_sparse(y):
    log(f"[DEBUG] Y SHAPE: {y.shape}")
    if y.shape.rank == 4 and y.shape[-1] != 1:
        y = tf.argmax(y, axis=-1)
        y = tf.expand_dims(y, axis=-1)
    return y

def garantir_dict_votos_models(votos_models):
    if isinstance(votos_models, dict):
        return votos_models
    elif isinstance(votos_models, list):
        return {f"modelo_{i}": v for i, v in enumerate(votos_models)}
    else:
        log(f"[SECURITY] votos_models tinha tipo inesperado: {type(votos_models)}. Substituindo por dict vazio.")
        return {}

def pixelwise_mode(stack):
    stack = tf.transpose(stack, [1, 2, 3, 0])
    flat = tf.reshape(stack, (-1, stack.shape[-1]))
    def pixel_mode(x):
        with tf.device("/CPU:0"):
            bincount = tf.math.bincount(tf.cast(x, tf.int32), minlength=10)
        return tf.argmax(bincount)
    moda_flat = tf.map_fn(pixel_mode, flat, fn_output_signature=tf.int64)
    return tf.reshape(moda_flat, (1, 30, 30))

def pad_or_truncate_channels(tensor, target_channels=40):
    """
    Ajusta o número de canais no último eixo, preservando as demais dimensões.
    """
    current_channels = tensor.shape[-1]
    rank = len(tensor.shape)

    if current_channels == target_channels:
        return tensor
    elif current_channels < target_channels:
        padding = target_channels - current_channels
        paddings = [[0, 0]] * rank
        paddings[-1] = [0, padding]
        return tf.pad(tensor, paddings)
    else:
        return tensor[..., :target_channels]


def prepare_input_for_model(model_index, base_input):
    # Garante que input seja 4D antes de truncar canais
    log(f"[DEBUG] PREPARE_INPUT_FOR_MODEL BASEINPUT SHAPE : {base_input.shape}")
    log(f"[DEBUG] PREPARE_INPUT_FOR_MODEL MODEL INDEX : {model_index}")
    if len(base_input.shape) == 5:
        base_input = safe_total_squeeze(base_input)  # remove eixo de tempo (ex: 1)
    
    # Aplica truncagem/padding
    if model_index in [0, 1, 2, 3]:
        x = pad_or_truncate_channels(base_input, 4)
    else:
        x = pad_or_truncate_channels(base_input, 40)

    # Agora adiciona eixo do tempo
    if len(x.shape) == 4:
        x = tf.expand_dims(x, axis=-2)
    elif len(x.shape) != 5:
        raise ValueError(f"[SECURITY] Entrada inválida para modelo_{model_index}: {x.shape}")

    log(f"[DEBUG] Modelo_{model_index} - shape final antes de entrar no modelo: {x.shape}")
    return x





def gerar_visualizacao_votos(votos_models, input_tensor_outros, idx, block_idx, task_id):
    votos_models = garantir_dict_votos_models(votos_models)
    votos_visuais = []
    try:
        for v in votos_models.values():
            # log(f"[DEBUG] preparando voto: type={type(v)}, shape={getattr(v, 'shape', 'indefinido')}")
            resultado = preparar_voto_para_visualizacao(v)
            if resultado is not None:
                votos_visuais.append(resultado)
    except Exception as e:
        log(f"[VISUAL] Erro ao processar votos_models: {e}")
    try:
        input_tensor_outros = ensure_numpy(input_tensor_outros)
        if input_tensor_outros.ndim == 5:
            input_visual = input_tensor_outros[0, :, :, 0, 0]
        elif input_tensor_outros.ndim == 4:
            input_visual = input_tensor_outros[0, :, :, 0]
        elif input_tensor_outros.ndim == 3:
            input_visual = input_tensor_outros[0, :, :]
        else:
            raise ValueError("Shape inesperado")
        if input_visual.ndim != 2:
            raise ValueError("input_visual nao e 2D")
    except Exception as e:
        log(f"[VISUAL] input_visual com shape inesperado ({getattr(input_tensor_outros, 'shape', 'N/A')}): {e}")
        input_visual = tf.zeros((30, 30), dtype=tf.int32)
    salvar_voto_visual(votos_visuais, idx, block_idx, input_visual, task_id=task_id)

def treinar_promotor_inicial(models, input_tensor_outros, votos_models, epochs):
    votos_models = garantir_dict_votos_models(votos_models)
    juradas_preds = [votos_models[f"modelo_{i}"] for i in range(3)]
    juradas_classes = tf.stack([tf.argmax(p, axis=-1) for p in juradas_preds], axis=0)
    y_moda = pixelwise_mode(juradas_classes)
    y_antitese = 9 - y_moda
    y_antitese = tf.clip_by_value(y_antitese, 0, 9)
    y_antitese = tf.expand_dims(y_antitese, axis=-1)

    x_promotor = prepare_input_for_model(6, input_tensor_outros)
    log(f"[DEBUG] x_promotor shape: {x_promotor.shape}")
    log(f"[DEBUG] y_antitese shape: {y_antitese.shape}")
    log("[PROMOTOR] Treinando promotor com antítese da moda das juradas.")
    models[6].fit(x=x_promotor, y=y_antitese, epochs=epochs, verbose=0)

def instanciar_promotor_e_supremo(models):
    model = load_model(6, 0.0005)
    models.append(model)
    return models

def safe_squeeze(tensor, axis):
    shape = tf.shape(tensor)
    if tensor.shape.rank is not None and tensor.shape[axis] == 1:
        return tf.squeeze(tensor, axis=axis)
    else:
        log(f"[WARN] Tentativa de squeeze em dim {axis} com size != 1: {tensor.shape}")
        return tensor

def safe_total_squeeze(t):
    shape = t.shape
    if shape.rank is None:
        return tf.squeeze(t)
    axes = [i for i in range(shape.rank) if shape[i] == 1]
    return tf.squeeze(t, axis=axes)


def safe_squeeze_last_dim(t):
    return tf.squeeze(t, axis=-1) if t.shape.rank is not None and t.shape[-1] == 1 else t

def extrair_classes_validas(y_real, pad_value=0):
    log(f"[DEBUG] extrair_classes_validas — y_real.shape={y_real.shape}")
    y_real = safe_total_squeeze(y_real)
    y_real = tf.convert_to_tensor(y_real)

    try:
        log(f"[DEBUG] y_real preview: {y_real.numpy()[0, 0, 0]}")
    except:
        pass

    # Se for (H, W, 1, 4), extrai apenas o canal 0 (cor da classe)
    if y_real.shape.rank == 4 and y_real.shape[-1] == 4:
        y_real = y_real[..., 0]  # Extrai canal da classe

    y_real = tf.squeeze(y_real)

    valores = tf.unique(tf.reshape(y_real, [-1]))[0]
    valores = tf.cast(valores, tf.int32)
    valores_validos = tf.boolean_mask(valores, valores != pad_value)

    log(f"[DEBUG] Valores únicos: {valores.numpy().tolist()}")
    log(f"[DEBUG] Classes extraídas: {valores_validos.numpy().tolist()}")
    return valores_validos


def inverter_classes_respeitando_valores(y, classes_validas, pad_value=0):
    y = tf.convert_to_tensor(y)
    y_shape = tf.shape(y)
    original_rank = y.shape.rank

    # Squeeze apenas se a dimensão for 1
    if original_rank == 4 and y.shape[0] == 1:
        y = tf.squeeze(y, axis=0)
    if y.shape.rank == 3 and y.shape[-1] == 1:
        y = tf.squeeze(y, axis=-1)
    elif y.shape.rank == 3 and y.shape[0] == 1:
        y = y[0]

    y_flat = tf.reshape(y, [-1])
    classes_validas = tf.cast(classes_validas, tf.int32)
    y_flat = tf.cast(y_flat, tf.int32)

    valores, _, counts = tf.unique_with_counts(tf.reshape(classes_validas, [-1]))
    idx_sorted = tf.argsort(counts, direction="DESCENDING")
    valores_sorted = tf.gather(valores, idx_sorted)

    def fallback():
        return tf.fill(tf.shape(y_flat), pad_value)

    def alternar():
        a = valores_sorted[0]
        b = valores_sorted[1]
        cond_a = tf.equal(y_flat, a)
        cond_b = tf.equal(y_flat, b)
        resultado = tf.where(cond_a, b, tf.where(cond_b, a, pad_value))
        return resultado

    resultado_flat = tf.cond(tf.size(valores_sorted) < 2, fallback, alternar)
    resultado = tf.reshape(resultado_flat, tf.shape(y))  # (H, W)

    resultado = tf.expand_dims(resultado, axis=-1)  # (H, W, 1)
    resultado = tf.expand_dims(resultado, axis=0)   # (1, H, W, 1)
    return resultado



def filtrar_classes_respeitando_valores(y, classes_validas, pad_value=0, preserve_invalids=False):
    y = tf.convert_to_tensor(y)
    log(f"[DEBUG] filtrando classes — y.shape={y.shape}, classes_validas={classes_validas.numpy()}")

    # Ajusta para formato comum (H, W)
    if y.shape.rank == 4:
        y = tf.squeeze(y, axis=0)  # (H, W, C)
    if y.shape.rank == 3 and y.shape[-1] == 1:
        channel = y[..., 0]  # (H, W)
    elif y.shape.rank == 3:
        channel = y[0]  # assume (1, H, W)
    elif y.shape.rank == 2:
        channel = y
    else:
        raise ValueError(f"[filtrar_classes_respeitando_valores] Shape inesperado: {y.shape}")

    # Cria máscara das classes válidas
    mask = tf.reduce_any(tf.equal(channel[..., tf.newaxis], tf.cast(classes_validas, channel.dtype)), axis=-1)

    # Aplica filtro
    if preserve_invalids:
        filtrado = tf.where(mask, channel, channel)  # mantém valor original se for inválido
    else:
        filtrado = tf.where(mask, channel, tf.constant(pad_value, dtype=channel.dtype))  # substitui por pad_value

    # Reconstrói shape (1, H, W, 1)
    filtrado = tf.expand_dims(filtrado, axis=-1)  # (H, W, 1)
    filtrado = tf.expand_dims(filtrado, axis=0)   # (1, H, W, 1)
    
    return filtrado



def gerar_padrao_simbolico(x_input):
    """
    Gera uma previsão simbólica baseada em alternância pura entre valores presentes no input.
    Ideal para tarefas onde se espera generalização de padrões lógicos (ex: alternar pares de cores).
    """
    if x_input.shape.rank == 5:
        canal_cor = x_input[0, :, :, 0, 0]
    else:
        canal_cor = x_input[0, :, :]

    valores_unicos = tf.unique(tf.reshape(canal_cor, [-1]))[0]
    valores_unicos = tf.boolean_mask(valores_unicos, valores_unicos != 0)
    valores = tf.sort(valores_unicos)
    if tf.size(valores) < 2:
        return tf.zeros((1, 30, 30), dtype=tf.int32)

    a, b = valores[0], valores[1]
    padrao = np.zeros((30, 30), dtype=np.int32)
    for i in range(30):
        for j in range(30):
            if (i + j) % 2 == 0:
                padrao[i, j] = a.numpy()
            else:
                padrao[i, j] = b.numpy()
    return tf.convert_to_tensor(padrao[None, ...], dtype=tf.int32)


def treinar_modelo_com_y_sparse(modelo, x_input, y_input, epochs=1):
    """Treina modelo garantindo que y_input esteja no formato correto."""
    if y_input.shape.rank == 5:
        y_sparse = tf.argmax(y_input, axis=-1)  # (B, H, W, C)
    elif y_input.shape.rank == 4 and y_input.shape[-1] > 1:
        y_sparse = tf.argmax(y_input, axis=-1)
    else:
        y_sparse = tf.squeeze(y_input)

    # Garante que a forma esteja compatível
    tf.debugging.assert_shapes([
        (x_input, ('N', 30, 30, 10, None)),
        (y_sparse, ('N', 30, 30, 10)),
    ], message="Shape incompatível entre X e Y")

    log(f"[DEBUG] x_input shape: {x_input.shape}")
    log(f"[DEBUG] y_sparse shape: {y_sparse.shape}")

    modelo.fit(x=x_input, y=y_sparse, epochs=epochs, verbose=0)