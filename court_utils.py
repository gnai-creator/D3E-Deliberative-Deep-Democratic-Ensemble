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
    if model_index in [0, 1, 2, 3]:
        return pad_or_truncate_channels(base_input, 4)
    else:
        return pad_or_truncate_channels(base_input, 40)

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


def extrair_classes_validas(y_real, pad_value=0):
    y_real = tf.convert_to_tensor(y_real)
    log(f"[DEBUG] extrair_classes_validas — y_real.shape={y_real.shape}")

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
    """
    Inverte classes com base nos dois valores mais frequentes nas classes válidas.
    Se o valor de y é igual ao primeiro, troca pelo segundo e vice-versa.
    Se não for nenhum dos dois, retorna pad_value.
    """
    y = tf.convert_to_tensor(y)
    y_shape = tf.shape(y)
    original_rank = y.shape.rank

    # Garante que y tenha shape (H, W)
    if original_rank == 4:
        y = tf.squeeze(y, axis=0)
    if y.shape.rank == 3 and y.shape[-1] == 1:
        y = tf.squeeze(y, axis=-1)
    elif y.shape.rank == 3:
        y = y[0]  # assume (1, H, W)

    y_flat = tf.reshape(y, [-1])

    # Garante tipo consistente
    classes_validas = tf.cast(classes_validas, tf.int32)
    y_flat = tf.cast(y_flat, tf.int32)

    # Seleciona os dois valores mais frequentes
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

    # Reconstrói para shape (1, H, W, 1)
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