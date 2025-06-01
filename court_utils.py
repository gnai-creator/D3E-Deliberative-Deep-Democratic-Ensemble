# court_logic.py — Sistema de julgamento simbólico com modelos cooperativos e adversariais
import numpy as np
import tensorflow as tf
from metrics_utils import preparar_voto_para_visualizacao
from runtime_utils import log
from metrics_utils import ensure_numpy, garantir_dict_votos_models
from models_loader import load_model
from graphics_utils import salvar_voto_visual
import sys
# Converte logits para rótulos se necessário
def normalizar_y_para_sparse(y):
    log(f"[DEBUG] Y SHAPE: {y.shape}")
    if y.shape.rank == 4 and y.shape[-1] != 1:
        y = tf.argmax(y, axis=-1)
        y = tf.expand_dims(y, axis=-1)
    return y



@tf.function
def pixelwise_mode(stack):
    """
    Calcula a moda (valor mais frequente) por pixel ao longo do eixo 0.
    Espera entrada com shape (N, H, W, C).
    Retorna tensor com shape (1, H, W, C, 1)
    """
    log(f"[DEBUG] pixelwise_mode - shape de entrada original: {stack.shape}")

    if tf.rank(stack) != 4:
        raise ValueError(f"[pixelwise_mode] Tensor esperado com rank=4, mas recebeu shape={stack.shape}")

    # Transpõe para (H, W, C, N)
    transposed = tf.transpose(stack, [1, 2, 3, 0])  # (H, W, C, N)
    flat = tf.reshape(transposed, [-1, tf.shape(transposed)[-1]])  # (H * W * C, N)

    # Força a operação de bincount e argmax na CPU
    @tf.function
    def bincount_mode(x):
        with tf.device('/CPU:0'):
            return tf.math.argmax(tf.math.bincount(tf.cast(x, tf.int32), minlength=10))

    moda_pixel = tf.map_fn(
        bincount_mode,
        flat,
        fn_output_signature=tf.int64
    )

    # Restaura shape: (1, H, W, C, 1)
    moda_reshaped = tf.reshape(moda_pixel, [1, tf.shape(stack)[1], tf.shape(stack)[2], tf.shape(stack)[3], 1])
    return tf.cast(moda_reshaped, tf.float32)






def pad_or_truncate_channels(tensor, target_channels=1):
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
    # log(f"[DEBUG] PREPARE_INPUT_FOR_MODEL BASEINPUT SHAPE : {base_input.shape}")
    # log(f"[DEBUG] PREPARE_INPUT_FOR_MODEL MODEL INDEX : {model_index}")
    # if len(base_input.shape) == 5:
    #     base_input = safe_total_squeeze(base_input)  # remove eixo de tempo (ex: 1)
    
    # Aplica truncagem/padding
    if model_index in [0, 1, 2, 3]:
        x = pad_or_truncate_channels(base_input, 1)
    else:
        x = pad_or_truncate_channels(base_input, 1)

    # # Agora adiciona eixo do tempo
    # if len(x.shape) == 4:
    #     x = tf.expand_dims(x, axis=-2)
    # elif len(x.shape) != 5:
    #     raise ValueError(f"[SECURITY] Entrada inválida para modelo_{model_index}: {x.shape}")

    log(f"[DEBUG] Modelo_{model_index} - shape final antes de entrar no modelo: {x.shape}")
    return x


def gerar_visualizacao_votos(votos_models, input_tensor_outros, input_tensor_train, iteracao, idx, block_idx, task_id, classes_validas, classes_objetivo, consenso):
    votos_models = garantir_dict_votos_models(votos_models)
    votos_visuais = []

    try:
        for i, (k, v) in enumerate(votos_models.items()):
            resultado = preparar_voto_para_visualizacao(v)
            if resultado is not None:
                votos_visuais.append(resultado)
    except Exception as e:
        log(f"[VISUAL] Erro ao processar votos_models: {e}")

    def extrair_input_visual(tensor, nome=""):
        try:
            tensor = ensure_numpy(tensor)
            if tensor.ndim == 5:
                return tensor[0, :, :, 0, 0]
            elif tensor.ndim == 4:
                return tensor[0, :, :, 0]
            elif tensor.ndim == 3:
                return tensor[0, :, :]
            else:
                raise ValueError(f"{nome} com shape inesperado: {tensor.shape}")
        except Exception as e:
            log(f"[VISUAL] erro ao extrair {nome}: {e}")
            return tf.zeros((30, 30), dtype=tf.int32)

    # input_visual_train = extrair_input_visual(input_tensor_train, "input_tensor_train")
    # input_visual_test = extrair_input_visual(input_tensor_outros, "input_tensor_outros")

    # Junta as duas imagens na vertical

    salvar_voto_visual(
        votos=votos_visuais,
        iteracao=iteracao,
        idx=idx,
        block_idx=block_idx,
        input_train=input_tensor_outros,
        input_test=input_tensor_train,
        task_id=task_id,
        filename=f"a.png",
        classes_validas=classes_validas,
        classes_objetivo=classes_objetivo,
        consenso=consenso
        # filename=f"voto_visual_idx{idx}_iter{iteracao}_bloco{block_idx}.png"

    )


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


def extrair_canal_cor(tensor_5d):
    """
    Recebe um tensor (1, 30, 30, 3, 1) e retorna apenas o canal da cor: (1, 30, 30, 1)
    """
    return tf.cast(tf.expand_dims(tensor_5d[:, :, :, 0, 0], axis=-1), tf.float32)  # (1, 30, 30, 1)

def expandir_para_3_canais(y_input):
    """
    Garante que y_input tenha shape (1, 30, 30, 3, 1), com valores int32.
    """
    if not isinstance(y_input, tf.Tensor):
        y_input = tf.convert_to_tensor(y_input)

    y_input = tf.cast(y_input, tf.int32)

    if y_input.shape.rank == 4:
        # (1, 30, 30, 1) → (1, 30, 30, 3)
        y_input = tf.repeat(y_input, repeats=3, axis=3)
        y_input = tf.expand_dims(y_input, axis=-1)
    elif y_input.shape.rank == 5 and y_input.shape[3] == 1:
        # (1, 30, 30, 1, 1) → (1, 30, 30, 3, 1)
        y_input = tf.repeat(y_input, repeats=3, axis=3)
    elif y_input.shape.rank == 5 and y_input.shape[3] == 3:
        pass  # já está no formato certo
    else:
        raise ValueError(f"[expandir_para_3_canais] Shape inválido: {y_input.shape}")

    return y_input


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


def extrair_classes_validas(y_real, pad_value=-1):
    log(f"[DEBUG] extrair_classes_validas — y_real.shape={y_real.shape}")
    y_real = tf.convert_to_tensor(y_real)

    # Extrai o canal de cor (canal 0 do eixo -2) se estiver em formato (1, 30, 30, 3, 1)
    if y_real.shape.rank == 5 and y_real.shape[-2] == 3:
        y_real = y_real[:, :, :, 0, :]  # Mantém o shape (1, 30, 30, 1)

    y_real = safe_total_squeeze(y_real)

    try:
        log(f"[DEBUG] y_real preview: {y_real.numpy()[0, 0]}")
    except:
        pass

    valores = tf.unique(tf.reshape(y_real, [-1]))[0]
    valores = tf.cast(valores, tf.int32)
    valores_validos = tf.boolean_mask(valores, valores != pad_value)

    log(f"[DEBUG] Valores únicos: {valores.numpy().tolist()}")
    log(f"[DEBUG] Classes extraídas: {valores_validos.numpy().tolist()}")
    return valores_validos

def extrair_todas_classes_validas(X_train, X_test, pad_value=-1):
    """
    Extrai as classes válidas presentes tanto no X_train quanto no X_test,
    considerando apenas o canal de cor (canal 0 do eixo -2).
    """
    def extrair_classes(x):
        x = tf.convert_to_tensor(x)

        # Extrai canal de cor se for (1, 30, 30, 3, 1)
        if x.shape.rank == 5 and x.shape[-2] == 3:
            x = x[:, :, :, 0, :]  # (1, 30, 30, 1)

        x = safe_total_squeeze(x)
        valores = tf.unique(tf.reshape(x, [-1]))[0]
        valores = tf.cast(valores, tf.int32)
        return tf.boolean_mask(valores, valores != pad_value)

    classes_train = extrair_classes(X_train)
    classes_test = extrair_classes(X_test)

    # Concatena e remove duplicatas
    classes_concatenadas = tf.concat([classes_train, classes_test], axis=0)
    classes_unicas = tf.unique(classes_concatenadas)[0]

    log(f"[DEBUG] Classes extraídas combinadas: {classes_unicas.numpy().tolist()}")
    return classes_unicas



def inverter_classes_respeitando_valores(y, classes_validas, pad_value=-1):
    """
    Recebe y com shape (1, 30, 30, 1, 1)
    Retorna o mesmo shape (1, 30, 30, 1, 1), invertendo os dois valores mais comuns presentes em `classes_validas`.
    """
    y = tf.convert_to_tensor(y)
    original_shape = tf.shape(y)
    original_rank = y.shape.rank

    log(f"[INVERTER] original rank {original_rank}")
    log(f"[INVERTER] y_shape {original_shape}")

    # Flatten para operação
    y_flat = tf.reshape(y, [-1])
    y_flat = tf.cast(y_flat, tf.int32)
    classes_validas = tf.cast(classes_validas, tf.int32)

    # Ordena classes por frequência
    valores, _, counts = tf.unique_with_counts(tf.reshape(classes_validas, [-1]))
    idx_sorted = tf.argsort(counts, direction="DESCENDING")
    valores_sorted = tf.gather(valores, idx_sorted)

    # Fallback se não houver pelo menos duas classes
    def fallback():
        return tf.fill(tf.shape(y_flat), pad_value)

    # Alterna entre as duas mais frequentes
    def alternar():
        a = valores_sorted[0]
        b = valores_sorted[1]
        cond_a = tf.equal(y_flat, a)
        cond_b = tf.equal(y_flat, b)
        return tf.where(cond_a, b, tf.where(cond_b, a, pad_value))

    resultado_flat = tf.cond(tf.size(valores_sorted) < 2, fallback, alternar)

    # Retorna ao shape original (1, 30, 30, 1, 1)
    resultado = tf.reshape(resultado_flat, original_shape)
    return tf.cast(resultado, tf.float32)




def filtrar_classes_respeitando_valores(y, classes_validas, pad_value=-1, preserve_invalids=False):
    y = tf.convert_to_tensor(y)
    log(f"[DEBUG] filtrando classes — y.shape={y.shape}")

    # Serializa classes_validas para log de forma segura
    try:
        log(f"[DEBUG] classes_validas={[int(c) for c in classes_validas.numpy()]}")
    except Exception as e:
        log(f"[DEBUG] classes_validas={classes_validas} (não serializável diretamente: {e})")

    # Se o último canal for maior que 1, aplica argmax para obter rótulo
    if y.shape.rank == 5 and y.shape[-1] != 1:
        log("[DEBUG] Aplicando argmax em y para reduzir canal de classe")
        y = tf.argmax(y, axis=-1)  # (1, H, W, C)
        y = tf.expand_dims(y, axis=-1)  # (1, H, W, C, 1)

    if y.shape.rank != 5 or y.shape[0] != 1 or y.shape[-1] != 1:
        raise ValueError(f"[filtrar_classes_respeitando_valores] Shape inesperado após ajuste: {y.shape}, esperado (1, H, W, C, 1)")

    # Remove batch e canal final → (H, W, C)
    y_squeezed = tf.squeeze(y, axis=[0, -1])  # (H, W, C)

    # Expande para comparar com classes válidas → (H, W, C, 1)
    y_exp = tf.expand_dims(y_squeezed, axis=-1)

    # Prepara classes válidas para broadcasting
    classes_validas = tf.convert_to_tensor(classes_validas)
    classes_validas = tf.cast(classes_validas, dtype=y.dtype)
    classes_exp = tf.reshape(classes_validas, shape=(1, 1, 1, -1))  # (1, 1, 1, N)

    # Máscara booleana para manter apenas valores válidos
    mask = tf.reduce_any(tf.equal(y_exp, classes_exp), axis=-1)  # (H, W, C)

    # Aplicação da máscara
    if preserve_invalids:
        filtrado = y_squeezed  # Mantém valores originais
    else:
        filtrado = tf.where(mask, y_squeezed, tf.constant(pad_value, dtype=y.dtype))  # Substitui inválidos

    # Restaura shape original: (1, H, W, C, 1)
    return tf.expand_dims(filtrado, axis=0)[..., tf.newaxis]








def gerar_padrao_simbolico(x_input, pad_value=-1):
    """
    Gera uma previsão simbólica baseada em alternância pura entre valores presentes no input.
    Ideal para inicialização de y_sup quando há colapso de classe.
    """
    if x_input.shape.rank == 5:
        canal_cor = x_input[0, :, :, 0, 0]  # assume canal simbólico
    else:
        canal_cor = x_input[0, :, :]

    valores_unicos = tf.unique(tf.reshape(canal_cor, [-1]))[0]
    valores_unicos = tf.boolean_mask(valores_unicos, valores_unicos != pad_value)
    valores = tf.sort(valores_unicos)

    if tf.size(valores) < 2:
        return tf.zeros((1, 30, 30), dtype=tf.int32)

    a, b = valores[0], valores[1]
    padrao = np.zeros((30, 30), dtype=np.int32)
    for i in range(30):
        for j in range(30):
            padrao[i, j] = a.numpy() if (i + j) % 2 == 0 else b.numpy()

    return tf.convert_to_tensor(padrao[None, ...], dtype=tf.int32)
def treinar_modelo_com_y_sparse(modelo, x_input, y_input, epochs=1):
    """
    Treina o modelo com:
    - x_input: (1, 30, 30, 3, 1)
    - y_input: (1, 30, 30, 3)
    """
    pad_value = -1

    log(f"[DEBUG] x_input shape inicial: {x_input.shape}")
    log(f"[DEBUG] y_input shape inicial: {y_input.shape}")

    if isinstance(x_input, np.ndarray):
        x_input = tf.convert_to_tensor(x_input, dtype=tf.float32)
    if isinstance(y_input, np.ndarray):
        y_input = tf.convert_to_tensor(y_input, dtype=tf.float32)

    if x_input.shape != (1, 30, 30, 3, 1):
        raise ValueError(f"[ERRO] x_input esperado com shape (1, 30, 30, 3, 1), mas recebeu {x_input.shape}")

    # Ajustar y_input para shape (1, 30, 30, 3, 1)
    if y_input.shape.rank == 4:
        y_input = tf.expand_dims(y_input, axis=-1)
    elif y_input.shape.rank != 5:
        raise ValueError(f"[ERRO] y_input com rank inválido: {y_input.shape}")

    if y_input.shape != (1, 30, 30, 3, 1):
        raise ValueError(f"[ERRO] y_input esperado com shape (1, 30, 30, 3, 1), mas recebeu {y_input.shape}")

    # Criar sample_weight com 1 para válidos e 0 para pad_value
    sample_weight = tf.cast(tf.not_equal(y_input, pad_value), tf.float32)

    log(f"[DEBUG] sample_weight shape: {sample_weight.shape}")

    modelo.fit(
        x=x_input,
        y=y_input,
        sample_weight=sample_weight,
        epochs=epochs,
        verbose=0
    )








# 2. Substituir valores em y_sup por essas classes
def mapear_cores_para_x_test(y_sup, cores_validas):
    """
    Reduz os valores de y_sup para as cores mais próximas das válidas (recolorização simbólica).
    """
    y = tf.convert_to_tensor(y_sup)
    y = tf.squeeze(y)

    # Mapear cada valor em y para a cor válida mais próxima
    def encontrar_mais_proxima(v):
        return min(cores_validas, key=lambda c: abs(c - int(v)))

    y_np = y.numpy()
    y_mapeado = np.vectorize(encontrar_mais_proxima)(y_np)
    return tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(y_mapeado, dtype=tf.int64), axis=0), axis=-1)




