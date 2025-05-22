def freeze_all_except_learned_color_permutation(model):
    for layer in model.layers:
        if hasattr(layer, 'learned_color_permutation'):
            # Deixa só o learned_color_permutation treinável
            layer.trainable = True
            if hasattr(layer.learned_color_permutation, 'trainable'):
                layer.learned_color_permutation.trainable = True
        else:
            layer.trainable = False

def unfreeze_all(model):
    for layer in model.layers:
        layer.trainable = True
        # Também garante subcomponentes liberados
        if hasattr(layer, 'learned_color_permutation'):
            layer.learned_color_permutation.trainable = True

def freeze_learned_color_permutation_only(model):
    for layer in model.layers:
        if hasattr(layer, 'learned_color_permutation'):
            if hasattr(layer.learned_color_permutation, 'trainable'):
                layer.learned_color_permutation.trainable = False

def freeze_color_permutation(model):
    for layer in model.layers:
        if hasattr(layer, 'learned_color_permutation'):
            layer.learned_color_permutation.permutation_weights._trainable = False

def unfreeze_color_permutation(model):
    for layer in model.layers:
        if hasattr(layer, 'learned_color_permutation'):
            layer.learned_color_permutation.permutation_weights._trainable = True
