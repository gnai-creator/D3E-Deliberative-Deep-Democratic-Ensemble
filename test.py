import tensorflow as tf
print("Dispositivos disponíveis:", tf.config.list_physical_devices())
print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))
