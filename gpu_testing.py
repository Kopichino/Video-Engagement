import torch
import tensorflow as tf

print("PyTorch CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("PyTorch Device Name:", torch.cuda.get_device_name(0))

print("TensorFlow GPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("TensorFlow Device Name:", tf.config.list_physical_devices('GPU')[0])