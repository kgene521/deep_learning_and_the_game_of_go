import tensorflow as tf
from tensorflow.python.client import device_lib


local_devices = device_lib.list_local_devices()
print(local_devices)

# hello = tf.constant('Hello, TensorFlow!')




