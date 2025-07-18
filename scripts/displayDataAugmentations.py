import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, array_to_img
#tf.enable_eager_execution()

from matplotlib import pyplot as plt

from object_detection.core import preprocessor


## Load image
img = load_img('frame_72.png', color_mode='rgb', target_size=(300,300))

## Display image
plt.axis("off")
plt.imshow(img)

img_tf = tf.convert_to_tensor(np.asarray(img))
img_tf = tf.image.resize(img_tf, (300, 300))
img_tf = tf.cast(img_tf, tf.float32)


def random_horizontal_flip():
    out_tf = preprocessor.random_horizontal_flip(img_tf)
    return array_to_img(out_tf[0])

def random_black_patches():
    out_tf = preprocessor.random_black_patches(img_tf)
    return array_to_img(out_tf)

def random_adjust_brightness():
    out_tf = preprocessor.random_adjust_brightness(img_tf, max_delta=0.5)
    return array_to_img(out_tf)

display_size = 3  # n x n
    
for i in range(display_size**2):
    plt.subplot(display_size,display_size,i+1)
    plt.imshow(random_adjust_brightness(), aspect='auto')
    plt.axis("off")

plt.show()
