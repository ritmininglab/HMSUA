import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


import matplotlib.pyplot as plt
from skimage.transform import resize as imresize
import scipy.io
import scipy.misc
import numpy as np
from PIL import ImageTk, Image
from imageio import imread, imwrite


def visualizedata(img2):
    if np.max(img2) > 1:
        img2 = img2.astype("uint8")
    plt.figure(figsize=(20, 20))
    plt.imshow(img2)
    plt.axis("off")
    plt.show()


def visualizeraw(i, imgpath):
    img = imread(imgpath[i])
    resized = imresize(img, (224, 224))
    plt.figure(figsize=(20, 20))
    plt.imshow(resized)
    plt.axis("off")
    plt.show()


def translate(i, dictionary, prednum, printstr="pred", fprint=True):
    Tkey = prednum.shape[1]
    matchnow = prednum[i, 0:Tkey]
    words = ""
    for t in range(Tkey):
        wordidx = matchnow[t]
        if wordidx == 2:
            break
        words += dictionary[wordidx] + " "
    if fprint:
        print(str(i) + " " + printstr + ": " + words)
    return words


def translatekw(i, dictionary, prednum, printstr="pred", fprint=True):
    Tkey = prednum.shape[1]
    matchnow = prednum[i, 0:Tkey]
    words = ""
    for t in range(Tkey):
        if matchnow[t] != 0:
            words += dictionary[t] + " "
    if fprint:
        print(str(i) + " " + printstr + ": " + words)
    return words
