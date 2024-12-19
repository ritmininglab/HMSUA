import os
from tensorflow import keras
import tensorflow as tf
import numpy as np
from imageio import imread, mimsave
import os
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt


MAX_CAP_LENGTH = 32
NUM_FEATURES = 1536
IMG_SIZE = 240

EPOCHS = 5


subfolder = "TrainValVideo"
datas = dict()

files = os.listdir(f"{subfolder}")

directory_intermediate = "frames_pkl"
if not os.path.exists(directory_intermediate):
    os.makedirs(directory_intermediate)
directory_features = "features_pkl"
if not os.path.exists(directory_features):
    os.makedirs(directory_features)


print(f"total files = {len(files)}")

load_frame = True
if not load_frame:
    import cv2

    def centercrop(img, cropwh=IMG_SIZE):
        dh = img.shape[0] // 2
        dw = img.shape[1] // 2
        return img[
            dh - cropwh // 2 : dh + cropwh // 2, dw - cropwh // 2 : dw + cropwh // 2, :
        ]

    def getFrames(filename, frameRate, gbr=True, cropwh=IMG_SIZE):
        vidcap = cv2.VideoCapture(filename)
        sec = 0
        frames = []
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()
        while hasFrames:
            if gbr:
                image = image[:, :, ::-1]
            if cropwh is not None:
                image = centercrop(image, cropwh)
            frames.append(image)
            sec = sec + frameRate
            sec = round(sec, 2)
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            hasFrames, image = vidcap.read()
        return frames

    for i in tqdm(range(len(files))):
        file = files[i]
        frames = getFrames(f"{subfolder}/{file}", frameRate=2.0)
        datas[file] = {
            "frames": [],
        }
        with open(f"{directory_intermediate}/{file}", "wb") as f:
            pickle.dump([fr for fr in frames], f)
    with open("datas0efficientnet.pkl", "wb") as f:
        pickle.dump(datas, f)
else:
    with open("datas0efficientnet.pkl", "rb") as f:
        datas = pickle.load(f)


from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras import Input, Model


def build_feature_extractor():
    feature_extractor = EfficientNetB1(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    inputs = Input((IMG_SIZE, IMG_SIZE, 3))
    outputs = feature_extractor(inputs)
    return Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()


import json

with open("train_val_videodatainfo.json") as f:
    cap_raw = json.load(f)


captions = dict()

for data in tqdm(cap_raw["sentences"]):
    videoid = data["video_id"]
    caption = data["caption"]
    path = f"{videoid}.mp4"
    if path not in datas:
        print(f"{videoid} in captions but not in videos")
        continue
    if "text" not in datas[path]:
        datas[path]["text"] = [caption.strip()]
    else:
        datas[path]["text"].append(caption.strip())


print("Tokenize data...")


sampling_interval = 1
MAX_SEQ_LENGTH = 16
idx = 0
for path in tqdm(datas):
    with open(f"{directory_intermediate}/{path}", "rb") as f:
        frames = pickle.load(f)
    subsampled = [frames[i] for i in range(len(frames)) if i % sampling_interval == 0]
    subsampled = subsampled[: min(MAX_SEQ_LENGTH, len(subsampled))]
    print(f"{path}: {len(subsampled)}")
    subsampled = np.stack(subsampled, axis=0)

    inputs = subsampled
    mask = np.zeros((MAX_SEQ_LENGTH,))
    mask[: min(MAX_SEQ_LENGTH, len(subsampled))] = 1
    features = feature_extractor.predict(inputs, verbose=0)
    datas[path]["features"] = []
    with open(f"{directory_features}/{path}", "wb") as f:
        pickle.dump(features, f)

    datas[path]["mask"] = mask


for j in range(4):
    img = inputs[j]
    plt.figure()
    plt.imshow(img.astype("uint8"))

with open("datas1efficientnet.pkl", "wb") as f:
    pickle.dump(datas, f)


from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re

MAX_VOCAB_SIZE = None


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


strip_chars = "!\"
strip_chars = strip_chars.replace("<", "")
strip_chars = strip_chars.replace(">", "")

vectorization = TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_CAP_LENGTH,
)
text_data = []
for path in tqdm(datas):
    datas[path]["text_refine"] = []
    for caption in datas[path]["text"]:
        caption = re.sub(f"[{strip_chars}]", "", caption)
        text_data.append(f"<sos> {caption.lower()} <eos>")
        datas[path]["text_refine"].append(caption.lower())


occurs = dict()
for caption in text_data:
    words = caption.split(" ")
    for word in words:
        if word not in occurs:
            occurs[word] = 1
        else:
            occurs[word] = occurs[word] + 1

min_occur = 4
hot_words = [word for word in list(occurs.keys()) if occurs[word] >= min_occur]


text_data_refine = []
for caption in tqdm(text_data):
    words = caption.split(" ")
    words_refine = [word for word in words if (word in hot_words)]
    text_data_refine.append(" ".join(words_refine))


vectorization.adapt(text_data_refine)
vocab = vectorization.get_vocabulary()
print(f"current vocab size is {len(vocab)}")
index_lookup = dict(zip(range(len(vocab)), vocab))


for path in tqdm(datas):
    captions = datas[path]["text_refine"]
    datas[path]["token"] = [
        vectorization(f"<sos> {caption} <eos>").numpy() for caption in captions
    ]

with open("datas2efficientnet.pkl", "wb") as f:
    pickle.dump([datas, index_lookup], f)
