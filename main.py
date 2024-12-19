import numpy as np
import pickle
import os
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback
from keras.models import load_model
from keras_nlp.models import GemmaTokenizer
import tensorflow as tf
from tqdm import tqdm
import random
from random import sample
from config import vocsize, prompt, KERAS_BACKEND, KAGGLE_USERNAME, KAGGLE_KEY, MODEL_VERSION
from model import evi_loss

"""
Define hyperparameters
"""

os.environ["KERAS_BACKEND"] = KERAS_BACKEND
os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"] = KAGGLE_KEY

tokenizer = GemmaTokenizer.from_preset(MODEL_VERSION)

adam = optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-07, clipnorm=1e-1
)
adamsmall = optimizers.Adam(
    learning_rate=0.0002, beta_1=0.9, beta_2=0.99, epsilon=1e-07, clipnorm=1e-1
)
adamtiny = optimizers.Adam(
    learning_rate=0.00005, beta_1=0.9, beta_2=0.99, epsilon=1e-07, clipnorm=1e-1
)
adamnano = optimizers.Adam(
    learning_rate=0.00002, beta_1=0.9, beta_2=0.99, epsilon=1e-07, clipnorm=1e-1
)


video_dim = 1280
text_dim = 2048
Tvid = 10
Tcap = 32 - 1
n_caption = 20
nblock = 2
active_select = True
augmentation = 5
Nsample = 10
perceiver = 16
baselines = ["evi", "jll", "entr", "vcluster", "seq"]
baseline = baselines[0]
testmodes = ["max", "avg", "low_unc"]
testmode = testmodes[0]
testtopk = 2
kldiv = 3200
Ntrain = 1000
Nactive = 1000
active_portion = 1
t_start = 6400
t_end = 7010
epoch = 20
max_round = 4
dropout = 0.1
smooth = 0.1
sparselabel = False
n_vcluster = 5
max_vcluster = 200
pca_dim = 8


"""
Data preparation
"""

directory_features = "features_pkl"
folder_artifact = "testkw"
isExist = os.path.exists(folder_artifact)
if not isExist:
    os.makedirs(folder_artifact)
print(
    f"baseline:{baseline}, active_select:{active_select}, \
      augmentation:{augmentation}, folder:{folder_artifact}"
)

with open("datas4kw.pkl", "rb") as f:
    datas, index_lookup, idx_kw = pickle.load(f)
    vocsize = len(index_lookup)
    kwsize = len(idx_kw)
print(f"vocabulary: {vocsize}, kwsize: {kwsize}")


"""
Preprocess video features
"""
if baseline == "vcluster":
    pca_dict = {}

    from sklearn.decomposition import PCA

    pcainput = np.zeros((len(datas), video_dim), dtype="float32")
    paths = list(datas.keys())
    for j in tqdm(range(len(paths))):
        path = paths[j]
        masknow = datas[path]["mask"]
        with open(directory_features + "/" + path, "rb") as f:
            feature = pickle.load(f)
        pcainput[j, :] = np.mean(feature[: int(np.sum(masknow)), :], axis=0)

    pca = PCA(n_components=pca_dim)
    pca_output = pca.fit_transform(pcainput)
    for j in range(len(paths)):
        path = paths[j]
        pca_dict[path] = pca_output[j]
    with open("pca.pkl", "wb") as f:
        pickle.dump(pca_dict, f)

    from sklearn.cluster import KMeans

    with open("pca.pkl", "rb") as f:
        pca_dict = pickle.load(f)


shuffled = list(datas.keys())
random.Random(0).shuffle(shuffled)
shuffled_full = [x for x in shuffled]
shuffled = shuffled[:Ntrain]


"""
Prepare generator
"""


def get_test_kw(path_batch, datas, kwsize):
    features = []
    videomask = []
    cap = []
    for path in path_batch:
        with open(directory_features + "/" + path, "rb") as f:
            featureraw = pickle.load(f)
        data = datas[path]
        Traw = featureraw.shape[0]
        feature = np.zeros((Tvid, video_dim), dtype="float32")
        feature[: min(Traw, Tvid), :] = featureraw[: min(Traw, Tvid), :]
        mask = data["mask"][:Tvid]
        features.append(np.repeat(feature[np.newaxis, :, :], 1, axis=0))
        videomask.append(
            np.repeat(
                mask[
                    np.newaxis,
                    :,
                ],
                1,
                axis=0,
            )
        )
        binarr = np.zeros((kwsize), dtype="float32")
        for kws in data["kw"]:
            for kw in kws:
                binarr[kw] = 1
        cap.append(binarr)
    features = np.concatenate(features, axis=0)
    videomask = np.concatenate(videomask, axis=0)
    cap = np.stack(cap, axis=0)
    return features, videomask, cap


def get_train_kw(paths, datas, kwsize, nstep, nbatch=2):
    while 1:
        for step in range(nstep):
            features, videomask, cap = get_test_kw(
                paths[step * nbatch : (step + 1) * nbatch], datas, kwsize=kwsize
            )
            zeroscap = np.zeros((cap.shape[0], cap.shape[1], 1), dtype="float32")
            yield ([features, videomask, cap], {"nll1": zeroscap, "kl1": zeroscap})


def get_test_single(path_batch, datas, augmentation=0):
    features = []
    videomask = []
    cap = []
    for path in path_batch:
        with open(directory_features + "/" + path, "rb") as f:
            featureraw = pickle.load(f)
        data = datas[path]
        n_caption = len(data["token"])
        Traw = featureraw.shape[0]
        feature = np.zeros((Tvid, video_dim), dtype="float32")
        feature[: min(Traw, Tvid), :] = featureraw[: min(Traw, Tvid), :]
        mask = data["mask"][:Tvid]
        features.append(np.repeat(feature[np.newaxis, :, :], n_caption, axis=0))
        videomask.append(
            np.repeat(
                mask[
                    np.newaxis,
                    :,
                ],
                n_caption,
                axis=0,
            )
        )
        cap.append(np.stack(data["token"], axis=0))
        if augmentation > 0:
            for itr in range(1, augmentation):
                lb = random.randint(0, min(Traw // 2 - 2, 1))
                ub = random.randint(min(Traw // 2 + 2, Traw - 1), Traw)
                if lb + 1 >= ub:
                    print("need to debug lb={lb} and ub={ub} Traw={Traw}")
                    lb = 0
                    ub = Traw
                feature0 = np.zeros_like(feature)
                mask0 = np.zeros_like(mask)
                feature_crop = featureraw[lb:ub, :]
                mask_crop = data["mask"][lb:ub]
                Tnow = ub - lb
                feature0[: min(Tvid, Tnow), :] = feature_crop[: min(Tvid, Tnow), :]
                mask0[: min(Tvid, Tnow)] = mask_crop[: min(Tvid, Tnow)]
                features.append(
                    np.repeat(feature0[np.newaxis, :, :], n_caption, axis=0)
                )
                videomask.append(
                    np.repeat(
                        mask0[
                            np.newaxis,
                            :,
                        ],
                        n_caption,
                        axis=0,
                    )
                )
                cap.append(np.stack(data["token"], axis=0))
    features = np.concatenate(features, axis=0)
    videomask = np.concatenate(videomask, axis=0)
    cap = np.concatenate(cap, axis=0)
    token1 = cap[:, :-1]
    token2 = cap[:, 1:]
    sampleweight = token2 > 0
    return features, videomask, token1, token2, sampleweight


def get_train_list(paths, datas, nstep, nbatch=2, smooth=None, augmentation=0):
    while 1:
        for step in range(nstep):
            features, videomask, token1, token2, sampleweight = get_test_single(
                paths[step * nbatch : (step + 1) * nbatch],
                datas,
                augmentation=augmentation,
            )
            if smooth is not None:
                token2 = tf.keras.utils.to_categorical(
                    token2, num_classes=vocsize, dtype="float32"
                )
                token2 = 1 / vocsize + (1 - smooth) * token2
            yield [features, videomask, token1], {"lnow": token2}, {
                "lnow": sampleweight
            }


def get_test_contrast(path_batch, datas, augmentation=0, kw=True):
    features = []
    videomask = []
    cap = []
    kwarrs = []
    npath = len(path_batch)
    for path in path_batch:
        with open(directory_features + "/" + path, "rb") as f:
            featureraw = pickle.load(f)
        data = datas[path]
        n_caption = len(data["token"])
        Traw = featureraw.shape[0]
        feature = np.zeros((Tvid, video_dim), dtype="float32")
        feature[: min(Traw, Tvid), :] = featureraw[: min(Traw, Tvid), :]
        mask = data["mask"][:Tvid]
        features.append(np.repeat(feature[np.newaxis, :, :], n_caption, axis=0))
        videomask.append(
            np.repeat(
                mask[
                    np.newaxis,
                    :,
                ],
                n_caption,
                axis=0,
            )
        )
        cap.append(np.stack(data["token"], axis=0))
        if augmentation > 0:
            for itr in range(1, augmentation):
                lb = random.randint(0, min(Traw // 2 - 2, 1))
                ub = random.randint(min(Traw // 2 + 2, Traw - 1), Traw)
                if lb + 1 >= ub:
                    print("need to debug lb={lb} and ub={ub} Traw={Traw}")
                    lb = 0
                    ub = Traw
                feature0 = np.zeros_like(feature)
                mask0 = np.zeros_like(mask)
                feature_crop = featureraw[lb:ub, :]
                mask_crop = data["mask"][lb:ub]
                Tnow = ub - lb
                feature0[: min(Tvid, Tnow), :] = feature_crop[: min(Tvid, Tnow), :]
                mask0[: min(Tvid, Tnow)] = mask_crop[: min(Tvid, Tnow)]
                features.append(
                    np.repeat(feature0[np.newaxis, :, :], n_caption, axis=0)
                )
                videomask.append(
                    np.repeat(
                        mask0[
                            np.newaxis,
                            :,
                        ],
                        n_caption,
                        axis=0,
                    )
                )
                cap.append(np.stack(data["token"], axis=0))
        if kw:
            kwarr = np.zeros((n_caption, 1), dtype="int32")
            for i in range(n_caption):
                kwnow = data["kw"][i]
                kwarr[i, 0] = 0 if len(kwnow) == 0 else sample(kwnow, 1)[0]
            kwarrs.append(kwarr)
            if augmentation > 0:
                for itr in range(1, augmentation):
                    kwarrs.append(kwarr)
    features = np.concatenate(features, axis=0)
    videomask = np.concatenate(videomask, axis=0)
    cap = np.concatenate(cap, axis=0)
    token1 = cap[:, :-1]
    token2 = cap[:, 1:]
    sampleweight = token2 > 0
    if augmentation > 0:
        n_caption = n_caption * augmentation
    gt = np.zeros((npath * n_caption, npath * n_caption), dtype="float32")
    for i in range(npath):
        gt[i * n_caption : (i + 1) * n_caption, i * n_caption : (i + 1) * n_caption] = 1
    if not kw:
        return features, videomask, token1, token2, gt, sampleweight
    else:
        kwarrs = np.concatenate(kwarrs, axis=0)
        return features, videomask, token1, token2, gt, sampleweight, kwarrs


def get_train_list_contrast(paths, datas, nstep, nbatch=2, augmentation=0):
    while 1:
        for step in range(nstep):
            features, videomask, token1, token2, gt, sampleweight = get_test_contrast(
                paths[step * nbatch : (step + 1) * nbatch],
                datas,
                augmentation=augmentation,
            )
            zeros = np.zeros((gt.shape[0], 1), dtype="float32")
            ones = np.ones((gt.shape[0],), dtype="float32")
            yield (
                [features, videomask, token1, token2, gt],
                {"lnow": token2, "nll": zeros, "kl": zeros},
                {"lnow": sampleweight, "nll": ones, "kl": ones},
            )


def get_train_list_contraevi(paths, datas, nstep, nbatch=2, augmentation=0):
    while 1:
        for step in range(nstep):
            features, videomask, token1, token2, gt, sampleweight = get_test_contrast(
                paths[step * nbatch : (step + 1) * nbatch],
                datas,
                augmentation=augmentation,
            )
            zeros = np.zeros((gt.shape[0], gt.shape[1], 1), dtype="float32")
            ones = np.ones((gt.shape[0], gt.shape[1], 1), dtype="float32")
            sampleweight = sampleweight[:, :, np.newaxis]
            zeroscap = np.zeros_like(sampleweight, dtype="float32")
            yield (
                [features, videomask, token1, token2, gt],
                {"nll1": zeroscap, "kl1": zeroscap, "nll2": zeros, "kl2": zeros},
                {"nll1": sampleweight, "kl1": sampleweight, "nll2": ones, "kl2": ones},
            )


def get_train_contraevikw(paths, datas, nstep, nbatch=2, augmentation=0):
    while 1:
        for step in range(nstep):
            (
                features,
                videomask,
                token1,
                token2,
                gt,
                sampleweight,
                kwarrs,
            ) = get_test_contrast(
                paths[step * nbatch : (step + 1) * nbatch],
                datas,
                augmentation=augmentation,
                kw=True,
            )
            zeros = np.zeros((gt.shape[0], gt.shape[1], 1), dtype="float32")
            ones = np.ones((gt.shape[0], gt.shape[1], 1), dtype="float32")
            sampleweight = sampleweight[:, :, np.newaxis]
            zeroscap = np.zeros_like(sampleweight, dtype="float32")
            yield (
                [features, videomask, token1, token2, gt, kwarrs],
                {"nll1": zeroscap, "kl1": zeroscap, "nll2": zeros, "kl2": zeros},
                {"nll1": sampleweight, "kl1": sampleweight, "nll2": ones, "kl2": ones},
            )



if baseline == "evi":
    features, vmask, token1, token2, gt, sampleweight, kw = get_test_contrast(
        shuffled[0:2], datas, augmentation=augmentation
    )
    features2, vmask2, cap = get_test_kw(shuffled[0:2], datas, kwsize=kwsize)
elif baseline in ["jll", "entr", "vcluster", "seq"]:
    features, vmask, token1, token2, sampleweight = get_test_single(
        shuffled[0:2], datas
    )

from utilInterpret import translate_kw, get_prompt_id
from utilInterpret import translate_wordpiece as translate

for i in range(min(10, token1.shape[0])):
    translate(index_lookup, token1[i : i + 1], f"token1:{i}", fprint=True)
    translate_kw(idx_kw, kw[i, 0], f"kw:{i}", fprint=True)


"""
Building the Transformer-based model
"""

if augmentation > 0:
    nbatch_video = 5
    nbatch = nbatch_video * n_caption * augmentation
else:
    nbatch_video = 5
    nbatch = nbatch_video * n_caption

if baseline in ["jll", "entr", "vcluster"]:
    from model import ViTcaptionSoftmax as get_compiled_model

    model = get_compiled_model(
        sequence_length=Tvid,
        cap_length=Tcap,
        embed_dim=video_dim,
        text_dim=text_dim,
        vocsize=len(index_lookup),
        nblock=nblock,
        dropout=dropout,
    )
    model.compile(
        optimizer=adamtiny,
        loss={
            "lnow": "sparse_categorical_crossentropy",
        },
        loss_weights={
            "lnow": 1,
        },
        weighted_metrics={
            "lnow": "accuracy",
        },
    )
elif baseline == "seq":
    from model import modelSeq as get_compiled_model

    model = get_compiled_model(
        sequence_length=Tvid,
        cap_length=Tcap,
        embed_dim=video_dim,
        text_dim=text_dim,
        vocsize=len(index_lookup),
    )
    model.compile(
        optimizer=adamsmall,
        loss={
            "lnow": "sparse_categorical_crossentropy",
        },
        loss_weights={
            "lnow": 1,
        },
        metrics={
            "lnow": "accuracy",
        },
    )
elif baseline == "evi":
    from model import ViTkw
    model = load_model("model0.keras")
    model.compile(
        optimizer=adamtiny,
        loss={"lnow":"evi_loss",},
        metrics={"lnow":evi_loss}
    )
    
    mkw = ViTkw(
        sequence_length=Tvid, video_dim=video_dim, text_dim=text_dim, kwsize=kwsize
    )
    mkw.compile(
        optimizer=adamtiny,
        loss={
            "nll1": "mae",
            "kl1": "mae",
        },
        loss_weights={
            "nll1": 1,
            "kl1": 1 / kldiv,
        },
        metrics={"nll1": "mae", "kl1": "mae"},
    )


from utilInterpret import (
    decodeEvi_topp,
    decodeCE_topp,
    decodeCE_beam,
    decode_kw,
    decodeEvi_gemma,
    get_prompt_id
)
from nltkeval import calculate_bleu, calculate_rouge, calculate_meteor

saving_epoch = [5 * i for i in range(20)]

data = {"video": [], "gt": []}


class CallbackTeacher(Callback):
    def __init__(
        self, logging2interval=5, logging2heatup=10, decay=0.5, prefix="", teacher=None
    ):
        super().__init__()
        self.logging2interval = logging2interval
        self.logging2heatup = logging2heatup
        self.teacher = teacher
        self.decay = decay
        self.prefix = prefix
        self.history = {
            "bleu": [],
            "rouge": [],
            "meteor": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        if self.teacher is not None:
            if epoch == self.logging2heatup:
                self.weightsma = self.model.get_weights()
                self.teacher.set_weights(self.weightsma)
            elif epoch > self.logging2heatup:
                weightsnew = self.model.get_weights()
                newma = []
                for i in range(len(self.weightsma)):
                    newlayer = self.weightsma[i] * self.decay + weightsnew[i] * (
                        1 - self.decay
                    )
                    newma.append(newlayer)
                self.weightsma = newma
                self.teacher.set_weights(self.weightsma)
        if (epoch + 1) in saving_epoch:
            self.teacher.save_weights(
                f"{folder_artifact}/teacher_{self.prefix}_{(epoch+1)}.h5"
            )

        model2 = self.teacher
        if epoch % self.logging2interval == 0 and epoch >= self.logging2heatup:
            fprint = False
            bleus = []
            rouges = []
            meteors = []
            vacs = []
            jllsall = []
            entsall = []

            gts = []
            preds = []
            data[str(epoch)] = []

            for idxtest in tqdm(range(t_start, min(t_end, t_start + 800))):
                features, vmask, token1, token2, gt, sampleweight = get_test_contrast(
                    shuffled_full[idxtest : idxtest + 1], datas
                )

                for i in range(1):
                    imgnow = features[i : i + 1]
                    vmasknow = vmask[i : i + 1]
                    nsample = 1

                    token1s = np.zeros((nsample, Tcap))
                    gts = np.zeros((nsample, nsample))

                    token2s = token2[i : i + n_caption]

                    if baseline == "evi":
                        predcaps, predvac, lnow = decodeEvi_topp(
                            model2,
                            imgnow,
                            vmasknow,
                            token1s,
                            gts,
                            Tcap,
                            vocsize=len(index_lookup),
                            topp=0.5,
                            topk=2,
                        )
                        vacs.append(np.mean(predvac))
                        if testmode == "low_unc":
                            jlls = -predvac
                    elif baseline in ["jll", "entr", "vcluster", "seq"]:
                        predcaps, jlls, ents = decodeCE_beam(
                            model2, imgnow, vmasknow, token1s, Tcap, B=4, numOutput=4
                        )
                        jllsall.append(np.mean(jlls))
                        entsall.append(np.mean(ents))

                    for j in range(n_caption):
                        token2now = token2s[j : j + 1]
                        gts.append(
                            translate(index_lookup, token2now, "targe", fprint=fprint)
                        )
                        if len(data["video"]) < ((t_end - t_start) * n_caption + 4):
                            data["video"].append(idxtest)
                            data["label"].append("gt")
                        data[str(epoch)].append(gts[-1])
                    for j in range(min(nsample, predcaps.shape[0])):
                        preds.append(
                            translate(
                                index_lookup,
                                predcaps[j : j + 1],
                                "predi",
                                fprint=fprint,
                            )
                        )
                        if len(data["video"]) < ((t_end - t_start) * n_caption + 4):
                            data["video"].append(idxtest)
                            data["label"].append("pred")
                        data[str(epoch)].append(preds[-1])

                    bleunow = []
                    rougenow = []
                    meteornow = []
                    for j in range(min(nsample, len(preds))):
                        if len(preds[j]) > 0:
                            bleunow.append(calculate_bleu(preds[j], gts)[0])
                            rougenow.append(calculate_rouge(preds[j], gts))
                            meteornow.append(calculate_meteor(preds[j], gts))
                    if testmode == "max":
                        bleus.append(np.max(bleunow))
                        rouges.append(np.max(rougenow))
                        meteors.append(np.max(meteornow))
                    elif testmode == "avg":
                        bleus.append(np.mean(bleunow))
                        rouges.append(np.mean(rougenow))
                        meteors.append(np.mean(meteornow))
                    elif testmode == "low_unc":
                        jll_candidate = np.mean(jlls, axis=-1)
                        score = -np.array(jll_candidate)
                        unc_sort = np.argsort(score)
                        bleus.append(
                            np.mean([bleunow[unc_sort[k]] for k in range(testtopk)])
                        )
                        rouges.append(
                            np.mean([rougenow[unc_sort[k]] for k in range(testtopk)])
                        )
                        meteors.append(
                            np.mean([meteornow[unc_sort[k]] for k in range(testtopk)])
                        )

            bleu = np.mean(np.array(bleus))
            rouge = np.mean(np.array(rouges))
            meteor = np.mean(np.array(meteors))
            print([bleu, rouge, meteor])
            self.history["bleu"].append(bleu)
            self.history["rouge"].append(rouge)
            self.history["meteor"].append(meteor)


if baseline in ["jll", "entr", "vcluster"]:
    teacher = get_compiled_model(
        sequence_length=Tvid,
        cap_length=Tcap,
        embed_dim=video_dim,
        text_dim=text_dim,
        vocsize=len(index_lookup),
        nblock=nblock,
        dropout=dropout,
    )
    num_step = len(shuffled) // nbatch_video
    generator = get_train_list(
        shuffled,
        datas,
        nstep=num_step,
        nbatch=nbatch_video,
        smooth=smooth,
        augmentation=augmentation,
    )

    Nstep4 = (t_end - t_start) // nbatch_video
    print(Nstep4)
    mygenerator_val = get_train_list(
        shuffled_full[Ntrain:], datas, nstep=Nstep4, nbatch=nbatch_video, smooth=smooth
    )
elif baseline in ["seq"]:
    teacher = get_compiled_model(
        sequence_length=Tvid,
        cap_length=Tcap,
        embed_dim=video_dim,
        text_dim=text_dim,
        vocsize=len(index_lookup),
    )
    num_step = len(shuffled) // nbatch_video
    generator = get_train_list(
        shuffled,
        datas,
        nstep=num_step,
        nbatch=nbatch_video,
        smooth=smooth,
        augmentation=augmentation,
    )
elif baseline == "evi":
    num_step = len(shuffled) // nbatch_video
    generator = get_train_contraevikw(
        shuffled, datas, nstep=num_step, nbatch=nbatch_video, augmentation=augmentation
    )
    teacher = get_compiled_model(
        sequence_length=Tvid,
        cap_length=Tcap,
        video_dim=video_dim,
        text_dim=text_dim,
        vocsize=len(index_lookup),
        kwsize=kwsize,
        nblock=nblock,
        nbatch=nbatch,
        dropout=dropout,
        sparselabel=sparselabel,
        perceiver=perceiver,
    )
    Nstep4 = (t_end - t_start) // nbatch_video
    print(Nstep4)
    mygenerator_val = get_train_contraevikw(
        shuffled, datas, nstep=num_step, nbatch=nbatch_video
    )

    generator_kw = get_train_kw(
        shuffled, datas, kwsize=kwsize, nstep=num_step, nbatch=nbatch_video
    )


logging2interval = 5
logging2heatup = 10
callback1 = CallbackTeacher(
    teacher=teacher, logging2interval=logging2interval, logging2heatup=logging2heatup
)
callback1.prefix = "active0"


"""
Model training process
"""

mode = 1
if mode == 0:
    model.summary()
    model.save_weights(f"{folder_artifact}/modelinit.h5")
    mkw.summary()
    model.save_weights(f"{folder_artifact}/mkwinit.h5")
    history = mkw.fit(
        generator_kw,
        steps_per_epoch=num_step,
        epochs=epoch,
    )
    mkw.save_weights(f"{folder_artifact}/kw0.h5")

    print(f"shuffled: {len(shuffled)}")
    history = model.fit(
        generator,
        steps_per_epoch=num_step,
        epochs=epoch,
        validation_data=mygenerator_val,
        validation_steps=Nstep4,
    )
    model.save_weights(f"{folder_artifact}/active0.h5")


elif mode == 1:
    model.load_weights(f"{folder_artifact}/active0.h5")
    mkw.load_weights(f"{folder_artifact}/kw0.h5")


"""
Interactive learning and inference
"""

nsample = Nsample

if baseline in ["jll", "entr", "vcluster"]:
    model2 = get_compiled_model(
        sequence_length=Tvid,
        cap_length=Tcap,
        embed_dim=video_dim,
        text_dim=text_dim,
        vocsize=len(index_lookup),
        nblock=nblock,
        dropout=dropout,
    )
elif baseline == "seq":
    model2 = get_compiled_model(
        sequence_length=Tvid,
        cap_length=Tcap,
        embed_dim=video_dim,
        text_dim=text_dim,
        vocsize=len(index_lookup),
    )
elif baseline == "evi":
    model2 = get_compiled_model(
        sequence_length=Tvid,
        cap_length=Tcap,
        video_dim=video_dim,
        text_dim=text_dim,
        vocsize=len(index_lookup),
        kwsize=kwsize,
        nbatch=Nsample,
        nblock=nblock,
        dropout=dropout,
        sparselabel=sparselabel,
        perceiver=perceiver,
    )


checkselected = []

for idxactive in range(1, max_round + 1):
    model2.load_weights(f"{folder_artifact}/active{idxactive-1}.h5")
    mkw.load_weights(f"{folder_artifact}/kw{idxactive-1}.h5")

    fprint = True
    bleus = []
    rouges = []
    meteors = []
    vacs = []
    jllsall = []
    entsall = []
    if f"pred_{idxactive}" not in data:
        data[f"pred_{idxactive}"] = []

    for idxtest in tqdm(range(t_start, t_end)):
        features, vmask, token1, token2, gt, sampleweight, kw = get_test_contrast(
            shuffled_full[idxtest : idxtest + 1], datas
        )
        features2, vmask2, kw2 = get_test_kw(
            shuffled_full[idxtest : idxtest + 1], datas, kwsize=kwsize
        )
        inter = decode_kw(mkw, features2, vmask2, kw2, nsample=n_caption)

        for i in range(1):
            imgnow = features[i : i + 1]
            vmasknow = vmask[i : i + 1]
            imgnow = np.repeat(imgnow, nsample, axis=0)
            vmasknow = np.repeat(vmasknow, nsample, axis=0)
            kwnow = kw[:nsample]

            token1s = np.zeros((nsample, Tcap))
            gts = np.zeros((nsample, nsample))

            token2s = token2[i : i + n_caption]
            prompt_id = get_prompt_id(tokenizer, prompt, kwnow)
            if baseline == "evi":
                predcaps, predvac, lnow = decodeEvi_gemma(model2, imgnow, vmasknow, token1s, gts,  Tcap, vocsize,
                                                          prompt_id=prompt_id)
                vacs.append(np.mean(predvac))
                check_evidence = np.sum(lnow, axis=-1, keepdims=True)
                if testmode == "low_unc":
                    jlls = -predvac
            elif baseline in ["jll", "entr", "vcluster", "seq"]:
                predcaps, jlls, ents = decodeCE_beam(
                    model2, imgnow, vmasknow, token1s, Tcap, B=4, numOutput=4
                )
                jllsall.append(np.mean(jlls))
                entsall.append(np.mean(ents))

            gts = []
            preds = []
            for j in range(n_caption):
                token2now = token2s[j : j + 1]
                gts.append(translate(index_lookup, token2now, "targe", fprint=fprint))
                if len(data["video"]) < ((6405 - t_start) * n_caption):
                    data["video"].append(idxtest)
                    data["gt"].append(gts[-1])

            for j in range(min(nsample, predcaps.shape[0])):
                preds.append(
                    translate(index_lookup, predcaps[j : j + 1], "predi", fprint=fprint)
                )
                data[f"pred_{idxactive}"].append(preds[-1])
            for k in range(n_caption - nsample):
                data[f"pred_{idxactive}"].append("")

            bleunow = []
            rougenow = []
            meteornow = []
            for j in range(min(nsample, len(preds))):
                bleunow.append(calculate_bleu(preds[j], gts)[0])
                rougenow.append(calculate_rouge(preds[j], gts))
                meteornow.append(calculate_meteor(preds[j], gts))
                row_pd = (idxtest - 0) * n_caption + j
                """
                data[f"pred_{idxactive}"][row_pd] = str("%.2f" % round(jlls[j], 2)) +" "\
                    + str("%.2f" % round(bleunow[-1], 2)) +" "+ str("%.2f" % round(rougenow[-1], 2)) \
                    +" "+ str("%.2f" % round(meteornow[-1], 2)) +" "+ data[f"pred_{idxactive}"][row_pd]
                """
            if testmode == "first":
                bleus.append((bleunow[0]))
                rouges.append((rougenow[0]))
                meteors.append((meteornow[0]))
            elif testmode == "max":
                bleus.append(np.max(bleunow))
                rouges.append(np.max(rougenow))
                meteors.append(np.max(meteornow))
            elif testmode == "avg":
                bleus.append(np.mean(bleunow))
                rouges.append(np.mean(rougenow))
                meteors.append(np.mean(meteornow))
            elif testmode == "low_unc":
                jll_candidate = np.mean(jlls, axis=-1)
                score = -np.array(jll_candidate)
                unc_sort = np.argsort(score)
                bleus.append(np.mean([bleunow[unc_sort[k]] for k in range(testtopk)]))
                rouges.append(np.mean([rougenow[unc_sort[k]] for k in range(testtopk)]))
                meteors.append(
                    np.mean([meteornow[unc_sort[k]] for k in range(testtopk)])
                )
    if len(vacs) == 0:
        vacs = [0]
    if len(jllsall) == 0:
        jllsall = [0]
    if len(entsall) == 0:
        entsall = [0]

    print(
        {
            "testbleu": np.mean(bleus),
            "testrouge": np.mean(rouges),
            "testmeteor": np.mean(meteors),
            "testvac": np.mean(vacs),
            "testjll": np.mean(jllsall),
            "testent": np.mean(entsall),
        }
    )

    if idxactive == max_round:
        break

    if not active_select:
        shuffled.extend(
            shuffled_full[
                Ntrain
                + Nactive * (idxactive - 1) : Ntrain
                + Nactive * (idxactive - 1)
                + int(Nactive * active_portion)
            ]
        )
    else:
        active_candidate = []
        for j in range(Ntrain, t_start):
            if j not in checkselected:
                active_candidate.append(j)
        print(f"total active_candidate: {active_candidate}")

        nsample = Nsample

        model2.load_weights(f"{folder_artifact}/active{idxactive-1}.h5")

        fprint = False
        bleus = []
        rouges = []
        meteors = []
        vacs = []
        jllsall = []
        entsall = []

        print(f"select activelearning {idxactive}")
        for idxtest in tqdm(active_candidate):
            features, vmask, token1, token2, gt, sampleweight, kw = get_test_contrast(
                shuffled_full[idxtest : idxtest + 1], datas
            )
            features2, vmask2, kw2 = get_test_kw(
                shuffled_full[idxtest : idxtest + 1], datas, kwsize=kwsize
            )
            inter = decode_kw(mkw, features2, vmask2, kw2, nsample=n_caption)

            for i in range(1):
                imgnow = features[i : i + 1]
                vmasknow = vmask[i : i + 1]
                imgnow = np.repeat(imgnow, nsample, axis=0)
                vmasknow = np.repeat(vmasknow, nsample, axis=0)
                kwnow = kw[:nsample]

                token1s = np.zeros((nsample, Tcap))
                gts = np.zeros((nsample, nsample))

                token2s = token2[i : i + n_caption]

                if baseline == "evi":
                    predcaps, predvac, lnow = decodeEvi_topp(
                        model2,
                        imgnow,
                        vmasknow,
                        token1s,
                        gts,
                        Tcap,
                        vocsize=len(index_lookup),
                        topp=0.8,
                        topk=5,
                        inter=inter,
                        kw=kwnow,
                    )
                    vacs.append(np.mean(predvac))
                    check_evidence = np.sum(lnow, axis=-1, keepdims=True)
                elif baseline in ["jll", "entr", "vcluster", "seq"]:
                    predcaps, jlls, ents = decodeCE_topp(
                        model2,
                        imgnow,
                        vmasknow,
                        token1s,
                        Tcap,
                        topp=0.8,
                        topk=5,
                        perceiver=perceiver,
                    )
                    jllsall.append(np.mean(jlls))
                    entsall.append(np.mean(ents))

                gts = []
                preds = []
                for j in range(n_caption):
                    token2now = token2s[j : j + 1]
                    gts.append(
                        translate(index_lookup, token2now, "targe", fprint=fprint)
                    )
                for j in range(nsample):
                    preds.append(
                        translate(
                            index_lookup, predcaps[j : j + 1], "predi", fprint=fprint
                        )
                    )

        if len(vacs) == 0:
            vacs = [0]
        if len(jllsall) == 0:
            jllsall = [0]
        if len(entsall) == 0:
            entsall = [0]

        print(
            {
                "vac": np.mean(vacs),
                "jll": np.mean(jllsall),
                "ent": np.mean(entsall),
            }
        )

        if baseline in ["jll", "entr", "evi", "seq"]:
            if baseline == "evi":
                score = vacs
            elif baseline == "jll":
                score = -np.array(jllsall)
            elif baseline == "entr" or baseline == "seq":
                score = np.array(entsall)
            unc_sort = np.flip(np.argsort(score))
            for j in range(int(Nactive * active_portion)):
                idx_true = active_candidate[unc_sort[j]]
                shuffled.append(shuffled_full[idx_true])
                checkselected.append(idx_true)

        elif baseline == "vcluster":
            score = -np.array(jllsall)
            unc_sort = np.flip(np.argsort(score))

            kmeans_input = np.zeros((Nactive, pca_dim))
            for j in range(Nactive):
                path = shuffled_full[j + Ntrain + Nactive * idxactive]
                kmeans_input[j] = pca_dict[path]

            kmeans = KMeans(n_clusters=n_vcluster, random_state=0).fit(kmeans_input)
            kmeans_output = kmeans.labels_
            cluster_counts = np.zeros((n_vcluster), dtype="int32")
            total_selected = 0
            for j in range(int(Nactive * active_portion)):
                idx_true = active_candidate[unc_sort[j]]
                path = shuffled_full[idx_true]
                idx_cluster = kmeans_output[j]
                cluster_counts[idx_cluster] = cluster_counts[idx_cluster] + 1
                if cluster_counts[idx_cluster] <= max_vcluster:
                    shuffled.append(shuffled_full[idx_true])
                    checkselected.append(idx_true)
                    total_selected += 1

                if total_selected >= int(Nactive * active_portion):
                    break

        print(np.max(checkselected))

    if baseline in ["jll", "entr", "vcluster", "seq"]:
        num_step = len(shuffled) // nbatch_video
        generator = get_train_list(
            shuffled,
            datas,
            nstep=num_step,
            nbatch=nbatch_video,
            smooth=smooth,
            augmentation=augmentation,
        )
    elif baseline == "evi":
        num_step = len(shuffled) // nbatch_video
        generator = get_train_contraevikw(
            shuffled,
            datas,
            nstep=num_step,
            nbatch=nbatch_video,
            augmentation=augmentation,
        )
        generator_kw = get_train_kw(
            shuffled, datas, kwsize=kwsize, nstep=num_step, nbatch=nbatch_video
        )
    mode = 0
    if mode == 0:
        print(f"retrain shuffled: {len(shuffled)}")
        model.load_weights(f"{folder_artifact}/modelinit.h5")
        history = model.fit(
            generator,
            steps_per_epoch=num_step,
            epochs=epoch,
        )
        model.save_weights(f"{folder_artifact}/active{idxactive}.h5")

        mkw.load_weights(f"{folder_artifact}/mkwinit.h5")
        history = mkw.fit(
            generator_kw,
            steps_per_epoch=num_step,
            epochs=epoch,
        )
        mkw.save_weights(f"{folder_artifact}/kw{idxactive}.h5")
