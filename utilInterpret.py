import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from config import MODEL_VERSION, prompt
from keras_nlp.models import GemmaTokenizer

"""
Define utility functions
"""

def logging(history, m2=None, folder="folder", msg="msg", note=None, txtfile="log.txt"):
    isExist = os.path.exists(folder)
    if not isExist:
        os.makedirs(folder)
        print("The new directory is created!")

    now = datetime.now()
    numpy_loss_history = np.array(history.history["loss"])
    with open(folder + "/" + txtfile, "a") as file1:
        file1.write(f"{now}:\n")
        file1.write(f"    msg: {folder}/{msg}.h5\n")
        if note is not None:
            file1.write(f"    note: {note}\n")
        file1.write(f"    loss: {np.array2string(numpy_loss_history)}\n")

    print(history.history.keys())

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("history object")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig(f"{folder}/trainLoss {msg}.png")
    plt.show()
    plt.clf()


def logging2(
    history,
    logging2interval=2,
    folder="folder",
    msg="msg",
    note=None,
    txtfile="log2.txt",
):
    isExist = os.path.exists(folder)
    if not isExist:
        os.makedirs(folder)
        print("The new directory is created!")

    now = datetime.now()
    numpy_bleu_history = np.array(history["bleu"])
    numpy_rouge_history = np.array(history["rouge"])
    numpy_meteor_history = np.array(history["meteor"])
    with open(folder + "/" + txtfile, "a") as file1:
        file1.write(f"{now}:\n")
        file1.write(f"    msg: {folder}/{msg}.h5\n")
        if note is not None:
            file1.write(f"    note: {note}\n")
        file1.write(f"    train_loss: {np.array2string(numpy_bleu_history)}\n")
        file1.write(f"    train_loss: {np.array2string(numpy_rouge_history)}\n")
        file1.write(f"    train_loss: {np.array2string(numpy_meteor_history)}\n")

    plt.plot(history["bleu"])
    plt.plot(history["rouge"])
    plt.plot(history["meteor"])
    plt.title(f"validation nltk {msg}")
    plt.ylabel("nltk")
    plt.xlabel("epoch")
    plt.legend(["blue", "rouge", "meteor"], loc="upper left")
    xticks_default = range(len(history["bleu"]))
    xticks = [logging2interval * i for i in xticks_default]
    plt.xticks(xticks_default, xticks)
    plt.savefig(f"{folder}/nltk {msg}.png")
    plt.show()
    plt.clf()


tokenizer = GemmaTokenizer.from_preset(MODEL_VERSION)
prompt_id = tokenizer(prompt)


def translate(dictionary, prednum, printstr="pred", fprint=False):
    Tkey = prednum.shape[1]
    words = ""
    for t in range(Tkey):
        wordidx = prednum[0, t]
        if wordidx == 4:
            break
        words += str(dictionary[wordidx]) + " "
    if fprint:
        print(printstr + ": " + words)
    return words


def translate_kw(dictionary, prednum, printstr="pred", fprint=False):
    words = dictionary[prednum]
    if fprint:
        print(printstr + ": " + words)
    return words


def translate_wordpiece(dictionary, prednum, printstr="pred", fprint=False, idxeos=4):
    Tkey = prednum.shape[1]
    words = ""
    for t in range(Tkey):
        wordidx = prednum[0, t]
        if wordidx == idxeos:
            break
        if wordidx == 0:
            if t == Tkey - 1:
                print("Reached Tmax")
            else:
                print("predicting pad before eos")
                print(prednum)
            break
        wordnow = str(dictionary[wordidx])
        if wordnow[0] == "#":
            wordnow = wordnow.strip("#")
            words = words.rstrip() + wordnow + " "
        else:
            words += str(dictionary[wordidx]) + " "
    if fprint:
        print(printstr + ": " + words)
    return words


def decode_greedy(model, imgnow, vmasknow, token1now, Tcap):
    temp = np.zeros(token1now.shape, dtype="int32")
    wordidx = 3
    for t in range(Tcap - 1):
        temp[:, t] = wordidx
        if wordidx == 4:
            break
        preds = model.predict([imgnow, vmasknow, temp], verbose=0)

        wordidx = np.argmax(preds[0, t, :], axis=-1)
        if wordidx == 4 and t < 5:
            ordered = np.argsort(-preds[0][0, t, :])
            print(f"{temp[0]} {ordered[0]}")
            assert ordered[0] == 2, f"{temp[0]} {ordered[0]} should be 2"
            wordidx = ordered[1]
    predcap = temp[:, 1:]
    return predcap


def decode_unc(model, imgnow, vmasknow, token1now, Tcap):
    temp = np.zeros(token1now.shape, dtype="int32")
    wordidx = 3
    jlls = np.zeros(token1now.shape, dtype="float32")
    ents = np.zeros(token1now.shape, dtype="float32")
    for t in range(Tcap - 1):
        temp[:, t] = wordidx
        if wordidx == 4:
            break
        preds = model.predict([imgnow, vmasknow, temp], verbose=0)
        jlls[0, t + 1] = np.log(preds[0, t, wordidx] + 1e-8)
        ents[0, t + 1] = -np.sum(preds[0, t, :] * np.log(preds[0, t, :] + 1e-8))

        wordidx = np.argmax(preds[0, t, :], axis=-1)

        """
        if wordidx==4 and t<5:
            ordered = np.argsort(-preds[0,t,:])
            print(f"{temp[0]} {ordered[0]}")
            assert ordered[0]==2, f"{temp[0]} {ordered[0]} should be 2"
            wordidx = ordered[1]
        """
    predcap = temp[:, 1:]
    jlls = jlls[:, 1:]
    ents = ents[:, 1:]
    return predcap, jlls, ents


def decodeEvi_multisample(model, imgnow, vmasknow, token1s, gts, Tcap, vocsize):
    temp = np.zeros(token1s.shape, dtype="int32")
    nsample = token1s.shape[0]
    wordidx = 3 * np.ones((nsample))
    vacs = np.zeros(token1s.shape, dtype="float32")
    eos = np.zeros((nsample), dtype="bool")
    eospos = np.ones((nsample), dtype="int32") * (Tcap - 1 - 1)
    for t in range(Tcap):
        temp[:, t] = wordidx

        eosnow = wordidx == 4
        for i in range(nsample):
            if eosnow[i] and (not eos[i]):
                eospos[i] = t
        eos = np.logical_or(eos, eosnow)
        if np.sum(eos) == nsample:
            break

        preds = model.predict([imgnow, vmasknow, temp, token1s, gts], verbose=0)

        lnow = preds[-1][:, t, :]
        vacs[:, t] = vocsize / (np.sum(lnow, axis=-1) + vocsize)
        prob = (lnow + 1) / np.sum(lnow + 1, axis=-1, keepdims=True)
        cum_prob = np.cumsum(prob, axis=-1)

        r = np.random.uniform(size=(cum_prob.shape[0], 1))
        wordidx = np.argmax(cum_prob > r, axis=-1)

    for i in range(nsample):
        temp[i, eospos[i] + 1 :] = 0
        vacs[i, eospos[i] + 1 :] = 0
    predcap = temp[:, 1:]
    predvac = vacs[:, :-1]
    return predcap, predvac, lnow


def decodeEvi_topp(
    model,
    imgnow,
    vmasknow,
    token1s,
    gts,
    Tcap,
    vocsize,
    kw=None,
    inter=None,
    topp=0.9,
    topk=8,
):
    temp = np.zeros(token1s.shape, dtype="int32")
    nsample = token1s.shape[0]
    wordidx = 3 * np.ones((nsample))
    vacs = np.zeros(token1s.shape, dtype="float32")
    eos = np.zeros((nsample), dtype="bool")
    eospos = np.ones((nsample), dtype="int32") * (Tcap - 1 - 1)
    for t in range(Tcap):
        temp[:, t] = wordidx

        eosnow = wordidx == 4
        for i in range(nsample):
            if eosnow[i] and (not eos[i]):
                eospos[i] = t
        eos = np.logical_or(eos, eosnow)
        if np.sum(eos) == nsample:
            break
        if inter is None:
            preds = model.predict([imgnow, vmasknow, temp, token1s, gts], verbose=0)
        else:
            preds = model.predict([imgnow, vmasknow, temp, token1s, gts, kw], verbose=0)
        lnow = preds[-1][:, t, :]
        vacs[:, t] = vocsize / (np.sum(lnow, axis=-1) + vocsize)
        prob = (lnow + 1) / np.sum(lnow + 1, axis=-1, keepdims=True)
        prob2 = -np.sort(-prob, axis=-1)
        prob3 = np.cumsum(prob2, axis=-1)
        for row in range(nsample):
            idx_thresh = np.argmax(prob3[row] >= topp)
            thresh = prob2[row, min(idx_thresh, topk - 1)]
            truncated = prob[row]
            truncated[truncated < thresh] = 0
            prob[row] = truncated / np.sum(truncated, axis=-1, keepdims=True)
        cum_prob = np.cumsum(prob, axis=-1)

        r = np.random.uniform(size=(cum_prob.shape[0], 1))
        wordidx = np.argmax(cum_prob > r, axis=-1)

    for i in range(nsample):
        temp[i, eospos[i] + 1 :] = 0
        vacs[i, eospos[i] + 1 :] = 0
    predcap = temp[:, 1:]
    predvac = vacs[:, :-1]
    return predcap, predvac, lnow


def decodeCE_topp(model, imgnow, vmasknow, token1s, Tcap, topp=0.9, topk=8):
    nsample = token1s.shape[0]
    temp = np.zeros((nsample, Tcap), dtype="int32")
    wordidx = 3 * np.ones((nsample))
    jlls = np.zeros((nsample, Tcap), dtype="float32")
    ents = np.zeros((nsample, Tcap), dtype="float32")
    eos = np.zeros((nsample), dtype="bool")
    eospos = np.ones((nsample), dtype="int32") * (Tcap - 1 - 1)
    for t in range(Tcap):
        temp[:, t] = wordidx

        eosnow = wordidx == 4
        for i in range(nsample):
            if eosnow[i] and (not eos[i]):
                eospos[i] = t
        eos = np.logical_or(eos, eosnow)
        if np.sum(eos) == nsample:
            break

        preds = model.predict([imgnow, vmasknow, temp], verbose=0)
        lnow = preds[:, t, :]
        prob = lnow / np.sum(lnow, axis=-1, keepdims=True)
        prob2 = -np.sort(-prob, axis=-1)
        prob3 = np.cumsum(prob2, axis=-1)
        for row in range(nsample):
            idx_thresh = np.argmax(prob3[row] >= topp)
            thresh = prob2[row, min(idx_thresh, topk - 1)]
            truncated = prob[row]
            truncated[truncated < thresh] = 0
            prob[row] = truncated / np.sum(truncated, axis=-1, keepdims=True)
        cum_prob = np.cumsum(prob, axis=-1)

        r = np.random.uniform(size=(cum_prob.shape[0], 1))
        wordidx = np.argmax(cum_prob > r, axis=-1)

        for j in range(nsample):
            jlls[j, t] = np.log(preds[j, t, wordidx[j]] + 1e-8)
            ents[j, t] = -np.sum(lnow[j] * np.log(lnow[j] + 1e-8))

    for i in range(nsample):
        temp[i, eospos[i] + 1 :] = 0
    predcap = temp[:, 1:]
    jlls = jlls[:, :-1]
    ents = ents[:, :-1]
    return predcap, jlls, ents


def decode_kw(mkw, features2, vmask2, kw2, nsample=1):
    preds = mkw.predict([features2, vmask2, kw2], verbose=0)
    evi = preds[-1]
    prob_raw = (1 + evi[:, :, -1]) / (2 + np.sum(evi, axis=-1))
    prob_raw[prob_raw < 0.1] = 0
    outputs = np.zeros((nsample))
    for i in range(nsample):
        prob = prob_raw / np.sum(prob_raw + 1e-8)
        cum_prob = np.cumsum(prob, axis=-1)
        r = np.random.uniform(size=cum_prob.shape)
        kw = np.argmax(cum_prob > r, axis=-1)
        outputs[i] = kw
    return outputs


def decodeEvi_gemma(model, imgnow, vmasknow, token1s, gts,  Tcap, vocsize, kw=None, 
              inter=None, topp=0.9, topk=8, smooth=None, prompt_id=prompt_id, len_prompt=64, idxsos=3, idxeos=4):
    temp = np.zeros(token1s.shape, dtype='int32')
    nbatch = token1s.shape[0]
    if prompt_id is not None:
        prompt_id = np.tile(prompt_id, [nbatch,1])
        temp = np.concatenate((prompt_id, temp), axis=1)
    padding = np.ones((nbatch, len_prompt+Tcap), dtype="int32")
    nsample = token1s.shape[0]
    wordidx = idxsos* np.ones((nsample)) 
    vacs = np.zeros(token1s.shape, dtype='float32')
    eos = np.zeros((nsample), dtype="bool")
    eospos = np.ones((nsample), dtype="int32") * (Tcap-1-1) 
    if smooth is not None:
        token1s = np.repeat(np.expand_dims(temp, axis=-1), vocsize, axis=-1)
    
    for t in range(len_prompt, Tcap+len_prompt):
        temp[:,t] = wordidx
        
        eosnow = (wordidx==idxeos) 
        for i in range(nsample):
            if eosnow[i] and (not eos[i]):
                eospos[i] = t 
        eos = np.logical_or(eos, eosnow)
        if np.sum(eos)==nsample:
            break
        if inter is None:
            preds = model.predict({"visual_features":imgnow, "x_vmask":vmasknow, "token_ids":temp,
                   "padding_mask":padding}, verbose=0)
        else:
            preds = model.predict([imgnow,vmasknow,temp, token1s,gts, kw], verbose=0)
        lnow = preds[:,t,:] 
        vacs[:,t] = vocsize / (np.sum(lnow, axis=-1)+vocsize)
        prob = (lnow + 1) / np.sum(lnow+1, axis=-1, keepdims=True)
        prob2 = -np.sort(-prob, axis=-1)
        prob3 = np.cumsum(prob2, axis=-1)
        for row in range(nsample):
            idx_thresh = np.argmax(prob3[row]>=topp)
            thresh = prob2[row, min(idx_thresh, topk-1)]
            truncated = prob[row]
            truncated[truncated<thresh] = 0
            prob[row] = truncated / np.sum(truncated, axis=-1, keepdims=True)
        cum_prob = np.cumsum(prob, axis=-1)
        
        r = np.random.uniform(size=(cum_prob.shape[0],1))
        wordidx = np.argmax(cum_prob > r, axis=-1)
        
    for i in range(nsample):
        temp[i,eospos[i]+1:] = 0
        vacs[i,eospos[i]+1:] = 0
    predcap = temp[:,len_prompt+1:]
    predvac = vacs[:,:-1]
    return predcap, predvac,lnow


def get_prompt_id(tokenizer, prompt, kw):
    kw_token = tokenizer.detokenize(kw)
    if kw_token.shape[-1]>1:
        kw_token = kw_token[:,1]
    prompt_id = tokenizer(prompt).numpy()
    prompt_id = np.concatenate([prompt_id, kw_token], axis=1)
    return prompt_id

def decodeCE_beam(
    model, imgnow, vmasknow, token1s, Tcap, B=3, maxFinished=20, numOutput=5
):
    candis1 = []
    temp = np.zeros((1, Tcap), dtype="int32")
    temp[:, 0] = 3
    candis1.append([temp, 0, 0])
    imgnow0 = imgnow[:1]
    vmasknow0 = vmasknow[:1]

    finished = []
    for t in range(Tcap - 1):
        if len(finished) > maxFinished:
            break

        temp = np.concatenate([x[0] for x in candis1], axis=0)
        imgnow = np.repeat(imgnow0, temp.shape[0], axis=0)
        vmasknow = np.repeat(vmasknow0, temp.shape[0], axis=0)
        preds = model.predict([imgnow, vmasknow, temp], verbose=0)

        candis2 = []
        for j in range(preds.shape[0]):
            token1, jll, ent = candis1[j]
            predprobs = preds[j, t, :]
            ranking = np.argsort(-predprobs)
            entnew = ent - np.sum(predprobs * np.log(predprobs + 1e-8))
            for k in range(B):
                token1[0, t + 1] = ranking[k]
                jllnew = jll + np.log(predprobs[ranking[k]] + 1e-8)
                if ranking[k] == 4:
                    finished.append([token1.copy(), jllnew, entnew])
                else:
                    candis2.append([token1.copy(), jllnew, entnew])

        ordered = sorted(candis2, key=lambda x: x[1])
        candis1 = ordered[-B:]

    if len(finished) == 0:
        finished = candis1

    ordered = sorted(finished, key=lambda x: x[1])
    ordered = ordered[-numOutput:]
    ordered.reverse()
    predcap = np.concatenate([x[0] for x in ordered], axis=0)
    predcap = predcap[:, 1:]
    jlls = np.array([x[1] for x in ordered])
    ents = np.array([x[2] for x in ordered])

    return predcap, jlls, ents
