import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM, Lambda, Concatenate
from tensorflow.keras.layers import MultiHeadAttention, TimeDistributed, LayerNormalization
from tensorflow.keras.regularizers import L2
from keras_nlp.src.models.gemma.gemma_decoder_block import GemmaDecoderBlock
from keras_nlp.src.models.gemma.rms_normalization import RMSNormalization
from keras_nlp.src.layers.modeling.reversible_embedding import ReversibleEmbedding
from keras_nlp.src.backend import ops
import scipy
from config import dim_visual, size_perceiver, sequence_length, vocsize
import math

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        features = inputs[0]
        length = tf.shape(features)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return features + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = inputs[1] != 0
        return mask


class TransformerEncoder(layers.Layer):
    def __init__(self, dense_dim, key_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.key_dim = key_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation=tf.nn.gelu),
                layers.Dense(dense_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


"""
Customized embedding layer for aggregating token embeddings and positional embeddings
Embedding layer automatically generates mask and pass to following layers
"""


class SeqEmbedding(Layer):
    def __init__(self, T, V, embed_dim, **kwargs):
        super(SeqEmbedding, self).__init__(**kwargs)
        self.token_embeddings = Embedding(
            input_dim=V,
            output_dim=embed_dim,
            embeddings_regularizer=L2(1e-7),
        )
        self.position_embeddings = Embedding(
            input_dim=T,
            output_dim=embed_dim,
            embeddings_regularizer=L2(1e-7),
        )
        self.T = T
        self.V = V
        self.embed_dim = embed_dim

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.T, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        output = embedded_tokens + embedded_positions
        return output

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


"""
Customized decoder block
attention_output	The result of the computation, of shape (B, T, dim)
query: Query Tensor of shape (B, T, dim). T=Tseq
value: Value Tensor of shape (B, S, dim). S=Timg+Tkey
key: Optional key Tensor of shape (B, S, dim).
attention_mask: a boolean mask of shape (B, T, S), 
The boolean mask specifies which query elements can attend to which key elements
Broadcasting can happen for the missing batch dimensions and the head dimension.
"""


class Decoder(Layer):
    def __init__(self, embed_dim, key_dim, num_heads, selfattn=True, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention1 = MultiHeadAttention(
            num_heads,
            key_dim=key_dim,
            kernel_regularizer=L2(1e-7),
        )
        self.attention2 = MultiHeadAttention(
            num_heads,
            key_dim=key_dim,
            kernel_regularizer=L2(1e-7),
        )
        self.attention3 = MultiHeadAttention(
            num_heads,
            key_dim=key_dim,
            kernel_regularizer=L2(1e-7),
        )
        self.dense1 = Dense(
            embed_dim,
            activation="relu",
            kernel_regularizer=L2(1e-7),
        )
        self.dense2 = Dense(
            embed_dim,
            activation=None,
            kernel_regularizer=L2(1e-7),
        )
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.layernorm4 = LayerNormalization()
        self.supports_masking = True
        self.selfattn = selfattn

    def call(self, inputs, encoder_outputs, vmask):
        causal_mask = self.get_causal_mask(inputs)
        """
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
        combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
        combined_mask = tf.minimum(combined_mask, causal_mask)
        """
        vmask = tf.cast(vmask[:, tf.newaxis, :], dtype="int32")

        if self.selfattn:
            attn1 = self.attention1(
                query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
            )
            out1 = self.layernorm1(inputs + attn1)
        else:
            out1 = inputs

        attn2 = self.attention2(
            query=out1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=vmask,
        )
        out2 = self.layernorm2(out1 + attn2)

        ff_output = self.dense1(out2)
        ff_output = self.dense2(ff_output)
        output = self.layernorm3(out2 + ff_output)

        return output

    def get_causal_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, T = input_shape[0], input_shape[1]
        i = tf.range(T)[:, tf.newaxis]
        j = tf.range(T)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, T, T))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype="int32")],
            axis=0,
        )
        output = tf.tile(mask, mult)
        return output


class Decoderkw(Layer):
    def __init__(self, embed_dim, key_dim, num_heads, selfattn=True, **kwargs):
        super(Decoderkw, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention1 = MultiHeadAttention(
            num_heads,
            key_dim=key_dim,
            kernel_regularizer=L2(1e-7),
        )
        self.attention2 = MultiHeadAttention(
            num_heads,
            key_dim=key_dim,
            kernel_regularizer=L2(1e-7),
        )
        self.attention3 = MultiHeadAttention(
            num_heads,
            key_dim=key_dim,
            kernel_regularizer=L2(1e-7),
        )
        self.dense1 = Dense(
            embed_dim,
            activation="relu",
            kernel_regularizer=L2(1e-7),
        )
        self.dense2 = Dense(
            embed_dim,
            activation=None,
            kernel_regularizer=L2(1e-7),
        )
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.layernorm4 = LayerNormalization()
        self.supports_masking = True
        self.selfattn = selfattn

    def call(self, inputs, encoder_outputs, vmask, kw):
        causal_mask = self.get_causal_mask(inputs)
        vmask = tf.cast(vmask[:, tf.newaxis, :], dtype="int32")

        if self.selfattn:
            attn1 = self.attention1(
                query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
            )
            out1 = self.layernorm1(inputs + attn1)
        else:
            out1 = inputs

        attn3 = self.attention3(
            query=out1,
            value=kw,
            key=kw,
        )
        out3 = self.layernorm2(out1 + attn3)

        attn2 = self.attention2(
            query=out1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=vmask,
        )
        out2 = self.layernorm2(out3 + attn2)

        ff_output = self.dense1(out2)
        ff_output = self.dense2(ff_output)
        output = self.layernorm3(out2 + ff_output)

        return output

    def get_causal_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, T = input_shape[0], input_shape[1]
        i = tf.range(T)[:, tf.newaxis]
        j = tf.range(T)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, T, T))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype="int32")],
            axis=0,
        )
        output = tf.tile(mask, mult)
        return output


def ViTcaptionSoftmax(
    sequence_length, cap_length, embed_dim, text_dim, vocsize, nblock, dropout=None
):
    x_features = keras.Input(shape=(sequence_length, embed_dim), name="x_feature")
    x_vmask = keras.Input(shape=(sequence_length), name="x_vmask")
    x_token1 = keras.Input(shape=(cap_length), name="x_token")

    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )([x_features, x_vmask])

    x = TransformerEncoder(embed_dim, 64, 8, name="transformer_layer")(x, mask=x_vmask)

    seq = SeqEmbedding(cap_length, V=vocsize, embed_dim=text_dim, name="seq_embedding")(
        x_token1
    )
    for i in range(nblock):
        seq = Decoder(text_dim, 64, num_heads=8, name="decoderl" + str(i))(
            seq, x, x_vmask
        )
        if dropout is not None:
            seq = Dropout(dropout, name="dropa" + str(i))(seq)

    for i in range(1):
        fd = TimeDistributed(
            Dense(
                text_dim,
                activation="relu",
                kernel_regularizer=L2(1e-7),
            ),
            name="finalc" + str(i),
        )(seq)
    lnow = TimeDistributed(
        Dense(
            vocsize,
            activation="softmax",
            kernel_regularizer=L2(1e-7),
        ),
        name="lnow",
    )(fd)

    model = keras.Model([x_features, x_vmask, x_token1], lnow)

    return model


class ContrastiveEvidential(Layer):
    def __init__(self, dim=64, activation="relu", **kwargs):
        super(ContrastiveEvidential, self).__init__(**kwargs)
        self.mlp1 = Dense(dim, activation=activation)
        self.dim = dim

    def call(self, cls1, cls2):
        input1 = cls1
        input2 = tf.transpose(cls2, perm=[1, 0, 2])
        logit = self.mlp1(input1 - input2)
        return logit

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][0], self.dim)


class ContrastiveDecoder(Layer):
    def __init__(self, cls_dim, key_dim, num_heads, **kwargs):
        super(ContrastiveDecoder, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.cls_dim = cls_dim
        self.num_heads = num_heads
        self.token_embeddings1 = Embedding(
            input_dim=1,
            output_dim=cls_dim,
            embeddings_regularizer=L2(1e-7),
        )
        self.attention2 = MultiHeadAttention(
            num_heads,
            key_dim=key_dim,
            kernel_regularizer=L2(1e-7),
        )
        self.dense1 = Dense(
            cls_dim,
            activation="relu",
            kernel_regularizer=L2(1e-7),
        )
        self.dense2 = Dense(
            cls_dim,
            activation=None,
            kernel_regularizer=L2(1e-7),
        )
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.supports_masking = True

    def call(self, encoder_outputs, vmask):
        clstoken = tf.reduce_sum(tf.zeros_like(vmask), axis=-1, keepdims=True)

        cls1 = self.token_embeddings1(clstoken)
        vmask = tf.cast(vmask[:, tf.newaxis, :], dtype="int32")

        attn2 = self.attention2(
            query=cls1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=vmask,
        )
        out2 = self.layernorm2(cls1 + attn2)

        ff_output = self.dense1(out2)
        ff_output = self.dense2(ff_output)
        output = self.layernorm3(out2 + ff_output)
        return output

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0], 1, self.cls_dim)


class NLL(layers.Layer):
    def __init__(self, numclass=2, usemask=False, name="name"):
        super(NLL, self).__init__(name=name)
        self.numclass = numclass

    def call(self, x):
        evidence, y = x[0], x[1]
        y1hot = tf.one_hot(y, self.numclass)
        opinion = evidence + 1.0
        S = tf.reduce_sum(opinion, axis=-1, keepdims=True)
        ep = tf.divide(opinion, S)

        term1 = tf.square(tf.cast(y1hot, dtype="float32") - ep)
        term2 = ep * (1.0 - ep) / (S + 1.0)
        loss = tf.reduce_sum(term1 + term2, axis=-1, keepdims=True)
        return loss


class NLL2(layers.Layer):
    def __init__(
        self, numclass=2, usemask=False, sparselabel=False, div=1, name="name"
    ):
        super(NLL2, self).__init__(name=name)
        self.numclass = numclass
        self.div = div
        self.sparselabel = sparselabel

    def call(self, x):
        evidence, y = x[0], x[1]
        if self.sparselabel:
            y1hot = tf.one_hot(y, self.numclass)
        else:
            y1hot = y
        opinion = evidence + 1.0 / self.div
        S = tf.reduce_sum(opinion, axis=-1, keepdims=True)
        term1 = y1hot * (tf.math.log(S) - tf.math.log(opinion))
        loss = tf.reduce_sum(term1, axis=-1, keepdims=True)
        return loss


class NLLsoftmax(layers.Layer):
    def __init__(self, numclass=2, usemask=False, name="name"):
        super(NLL2, self).__init__(name=name)
        self.numclass = numclass

    def call(self, x):
        evidence, y = x[0], x[1]
        y1hot = tf.one_hot(y, self.numclass)
        opinion = evidence + 1.0
        S = tf.reduce_sum(opinion, axis=-1, keepdims=True)
        term1 = y1hot * (tf.math.log(S) - tf.math.log(opinion))
        loss = tf.reduce_sum(term1, axis=-1, keepdims=True)
        return loss


class KL1(layers.Layer):
    def __init__(self, numclass=2, sparselabel=True, name="name"):
        super(KL1, self).__init__(name=name)
        self.numclass = numclass
        self.sparselabel = sparselabel

    def call(self, x):
        evidence, y = x[0], x[1]
        if self.sparselabel:
            y1hot = tf.one_hot(y, self.numclass)
        else:
            y1hot = y
        opinion = evidence + 1.0
        tilde_a = (
            (1.0 - y1hot) * opinion / tf.reduce_sum(opinion, axis=-1, keepdims=True)
        )
        sum_a = tf.reduce_sum(tilde_a, axis=-1, keepdims=True)
        return sum_a


class KL2(layers.Layer):
    def __init__(self, numclass=2, sparselabel=True, name="name"):
        super(KL2, self).__init__(name=name)
        self.numclass = numclass
        self.sparselabel = sparselabel

    def call(self, x):
        evidence, y = x[0], x[1]
        if self.sparselabel:
            y1hot = tf.one_hot(y, self.numclass)
        else:
            y1hot = y
        tilde_a = (1.0 - y1hot) * evidence
        sum_a = tf.reduce_sum(tilde_a, axis=-1, keepdims=True)

        return sum_a


def ViTcontrastive(
    sequence_length,
    cap_length,
    embed_dim,
    text_dim,
    vocsize,
    nblock,
    nbatch,
    dropout=None,
):
    x_features = keras.Input(shape=(sequence_length, embed_dim), name="x_feature")
    x_vmask = keras.Input(shape=(sequence_length), name="x_vmask")
    x_token1 = keras.Input(shape=(cap_length), name="x_token1")
    x_token2 = keras.Input(shape=(cap_length), name="x_token2")
    x_tmask = Lambda(lambda x: tf.cast(x > 0, dtype="float32"), name="x_tmask")(
        x_token1
    )
    x_gt = keras.Input(shape=(nbatch), dtype="int32", name="x_gt")

    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )([x_features, x_vmask])

    x = TransformerEncoder(embed_dim, 64, 8, name="transformer_layer")(x, mask=x_vmask)

    seq0 = SeqEmbedding(
        cap_length, V=vocsize, embed_dim=text_dim, name="seq_embedding"
    )(x_token1)
    seq = Lambda(lambda x: x, name="same")(seq0)
    for i in range(nblock):
        seq = Decoder(text_dim, 64, num_heads=8, name="decoderl" + str(i))(
            seq, x, x_vmask
        )
        if dropout is not None:
            seq = Dropout(dropout, name="dropa" + str(i))(seq)

    for i in range(1):
        fd = TimeDistributed(
            Dense(
                text_dim,
                activation="relu",
                kernel_regularizer=L2(1e-7),
            ),
            name="finalc" + str(i),
        )(seq)
    lnow = TimeDistributed(
        Dense(
            vocsize,
            activation="softmax",
            kernel_regularizer=L2(1e-7),
        ),
        name="lnow",
    )(fd)

    cls1 = ContrastiveDecoder(
        embed_dim, key_dim=64, num_heads=8, nbatch=nbatch, name="cls1"
    )(x, x_vmask)
    seq2 = TransformerEncoder(text_dim, 64, 8, name="cls20")(seq0, mask=x_tmask)
    cls2 = ContrastiveDecoder(
        embed_dim, key_dim=64, num_heads=8, nbatch=nbatch, name="cls2"
    )(seq2, x_tmask)
    contrast = ContrastiveEvidential(nbatch, name="contrast1")(cls1, cls2)
    contrast = TimeDistributed(
        Dense(
            2,
            activation="exponential",
            kernel_regularizer=L2(1e-7),
        ),
        name="contrast2",
    )(contrast)
    nll = NLL(name="nll")([contrast, x_gt])
    kl = KL2(name="kl")([contrast, x_gt])

    model = keras.Model(
        [x_features, x_vmask, x_token1, x_token2, x_gt], [lnow, nll, kl]
    )
    return model


class KL0(layers.Layer):
    def __init__(self, numclass=2, sparselabel=True, name="name"):
        super(KL0, self).__init__(name=name)
        self.numclass = numclass
        self.sparselabel = sparselabel
        self.lgammak = scipy.special.loggamma(numclass)

    def call(self, x):
        evidence, y = x[0], x[1]
        if self.sparselabel:
            y1hot = tf.one_hot(y, self.numclass)
        else:
            y1hot = y
        opinion = evidence + 1.0
        tilde_a = y1hot + (1.0 - y1hot) * opinion
        sum_a = tf.reduce_sum(tilde_a, axis=-1, keepdims=True)

        term1 = tf.math.lgamma(sum_a)
        term2 = tf.reduce_sum(tf.math.lgamma(tilde_a), axis=-1, keepdims=True)
        term3 = tf.math.digamma(tilde_a) - tf.math.digamma(sum_a)

        term4 = tf.reduce_sum((tilde_a - 1) * term3, axis=-1, keepdims=True)
        output = (
            tf.reduce_sum(term1 - term2, axis=-1, keepdims=True) + term4 - self.lgammak
        )
        return output


def ViTEviContra(
    sequence_length,
    cap_length,
    embed_dim,
    text_dim,
    vocsize,
    nbatch,
    nblock=2,
    dropout=None,
    sparselabel=True,
    perceiver=0,
):
    x_features = keras.Input(shape=(sequence_length, embed_dim), name="x_feature")
    x_vmask = keras.Input(shape=(sequence_length), name="x_vmask")
    x_token1 = keras.Input(shape=(cap_length), dtype="int32", name="x_token1")
    x_token2 = keras.Input(shape=(cap_length), dtype="int32", name="x_token2")
    x_tmask = Lambda(lambda x: tf.cast(x > 0, dtype="float32"), name="x_tmask")(
        x_token1
    )
    x_gt = keras.Input(shape=(nbatch), dtype="int32", name="x_gt")

    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )([x_features, x_vmask])

    x = TransformerEncoder(embed_dim, 64, 8, name="transformer_layer")(x, mask=x_vmask)

    seq0 = SeqEmbedding(
        cap_length, V=vocsize, embed_dim=text_dim, name="seq_embedding"
    )(x_token1)
    seq = Lambda(lambda x: x, name="same")(seq0)
    for i in range(nblock):
        seq = Decoder(text_dim, 64, num_heads=8, name="decoderl" + str(i))(
            seq, x, x_vmask
        )
        if dropout is not None:
            seq = Dropout(dropout, name="dropa" + str(i))(seq)

    for i in range(1):
        fd = TimeDistributed(
            Dense(
                text_dim,
                activation="relu",
                kernel_regularizer=L2(1e-7),
            ),
            name="finalc" + str(i),
        )(seq)
    lnow = TimeDistributed(
        Dense(
            vocsize,
            activation="exponential",
            kernel_regularizer=L2(1e-7),
            activity_regularizer=L2(1e-12),
        ),
        name="lnow",
    )(fd)
    nll1 = NLL2(numclass=vocsize, sparselabel=sparselabel, name="nll1")(
        [lnow, x_token2]
    )
    kl1 = KL1(numclass=vocsize, sparselabel=sparselabel, name="kl1")([lnow, x_token2])

    cls1 = ContrastiveDecoder(
        embed_dim, key_dim=64, num_heads=8, nbatch=nbatch, name="cls1"
    )(x, x_vmask)
    seq2 = TransformerEncoder(text_dim, 64, 8, name="cls20")(seq0, mask=x_tmask)
    cls2 = ContrastiveDecoder(
        embed_dim, key_dim=64, num_heads=8, nbatch=nbatch, name="cls2"
    )(seq2, x_tmask)
    contrast = ContrastiveEvidential(nbatch, name="contrast1")(cls1, cls2)
    contrast = TimeDistributed(
        Dense(
            2,
            activation="exponential",
            kernel_regularizer=L2(1e-7),
        ),
        name="contrast2",
    )(contrast)
    nll2 = NLL2(sparselabel=True, name="nll2")([contrast, x_gt])
    kl2 = KL1(sparselabel=True, name="kl2")([contrast, x_gt])

    model = keras.Model(
        [x_features, x_vmask, x_token1, x_token2, x_gt], [nll1, kl1, nll2, kl2, lnow]
    )
    return model


class PerceiverToken(Layer):
    def __init__(self, perceiver, **kwargs):
        super(PerceiverToken, self).__init__(**kwargs)
        self.perceiver = perceiver

    def call(self, vmask):
        aux_batch = tf.reduce_sum(
            tf.zeros_like(vmask, dtype=tf.int32), axis=-1, keepdims=True
        )
        clstoken = tf.expand_dims(tf.range(self.perceiver), axis=0)
        clstoken = clstoken + aux_batch
        return clstoken


def ViTEviPerceiver(
    sequence_length,
    cap_length,
    video_dim,
    text_dim,
    vocsize,
    nbatch,
    nblock=2,
    dropout=None,
    sparselabel=True,
    perceiver=16,
):
    x_features = keras.Input(shape=(sequence_length, video_dim), name="x_feature")
    x_vmask = keras.Input(shape=(sequence_length), name="x_vmask")
    x_token1 = keras.Input(shape=(cap_length), dtype="int32", name="x_token1")
    x_token2 = keras.Input(shape=(cap_length), dtype="int32", name="x_token2")
    x_tmask = Lambda(lambda x: tf.cast(x > 0, dtype="float32"), name="x_tmask")(
        x_token1
    )
    x_gt = keras.Input(shape=(nbatch), dtype="int32", name="x_gt")
    x_perceiver = PerceiverToken(perceiver, name="prec_token")(x_vmask)
    x = Dense(text_dim, activation=None, kernel_regularizer=L2(1e-7), name="imgmap")(
        x_features
    )
    x = PositionalEmbedding(sequence_length, text_dim, name="frame_position_embed")(
        [x, x_vmask]
    )
    x2 = Embedding(
        input_dim=perceiver,
        output_dim=text_dim,
        embeddings_regularizer=L2(1e-7),
        name="perc_embed",
    )(x_perceiver)
    x = Decoder(text_dim, 64, num_heads=8, selfattn=False, name="xattn_perc")(
        x2, x, x_vmask
    )
    x_percmask = Lambda(lambda x: tf.ones_like(x), name="precmask")(x_perceiver)

    seq0 = SeqEmbedding(
        cap_length, V=vocsize, embed_dim=text_dim, name="seq_embedding"
    )(x_token1)
    seq = Lambda(lambda x: x, name="same")(seq0)
    for i in range(nblock):
        seq = Decoder(text_dim, 64, num_heads=8, name="decoderl" + str(i))(
            seq, x, x_percmask
        )
        if dropout is not None:
            seq = Dropout(dropout, name="dropa" + str(i))(seq)

    for i in range(1):
        fd = TimeDistributed(
            Dense(
                text_dim,
                activation="relu",
                kernel_regularizer=L2(1e-7),
            ),
            name="finalc" + str(i),
        )(seq)
    lnow = TimeDistributed(
        Dense(
            vocsize,
            activation="exponential",
            kernel_regularizer=L2(1e-7),
            activity_regularizer=L2(1e-12),
        ),
        name="lnow",
    )(fd)
    nll1 = NLL2(numclass=vocsize, sparselabel=sparselabel, name="nll1")(
        [lnow, x_token2]
    )
    kl1 = KL1(numclass=vocsize, sparselabel=sparselabel, name="kl1")([lnow, x_token2])

    cls1 = ContrastiveDecoder(text_dim, key_dim=64, num_heads=8, name="cls1")(
        x, x_percmask
    )
    seq2 = TransformerEncoder(text_dim, 64, 8, name="cls20")(seq0, mask=x_tmask)
    cls2 = ContrastiveDecoder(text_dim, key_dim=64, num_heads=8, name="cls2")(
        seq2, x_tmask
    )
    contrast = ContrastiveEvidential(name="contrast1")(cls1, cls2)
    contrast = TimeDistributed(
        Dense(
            2,
            activation="exponential",
            kernel_regularizer=L2(1e-7),
        ),
        name="contrast2",
    )(contrast)
    nll2 = NLL2(sparselabel=True, name="nll2")([contrast, x_gt])
    kl2 = KL1(sparselabel=True, name="kl2")([contrast, x_gt])

    model = keras.Model(
        [x_features, x_vmask, x_token1, x_token2, x_gt], [nll1, kl1, nll2, kl2, lnow]
    )
    return model


def modelSeq(sequence_length, cap_length, embed_dim, text_dim, vocsize):
    time_steps_encoder = sequence_length
    num_encoder_tokens = embed_dim
    latent_dim = text_dim
    time_steps_decoder = cap_length
    num_decoder_tokens = vocsize

    x_vmask = keras.Input(shape=(sequence_length), dtype="bool", name="x_vmask")
    onehotlayer = Embedding(
        input_dim=vocsize,
        output_dim=text_dim,
        embeddings_regularizer=L2(1e-7),
        name="one_hot",
    )

    encoder_inputs = keras.Input(
        shape=(time_steps_encoder, num_encoder_tokens), name="encoder_inputs"
    )
    encoder = LSTM(
        latent_dim,
        return_state=True,
        return_sequences=True,
        kernel_regularizer=L2(1e-7),
        recurrent_regularizer=L2(1e-7),
        bias_regularizer=L2(1e-7),
        name="endcoder_lstm",
    )
    _, state_h, state_c = encoder(encoder_inputs, mask=x_vmask)
    encoder_states = [state_h, state_c]

    decoder_inputs = keras.Input(
        shape=(time_steps_decoder), dtype="int32", name="decoder_inputs"
    )
    decoder_onehot = onehotlayer(decoder_inputs)

    decoder_lstm = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        kernel_regularizer=L2(1e-7),
        recurrent_regularizer=L2(1e-7),
        bias_regularizer=L2(1e-7),
        name="decoder_lstm",
    )
    decoder_outputs, _, _ = decoder_lstm(decoder_onehot, initial_state=encoder_states)
    decoder_dense = Dense(
        latent_dim,
        activation="relu",
        name="dense",
        kernel_regularizer=L2(1e-7),
        bias_regularizer=L2(1e-7),
    )(decoder_outputs)
    lnow = Dense(
        num_decoder_tokens,
        activation="softmax",
        name="lnow",
        kernel_regularizer=L2(1e-7),
        bias_regularizer=L2(1e-7),
    )(decoder_dense)

    model = keras.Model([encoder_inputs, x_vmask, decoder_inputs], lnow)
    return model


def ViTEviPerceiverkw(
    sequence_length,
    cap_length,
    video_dim,
    text_dim,
    vocsize,
    kwsize,
    nbatch,
    nblock=2,
    dropout=None,
    sparselabel=True,
    perceiver=16,
):
    x_features = keras.Input(shape=(sequence_length, video_dim), name="x_feature")
    x_vmask = keras.Input(shape=(sequence_length), name="x_vmask")
    x_token1 = keras.Input(shape=(cap_length), dtype="int32", name="x_token1")
    x_token2 = keras.Input(shape=(cap_length), dtype="int32", name="x_token2")
    x_tmask = Lambda(lambda x: tf.cast(x > 0, dtype="float32"), name="x_tmask")(
        x_token1
    )
    x_gt = keras.Input(shape=(nbatch), dtype="int32", name="x_gt")
    x_kw = keras.Input(shape=(1), dtype="int32", name="x_kw")
    x_perceiver = PerceiverToken(perceiver, name="prec_token")(x_vmask)
    x = Dense(text_dim, activation=None, kernel_regularizer=L2(1e-7), name="imgmap")(
        x_features
    )
    x = PositionalEmbedding(sequence_length, text_dim, name="frame_position_embed")(
        [x, x_vmask]
    )
    x2 = Embedding(
        input_dim=perceiver,
        output_dim=text_dim,
        embeddings_regularizer=L2(1e-7),
        name="perc_embed",
    )(x_perceiver)
    x3 = Embedding(
        input_dim=kwsize,
        output_dim=text_dim,
        embeddings_regularizer=L2(1e-7),
        name="kw_embed",
    )(x_kw)
    x = Decoder(text_dim, 64, num_heads=8, selfattn=False, name="xattn_perc")(
        x2, x, x_vmask
    )
    x_percmask = Lambda(lambda x: tf.ones_like(x), name="precmask")(x_perceiver)

    seq0 = SeqEmbedding(
        cap_length, V=vocsize, embed_dim=text_dim, name="seq_embedding"
    )(x_token1)
    seq = Lambda(lambda x: x, name="same")(seq0)
    for i in range(nblock):
        seq = Decoderkw(text_dim, 64, num_heads=8, name="decoderl" + str(i))(
            seq, x, x_percmask, x3
        )
        if dropout is not None:
            seq = Dropout(dropout, name="dropa" + str(i))(seq)

    for i in range(1):
        fd = TimeDistributed(
            Dense(
                text_dim,
                activation="relu",
                kernel_regularizer=L2(1e-7),
            ),
            name="finalc" + str(i),
        )(seq)
    lnow = TimeDistributed(
        Dense(
            vocsize,
            activation="exponential",
            kernel_regularizer=L2(1e-7),
            activity_regularizer=L2(1e-12),
        ),
        name="lnow",
    )(fd)
    nll1 = NLL2(numclass=vocsize, sparselabel=sparselabel, name="nll1")(
        [lnow, x_token2]
    )
    kl1 = KL1(numclass=vocsize, sparselabel=sparselabel, name="kl1")([lnow, x_token2])

    cls1 = ContrastiveDecoder(text_dim, key_dim=64, num_heads=8, name="cls1")(
        x, x_percmask
    )
    seq2 = TransformerEncoder(text_dim, 64, 8, name="cls20")(seq0, mask=x_tmask)
    cls2 = ContrastiveDecoder(text_dim, key_dim=64, num_heads=8, name="cls2")(
        seq2, x_tmask
    )
    contrast = ContrastiveEvidential(name="contrast1")(cls1, cls2)
    contrast = TimeDistributed(
        Dense(
            2,
            activation="exponential",
            kernel_regularizer=L2(1e-7),
        ),
        name="contrast2",
    )(contrast)
    nll2 = NLL2(sparselabel=True, name="nll2")([contrast, x_gt])
    kl2 = KL1(sparselabel=True, name="kl2")([contrast, x_gt])

    model = keras.Model(
        [x_features, x_vmask, x_token1, x_token2, x_gt, x_kw],
        [nll1, kl1, nll2, kl2, lnow],
    )
    return model


def ViTkw(sequence_length, video_dim, text_dim, kwsize):
    x_features = keras.Input(shape=(sequence_length, video_dim), name="x_feature")
    x_vmask = keras.Input(shape=(sequence_length), name="x_vmask")
    x_token2 = keras.Input(shape=(kwsize), dtype="int32", name="x_token2")
    x_perceiver = PerceiverToken(kwsize, name="prec_token")(x_vmask)
    x = Dense(text_dim, activation=None, kernel_regularizer=L2(1e-7), name="imgmap")(
        x_features
    )
    x = PositionalEmbedding(sequence_length, text_dim, name="frame_position_embed")(
        [x, x_vmask]
    )
    x2 = Embedding(
        input_dim=kwsize,
        output_dim=text_dim,
        embeddings_regularizer=L2(1e-7),
        name="perc_embed",
    )(x_perceiver)
    x = Decoder(text_dim, 64, num_heads=8, selfattn=False, name="xattn_perc")(
        x2, x, x_vmask
    )

    for i in range(1):
        fd = TimeDistributed(
            Dense(
                text_dim,
                activation="relu",
                kernel_regularizer=L2(1e-7),
            ),
            name="finalc" + str(i),
        )(x)
    lnow = TimeDistributed(
        Dense(
            2,
            activation="exponential",
            kernel_regularizer=L2(1e-7),
            activity_regularizer=L2(1e-12),
        ),
        name="lnow",
    )(fd)
    nll1 = NLL2(numclass=2, sparselabel=True, name="nll1")([lnow, x_token2])
    kl1 = KL1(numclass=2, sparselabel=True, name="kl1")([lnow, x_token2])
    model = keras.Model([x_features, x_vmask, x_token2], [nll1, kl1, lnow])
    return model


def modifyTensorRowColumn(a, isRow, index, updatedValue, isVector):
	if(not isRow):
		a = tf.transpose(a)
		if(isVector):
			updatedValue = tf.transpose(updatedValue)
	if(index == 0):
		if(isVector):
			values = [updatedValue, a[index+1:]]
		else:
			values = [[updatedValue], a[index+1:]]
	elif(index == a.shape[0]-1):
		if(isVector):
			values = [a[:index], updatedValue]
		else:
			values = [a[:index], [updatedValue]]
	else:
		if(isVector):
			values = [a[:index], updatedValue, a[index+1:]]
		else:
			values = [a[:index], [updatedValue], a[index+1:]]
	a = tf.concat(axis=0, values=values)
	if(not isRow):
		a = tf.transpose(a)
	return a


@keras.saving.register_keras_serializable()
class Perceiver(Layer):
    def __init__(self, embed_dim,key_dim, num_heads, **kwargs):
        super(Perceiver, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention2 = MultiHeadAttention(num_heads, key_dim=key_dim,
                                             kernel_regularizer=L2(1e-7),)
        self.dense1 = Dense(embed_dim, activation="relu",
                            kernel_regularizer=L2(1e-7),)
        self.dense2 = Dense(embed_dim, activation=None,
                            kernel_regularizer=L2(1e-7),)
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, vmask):
        vmask = tf.cast(vmask[:, tf.newaxis, :], dtype="int32")
        
        attn2 = self.attention2(
            query=inputs, value=encoder_outputs, key=encoder_outputs,
            attention_mask=vmask,)
        out2 = self.layernorm2(inputs + attn2)
        
        ff_output = self.dense1(out2)
        ff_output = self.dense2(ff_output)
        output = self.layernorm3(out2 + ff_output)
        
        return output

    def get_causal_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, T = input_shape[0], input_shape[1]
        i = tf.range(T)[:, tf.newaxis]
        j = tf.range(T)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, T, T))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype="int32")],
            axis=0,)
        output = tf.tile(mask, mult)
        return output


def GemmaBackbone2(
        vocabulary_size=256000,
        num_layers=18,
        num_query_heads = 8,
        num_key_value_heads = 1,
        hidden_dim = 2048,
        intermediate_dim = 32768,
        head_dim = 256,
        layer_norm_epsilon=1e-6,
        dropout=0,
        dtype=None,):
        token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=True,
            embeddings_initializer=keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_in",
                distribution="untruncated_normal",
                seed=None,
            ),
            dtype=dtype,
            name="token_embedding",
        )
        transformer_layers = []
        for i in range(num_layers):
            layer = GemmaDecoderBlock(
                intermediate_dim=intermediate_dim,
                hidden_dim=hidden_dim,
                num_query_heads=num_query_heads,
                head_dim=head_dim,
                num_key_value_heads=num_key_value_heads,
                dropout=dropout,
                dtype=dtype,
                name=f"decoder_block_{i}",
            )
            transformer_layers.append(layer)
        layer_norm = RMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="final_normalization",
        )
        evi_layer = EviLayer(output_dim=vocabulary_size, name="lnow")

        visual_features = keras.Input(
            shape=(sequence_length,dim_visual), dtype="float32", name="visual_features"
        )
        x_vmask = keras.Input(shape=(sequence_length,), name="x_vmask")
        x_perceiver_token = PerceiverToken(size_perceiver, name="prec_token")(x_vmask)
        x2 = Embedding(input_dim=size_perceiver, output_dim=hidden_dim,
            embeddings_regularizer=L2(1e-7), name="perc_embed")(x_perceiver_token)
        x_perceiver = Perceiver(hidden_dim, 64, 8, name='xattn_perc')(x2, visual_features, x_vmask) 
        token_id_input = keras.Input(
            shape=(None,), dtype="float32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="float32", name="padding_mask"
        )
        x = token_embedding(token_id_input)
        x = x * ops.cast(ops.sqrt(hidden_dim), x.dtype)
        
        x = Concatenate(axis=-2, name="concat")([x_perceiver, x]) 
        for transformer_layer in transformer_layers:
            x = transformer_layer(x, padding_mask=padding_mask_input)
        sequence_output = layer_norm(x)
        sequence_output = token_embedding(sequence_output, reverse=True)
        lnow = evi_layer(sequence_output)
        
        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input, 
            "visual_features": visual_features,
            "x_vmask": x_vmask,
        }
        
        model = keras.Model(inputs, lnow)
        return model




def get_lora(rank):
    """Enable Lora on the backbone.

    Calling this method will freeze all weights on the backbone,
    while enabling Lora on the query & value `EinsumDense` layers
    of the attention layers.
    """
    gemma_lm = GemmaBackbone2()
    
    gemma_lm._custom_layers = []
    for layer in gemma_lm.layers:
        name = layer.name
        if name in ["token_embedding", "final_normalization"] or name.startswith("decoder_block_"):
            layer.trainable = False
        else:
            if layer.weights:
                gemma_lm._custom_layers.append(name)
                
        
    target_names = ["query_dense", "value_dense", "query", "value"]
    gemma_lm._lora_enabled_layers = []
    gemma_lm._lora_rank = rank
    all_layers = gemma_lm._flatten_layers(include_self=False)
    all_layers = [lyr for lyr in all_layers if lyr.weights]
    for i, layer in enumerate(all_layers):
        for name in target_names:
            if layer.name == name:
                if hasattr(layer, "enable_lora"):
                    layer.trainable = True
                    layer.enable_lora(rank)
                    gemma_lm._lora_enabled_layers.append(i)
    return gemma_lm

def save_lora_weights(gemma_lm, filepath):
    if not getattr(gemma_lm, "_lora_enabled_layers", []):
        raise ValueError(
            "There are no lora-enabled layers in this model. "
            "Make sure to call `.enable_lora(rank)` first."
        )
    if not str(filepath).endswith(".lora.h5"):
        raise ValueError(
            "The filename must end in `.lora.h5`. "
            f"Received: filepath={filepath}"
        )

    store = keras.src.saving.saving_lib.H5IOStore(filepath, mode="w")
    lora_store = store.make("lora")
    lora_store["rank"] = gemma_lm._lora_rank
    all_layers = gemma_lm._flatten_layers(include_self=False)
    all_layers = [lyr for lyr in all_layers if lyr.weights]
    for layer_index in gemma_lm._lora_enabled_layers:
        layer = all_layers[layer_index]
        inner_store = store.make(f"lora/{layer_index}")
        inner_store["lora_kernel_a"] = layer.lora_kernel_a
        inner_store["lora_kernel_b"] = layer.lora_kernel_b
    
    custome_store = store.make("custom")
    for name in gemma_lm._custom_layers:
        weights = gemma_lm.get_layer(name = name).get_weights()
        custome_store[name] = weights
    store.close()

def load_lora_weights(gemma_lm, filepath):
    store = keras.src.saving.saving_lib.H5IOStore(filepath, mode="r")
    lora_store = store.get("lora")
    rank = int(lora_store["rank"][()])

    if not getattr(gemma_lm, "_lora_enabled_layers", []):
        gemma_lm.enable_lora(rank)
    else:
        if gemma_lm._lora_rank != rank:
            raise ValueError(
                f"The Lora rank expected by file '{filepath}' "
                f"is rank={rank}, but the model was called with "
                f"`.enable_lora(rank={gemma_lm._lora_rank})`. "
                "Both ranks must match."
            )
    all_layers = gemma_lm._flatten_layers(include_self=False)
    all_layers = [lyr for lyr in all_layers if lyr.weights]
    for layer_index in gemma_lm._lora_enabled_layers:
        layer = all_layers[layer_index]
        lora_kernel_a = store.get(f"lora/{layer_index}")["lora_kernel_a"]
        lora_kernel_b = store.get(f"lora/{layer_index}")["lora_kernel_b"]
        layer.lora_kernel_a.assign(lora_kernel_a)
        layer.lora_kernel_b.assign(lora_kernel_b)
    
    for name in gemma_lm._custom_layers:
        weights = store.get("custom")[name]
        gemma_lm.get_layer(name = name).set_weights(weights)
    store.close()


def evi_loss(y_true, y_pred):
    numclass = vocsize
    alpha = 1
    sparselabel = True
    
    a = tf.exp(y_pred) + 1
    if sparselabel:
        y1hot = tf.one_hot(y_true, numclass)
    else:
        y1hot = y_true
    S = tf.reduce_sum(a, axis=-1, keepdims=True)
    term1 = y1hot * (tf.math.log(S) - tf.math.log(a))
    nll = tf.reduce_sum(term1, axis=-1, keepdims=True)
    
    tilde_a = (1.-y1hot)*a / tf.reduce_sum(a, axis=-1, keepdims=True)
    sum_a = tf.reduce_sum(tilde_a, axis=-1, keepdims=True)
    total = nll + alpha * sum_a
    return total


@keras.saving.register_keras_serializable()
class EviLayer(keras.layers.Layer):
    def __init__(
        self,
        output_dim,
        rank=8,
        alpha=32,
        trainable=True,
        **kwargs,
    ):
        super(EviLayer, self).__init__(**kwargs)

        self.rank = rank
        self.alpha = alpha
        self._scale = alpha / rank

        self.A = keras.layers.Dense(
            units=rank,
            use_bias=False,
            kernel_initializer=keras.initializers.VarianceScaling(
                scale=math.sqrt(5), mode="fan_in", distribution="uniform"
            ),
            trainable=trainable,
            name="lora_A",
        )
        self.B = keras.layers.Dense(
            units=output_dim,
            use_bias=False,
            kernel_initializer="zeros",
            trainable=trainable,
            name="lora_B",
        )
        self.w = self.add_weight(
            shape=(1,),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        lora_output = self.B(self.A(inputs)) * self._scale
        scaled = inputs * tf.exp(self.w)
        output = scaled + lora_output
        return output
