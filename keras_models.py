from keras.constraints import maxnorm
from keras.engine import Input
from keras.engine import Model
from keras.engine import merge
from keras.layers import Dropout, Dense, Bidirectional, LSTM, \
    Embedding, GaussianNoise, Activation, Flatten, \
    TimeDistributed, RepeatVector, Permute, MaxoutDense, Highway, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2, WeightRegularizer
from sklearn import preprocessing

from keras_helpers.custom_layers import Attention, AttentionWithContext, MeanOverTime
from utilities.ignore_warnings import set_ignores

set_ignores()


def embeddings_layer(max_length, embeddings, trainable=False, masking=False, scale=False, normalize=False):
    if scale:
        print("Scaling embedding weights...")
        embeddings = preprocessing.scale(embeddings)
    if normalize:
        print("Normalizing embedding weights...")
        embeddings = preprocessing.normalize(embeddings)

    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]

    _embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        input_length=max_length if max_length > 0 else None,
        trainable=trainable,
        mask_zero=masking if max_length > 0 else False,
        weights=[embeddings]
    )

    return _embedding


def weighted_states(activations, rnn_size, input_length, attention="single"):
    if attention == "all":
        attention = Flatten()(activations)
        attention = Dense(input_length, activation='tanh')(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(rnn_size)(attention)
        attention = Permute([2, 1])(attention)
        return merge([activations, attention], mode='mul')
    elif attention == "single":
        attention = TimeDistributed(Dense(1, activation='tanh'))(activations)
        # attention = Dense(1, activation='tanh')(activations)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(rnn_size)(attention)
        attention = Permute([2, 1])(attention)
        return merge([activations, attention], mode='mul')


def get_RNN(unit=LSTM, cells=64, bi=False, return_sequences=True, dropout_U=0., consume_less='cpu', l2_reg=0):
    rnn = unit(cells, return_sequences=return_sequences, consume_less=consume_less, dropout_U=dropout_U,
               W_regularizer=l2(l2_reg))
    if bi:
        rnn = Bidirectional(rnn)
    return rnn


def build_attention_RNN(embeddings, classes, max_length, unit=LSTM, cells=64, layers=1, **kwargs):
    # parameters
    bi = kwargs.get("bidirectional", False)
    noise = kwargs.get("noise", 0.)
    dropout_words = kwargs.get("dropout_words", 0)
    dropout_rnn = kwargs.get("dropout_rnn", 0)
    dropout_rnn_U = kwargs.get("dropout_rnn_U", 0)
    dropout_attention = kwargs.get("dropout_attention", 0)
    dropout_final = kwargs.get("dropout_final", 0)
    attention = kwargs.get("attention", None)
    final_layer = kwargs.get("final_layer", False)
    clipnorm = kwargs.get("clipnorm", 1)
    loss_l2 = kwargs.get("loss_l2", 0.)
    lr = kwargs.get("lr", 0.001)

    model = Sequential()
    model.add(embeddings_layer(max_length=max_length, embeddings=embeddings,
                               trainable=False, masking=True, scale=False, normalize=False))

    if noise > 0:
        model.add(GaussianNoise(noise))
    if dropout_words > 0:
        model.add(Dropout(dropout_words))

    for i in range(layers):
        model.add(get_RNN(unit, cells, bi,
                          return_sequences=(layers > 1 and i < layers - 1) or attention, dropout_U=dropout_rnn_U))
        if dropout_rnn > 0:
            model.add(Dropout(dropout_rnn))

    if attention == "memory":
        model.add(AttentionWithContext())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))
    elif attention == "simple":
        model.add(Attention())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))

    if final_layer:
        model.add(MaxoutDense(100, W_constraint=maxnorm(2)))
        # model.add(Highway())
        if dropout_final > 0:
            model.add(Dropout(dropout_final))

    model.add(Dense(classes, activity_regularizer=l2(loss_l2)))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr), loss='categorical_crossentropy')
    return model


def aspect_RNN(wv, text_length, target_length, loss, activation, **kwargs):
    ######################################################
    # HyperParameters
    ######################################################
    noise = kwargs.get("noise", 0)
    trainable = kwargs.get("trainable", False)
    rnn_size = kwargs.get("rnn_size", 75)
    rnn_type = kwargs.get("rnn_type", LSTM)
    final_size = kwargs.get("final_size", 100)
    final_type = kwargs.get("final_type", "linear")
    use_final = kwargs.get("use_final", False)
    drop_text_input = kwargs.get("drop_text_input", 0.)
    drop_text_rnn = kwargs.get("drop_text_rnn", 0.)
    drop_text_rnn_U = kwargs.get("drop_text_rnn_U", 0.)
    drop_target_rnn = kwargs.get("drop_target_rnn", 0.)
    drop_rep = kwargs.get("drop_rep", 0.)
    drop_final = kwargs.get("drop_final", 0.)
    activity_l2 = kwargs.get("activity_l2", 0.)
    clipnorm = kwargs.get("clipnorm", 5)
    bi = kwargs.get("bi", False)
    lr = kwargs.get("lr", 0.001)

    attention = kwargs.get("attention", "simple")
    #####################################################

    shared_RNN = get_RNN(rnn_type, rnn_size, bi=bi, return_sequences=True, dropout_U=drop_text_rnn_U)
    # shared_RNN = LSTM(rnn_size, return_sequences=True, dropout_U=drop_text_rnn_U)

    input_text = Input(shape=[text_length], dtype='int32')
    input_target = Input(shape=[target_length], dtype='int32')

    ######################################################
    # Embeddings
    ######################################################
    emb_text = embeddings_layer(max_length=text_length, embeddings=wv, trainable=trainable, masking=True)(input_text)
    emb_text = GaussianNoise(noise)(emb_text)
    emb_text = Dropout(drop_text_input)(emb_text)

    emb_target = embeddings_layer(max_length=target_length, embeddings=wv, trainable=trainable, masking=True)(input_target)
    emb_target = GaussianNoise(noise)(emb_target)

    ######################################################
    # RNN - Tweet
    ######################################################
    enc_text = shared_RNN(emb_text)
    enc_text = Dropout(drop_text_rnn)(enc_text)

    ######################################################
    # RNN - Aspect
    ######################################################
    enc_target = shared_RNN(emb_target)
    enc_target = MeanOverTime()(enc_target)
    enc_target = Dropout(drop_target_rnn)(enc_target)
    enc_target = RepeatVector(text_length)(enc_target)

    ######################################################
    # Merge of Aspect + Tweet
    ######################################################
    representation = merge([enc_text, enc_target], mode='concat')
    att_layer = AttentionWithContext if attention == "context" else Attention
    representation = att_layer()(representation)
    representation = Dropout(drop_rep)(representation)

    if use_final:
        if final_type == "maxout":
            representation = MaxoutDense(final_size)(representation)
        else:
            representation = Dense(final_size, activation=final_type)(representation)
        representation = Dropout(drop_final)(representation)

    ######################################################
    # Probabilities
    ######################################################
    probabilities = Dense(1, activation=activation, activity_regularizer=l2(activity_l2))(representation)

    model = Model(input=[input_target, input_text], output=probabilities)
    # model = Model(input=[input_text, input_target], output=probabilities)
    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr), loss=loss)
    return model


def siamese_RNN(wv, sent_length, **params):
    rnn_size = params.get("rnn_size", 100)
    rnn_drop_U = params.get("rnn_drop_U", 0.2)
    noise_words = params.get("noise_words", 0.3)
    drop_words = params.get("drop_words", 0.2)
    drop_sent = params.get("drop_sent", 0.3)
    sent_dense = params.get("sent_dense", 50)
    final_size = params.get("final_size", 100)
    drop_final = params.get("drop_final", 0.5)

    ###################################################
    # Shared Layers
    ###################################################
    embedding = embeddings_layer(max_length=sent_length, embeddings=wv, masking=True)
    encoder = get_RNN(LSTM, rnn_size, bi=False, return_sequences=True, dropout_U=rnn_drop_U)
    attention = Attention()
    sent_dense = Dense(sent_dense, activation="relu")

    ###################################################
    # Input A
    ###################################################
    input_a = Input(shape=[sent_length], dtype='int32')
    # embed sentence A
    emb_a = embedding(input_a)
    emb_a = GaussianNoise(noise_words)(emb_a)
    emb_a = Dropout(drop_words)(emb_a)
    # encode sentence A
    enc_a = encoder(emb_a)
    enc_a = Dropout(drop_sent)(enc_a)
    enc_a = attention(enc_a)
    enc_a = sent_dense(enc_a)
    enc_a = Dropout(drop_sent)(enc_a)

    ###################################################
    # Input B
    ###################################################
    input_b = Input(shape=[sent_length], dtype='int32')
    # embed sentence B
    emb_b = embedding(input_b)
    emb_b = GaussianNoise(noise_words)(emb_b)
    emb_b = Dropout(drop_words)(emb_b)
    # encode sentence B
    enc_b = encoder(emb_b)
    enc_b = Dropout(drop_sent)(enc_b)
    enc_b = attention(enc_b)
    enc_b = sent_dense(enc_b)
    enc_b = Dropout(drop_sent)(enc_b)

    ###################################################
    # Comparison
    ###################################################
    comparison = merge([enc_a, enc_b], mode='concat')
    comparison = MaxoutDense(final_size)(comparison)
    comparison = Dropout(drop_final)(comparison)

    probabilities = Dense(1, activation='sigmoid')(comparison)
    model = Model(input=[input_a, input_b], output=probabilities)

    model.compile(optimizer=Adam(clipnorm=1., lr=0.001), loss='binary_crossentropy', metrics=["binary_accuracy"])
    return model
