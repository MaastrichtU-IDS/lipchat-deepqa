from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import *
from keras.models import *
from keras.layers.core import *
from keras.layers import Input, LSTM, Dense, merge

def cnn_bid_lstm_att(embedding, attention, out_size, width):
    encoding_input = Input(shape=(embedding, attention))
    modconv = Conv1D(filters=width, kernel_size=5, padding='same', activation='relu')(encoding_input)
    maxp = MaxPooling1D(pool_size=2)(modconv)
    modconv2 = Conv1D(filters=width, kernel_size=5, padding='same', activation='relu')(maxp)
    maxp2 = MaxPooling1D(pool_size=2)(modconv2)
    drop_out = Dropout(0.2, name='dropout')(maxp2)

    lstm_fwd = LSTM(width, return_sequences=True, name='lstm_fwd')(drop_out)
    lstm_bwd = LSTM(width, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)
    bilstm = merge.concatenate([lstm_fwd, lstm_bwd], name='bilstm')
    drop_out = Dropout(0.2)(bilstm)

    attention = Dense(1, activation='tanh')(drop_out)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)

    sent_representation = merge.dot([attention, drop_out], axes=1)
    out_relu = Dense(width, activation="relu")(sent_representation)
    out = Dense(out_size, activation='softmax')(out_relu)
    output = out
    model = Model(input=[encoding_input], output=output)
    return model

def get_answer(in_str, out_size=5):
    with open('lab_dict_rev.json', 'r') as f:
        lab_dict_rev = json.load(f)

    max_encoder_seq_length = 21
    num_encoder_tokens = 299
    with open('in_tok_index.json', 'r') as f:
        input_token_index = json.load(f)

    model = cnn_bid_lstm_att(num_encoder_tokens, max_encoder_seq_length, 41, 256)
    model.load_weights('mod_words.npy')

    enc_input = np.zeros(
        (1, max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    for t, word in enumerate(in_str.lower().split()):
        if word in input_token_index.keys():
            enc_input[0, t, input_token_index[word]] = 1.

    enc_input = np.swapaxes(enc_input, 1, 2)
    pred = model.predict(enc_input).flatten()
    ret = []
    for i in range(out_size):
        ret.append((lab_dict_rev[str(np.argmax(pred))], (pred.max())))
        pred[np.argmax(pred)] = 0

    return ret

# if __name__ == '__main__':
#     print(get_answer('can the landlord enter my room'))