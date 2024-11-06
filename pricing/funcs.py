import tensorflow as tf


def get_data_per_category(df):

    df_dict = {}

    for category in df['CATEGORY'].unique():
        df_dict[category] = df[df['CATEGORY'] == category].drop('CATEGORY', axis=1).sort_values(by='DATE', ascending=True)

    return df_dict


def windowed(dataset, length):
    dataset = dataset.window(length, shift=1, drop_remainder=True)
    return dataset.flat_map(lambda window_ds: window_ds.batch(length))


def set_dataset(series, seq_length=30, ahead=21, batch_size=64, shuffle=False, seed=None):
    ds = windowed(tf.data.Dataset.from_tensor_slices(series), ahead + 1)
    ds = windowed(ds, seq_length).map(lambda S: (S[:, 0], S[:, 1:, 0]))

    if shuffle:
        ds = ds.shuffle(8 * batch_size, seed=seed)

    return ds.batch(batch_size)


def make_rnn(hp):
    input_ = tf.keras.layers.Input(shape=[None, 7])
    model_length = hp.Int('model_length', min_value=2, max_value=50, step=2)
    size = hp.Int('size', min_value=1, max_value=500, step=2)
    dropout_percentage = hp.Float('dropout_percentage', min_value=0.2, max_value=0.6, step=0.05)
    chosen_optimizer = 'rmsprop'  # hp.Choice('chosen_optimizer', values=['adam', 'rmsprop', 'sgd', 'nadam'])
    chosen_layer = 'simple'  # hp.Choice('chosen_layer', values=['GRU', 'LSTM', 'simple'])
    chosen_loss = 'huber'  # hp.Choice('chosen_loss', values=['huber', 'mae'])

    if chosen_optimizer == 'adam':
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG')
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    elif chosen_optimizer == 'rmsprop':
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG')
        decay = hp.Float('decay', min_value=0.0, max_value=0.9)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, decay=decay)

    elif chosen_optimizer == 'sgd':
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG')
        momentum = hp.Float('momentum', min_value=0.0, max_value=0.9)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

    elif chosen_optimizer == 'nadam':
        learning_rate = hp.Float('learning_rate', min_value=5e-6, max_value=1e-3, sampling='LOG')
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

    for i in range(model_length):
        if chosen_layer == 'GRU':
            rnn = tf.keras.layers.GRU(size, return_sequences=True)
        elif chosen_layer == 'LSTM':
            rnn = tf.keras.layers.LSTM(size, return_sequences=True)
        elif chosen_layer == 'simple':
            rnn = tf.keras.layers.SimpleRNN(size, return_sequences=True)

        dr = tf.keras.layers.Dropout(rate=dropout_percentage)

        x = dr(rnn(x)) if i != 0 else dr(rnn(input_))

    out_layer = tf.keras.layers.Dense(21)

    output = out_layer(x)

    model = tf.keras.Model(inputs=[input_], outputs=[output])

    if chosen_loss == 'huber':
        loss = tf.keras.losses.Huber()
    else:
        loss = 'mae'

    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])

    return model
