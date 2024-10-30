import numpy as np
import tensorflow as tf

def get_data_per_category(df):

    df_dict = {}

    for category in df['CATEGORY'].unique():
        df_dict[category] = df[df['CATEGORY'] == category].drop('CATEGORY', axis=1).sort_values(by='DATE', ascending=True)

    return df_dict


def windowed(dataset, length):
    dataset = dataset.window(length, shift=1, drop_remainder=True)
    return dataset.flat_map(lambda window_ds: window_ds.batch(length))


def set_dataset(series, seq_length=30, ahead=21, batch_size=32, shuffle=False, seed=None):
    ds = windowed(tf.data.Dataset.from_tensor_slices(series), ahead + 1)
    ds = windowed(ds, seq_length).map(lambda S: (S[:, 0], S[:, 1:, 1]))

    if shuffle:
        ds = ds.shuffle(8 * batch_size, seed=seed)
        
    return ds.batch(batch_size)

 