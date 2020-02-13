import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
import numpy as np
from pickle import dump
import datetime
import os


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe['label']
    dataframe = dataframe.drop(['label'], axis=1)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), np.array(labels)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

def create_model(feature_layer):

        model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(1000, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
        ])

        opt = tf.keras.optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

def train():

    df2020 = pd.read_csv('data/all_tournaments_with_stats_for_DNN2020.csv').dropna()

    df2020.label = df2020.label.replace('joueur0', 0)
    df2020.label = df2020.label.replace('joueur1', 1)

    df2020.pointsDropping0 = df2020.pointsDropping0.apply(lambda x: x.replace(',', '.'))
    df2020.pointsDropping1 = df2020.pointsDropping1.apply(lambda x: x.replace(',', '.'))

    df = pd.read_csv('data/all_tournaments_with_stats_for_DNN.csv').dropna()

    print(df.shape)

    df = pd.concat([df, df2020], axis=0)

    index_to_delete = []
    for i, phase in enumerate(df.phase):
        try:
            int(phase)
        except:
            index_to_delete.append(i)

    df = df.drop(index_to_delete)

    df = df.drop(['index', 'year', 'duree', 'player0', 'player1', 'weight0', 'weight1', 'surface', 'inOut', 'startCareer0',
                  'startCareer1',
                  'winnerRate', 'looserRate', 'winnerRateSurface', 'looserRateSurface', 'phase'], axis=1)

    df.aces1 = (df.aces1 / (df.serviceGamePlayed1 * 4)).astype('float64')
    df.aces0 = df.aces0 / (df.serviceGamePlayed0 * 4)
    df.doubleFautes1 = df.doubleFautes1 / (df.serviceGamePlayed1 * 3.5)
    df.doubleFautes0 = df.doubleFautes0 / (df.serviceGamePlayed0 * 3.5)
    df.RateFace2Face0 = df.RateFace2Face0 / 100
    df.RateFace2Face1 = df.RateFace2Face1 / 100
    df = df.dropna()

    scaler = MinMaxScaler()
    scaler.fit(df)

    df_scaled = scaler.transform(df)
    dataframe = pd.DataFrame(df_scaled, columns=df.columns)

    # save the scaler
    dump(scaler, open('data/scaler.pkl', 'wb'))

    train, test = train_test_split(dataframe, test_size=0.1)
    train, val = train_test_split(train, test_size=0.1)

    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')

    train.head()

    feature_columns = []

    for header in train.columns:
        if header != 'label':
            feature_columns.append(feature_column.numeric_column(header))

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    batch_size = 64
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    model = create_model(feature_layer=feature_layer)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

    print("Start Training Model")

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=50,
              callbacks=[tensorboard_callback, early_stopping])

    model.save_weights('./checkpoint/my_checkpoint')

    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)
