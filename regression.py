import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

def lataa_data():
    data = pd.read_csv('Codeacademy/DLregression/aineistot/admissions_data.csv')
    print(data.head)
    print(data.dtypes)
    return data

def datan_muokkaus(data):
    labels = data.iloc[:,-1]
    features = data.iloc[:,1:-1]
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.33,random_state = 42)
    columns_trans = data.iloc[:,1:-2].columns
    ct = ColumnTransformer([('normalize', Normalizer(), columns_trans)], remainder='passthrough')
    train_features = ct.fit_transform(train_features)
    test_features = ct.fit_transform(test_features)
    train_features_norm = pd.DataFrame(train_features, columns = features.columns)
    test_features_norm = pd.DataFrame(test_features, columns = features.columns)

    return train_features_norm, test_features_norm, train_labels, test_labels

def mallin_kokoaminen(train_features_norm, train_labels, learning_rate = 0.01, epochs = 500, batch_size = 128):
    malli = tf.keras.models.Sequential()
    malli.add(tf.keras.layers.InputLayer(input_shape=(train_features_norm.shape[1],)))
    malli.add(tf.keras.layers.Dense(8,activation='relu'))
    malli.add(tf.keras.layers.Dense(1))
    print(malli.summary())

    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    malli.compile(
        loss = 'mse',
        metrics = ['mae'],
        optimizer = opt)

    stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        mode='min',
        verbose=1,
        patience=100)

    history = malli.fit(
        train_features_norm,
        train_labels,
        epochs=epochs,
        verbose = 1,
        batch_size = batch_size,
        validation_split = 0.2,
        callbacks=[stop])

    return malli, history

def mallin_arviointi(malli, test_features_norm, test_labels):
    val_mse, val_mae = malli.evaluate(test_features_norm, test_labels, verbose=0)
    print('mse:', val_mse)
    print('mae:', val_mae)
    predicted_values = malli.predict(test_features_norm)
    r2 = r2_score(test_labels, predicted_values)
    print('selitysaste:', r2)

    return predicted_values


def baseline_malli(train_features_norm, train_labels, test_features_norm, test_labels):
    dummy_regr = DummyRegressor(strategy="mean")
    dummy_regr.fit(train_features_norm, train_labels)
    y_pred = dummy_regr.predict(test_features_norm)
    MAE_baseline = mean_absolute_error(test_labels, y_pred)
    print('mae_baseline:', MAE_baseline)


def plot_ennustettu(test_labels, predicted_values):
    plt.scatter(test_labels, predicted_values, label = 'data')
    lims = [0.4,1]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel('Todennäköisyys tulla valituksi')
    plt.ylabel('Mallin ennustama todennäköisyys tulla valituksi')
    plt.plot(lims, lims)
    plt.show()

def piirraloss(history, index = 0, hyperparametri = 0):
    if hyperparametri == 0:
        ax = plt.subplot(2,1, index +1)
    else:
        ax = plt.subplot(2, 3, index + 1)
    ax.plot(history.history['loss'], label = 'training loss')
    ax.plot(history.history['val_loss'], label = 'validation loss')
    ax.set_title('Training and validation loss ' + str(hyperparametri))
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.legend()
    return ax

def piirraacc(history, index = 0, hyperparametri = 0):
    if hyperparametri == 0:
        bx = plt.subplot(2, 1, index + 2)
    else:
        bx = plt.subplot(2, 3, index + 4)
    bx.plot(history.history['mae'], label = 'training mae')
    bx.plot(history.history['val_mae'], label = 'validation mae')
    bx.set_title('Training and validation mae ' + str(hyperparametri))
    bx.set_xlabel('epochs')
    bx.set_ylabel('accuracy')
    bx.legend()
    return bx

def main():
    data = lataa_data()
    train_features_norm, test_features_norm, train_labels, test_labels = datan_muokkaus(data)
    malli, history = mallin_kokoaminen(train_features_norm, train_labels)
    predicted_values = mallin_arviointi(malli, test_features_norm, test_labels)
    plot_ennustettu(test_labels, predicted_values)
    baseline_malli(train_features_norm, train_labels, test_features_norm, test_labels)
    ax = piirraloss(history)
    bx = piirraacc(history)
    (ax, bx) = plt.subplots()
    plt.close(2)
    plt.show()



def hyperparametrit():
    epochs = [200, 400, 600]
    batch_size = [32, 64, 128]
    learning_rate = [0.1, 0.01, 0.001]
    data = lataa_data()
    train_features_norm, test_features_norm, train_labels, test_labels = datan_muokkaus(data)

    
    axeslist = []
    bxeslist = []
    for i in range(3):
        malli, history = mallin_kokoaminen(train_features_norm, train_labels, learning_rate=learning_rate[i])
        mallin_arviointi(malli, test_features_norm, test_labels)
        ax = piirraloss(history, index=i, hyperparametri=learning_rate[i])
        bx = piirraacc(history, index=i, hyperparametri=learning_rate[i])
        axeslist.append(ax)
        bxeslist.append(bx)
    fig, ((axeslist[0], axeslist[1]), (axeslist[2],bxeslist[0]), (bxeslist[1], bxeslist[2])) = plt.subplots(3,2)
    plt.close(2)
    plt.show()


main()
#hyperparametrit()