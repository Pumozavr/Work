#inputs necessary to run functions build_model and RandomizedSearch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow import keras
import numpy as np
import torch.nn as nn
import torch
import random


#a function that build a model object
def build_model(input_shape = None, output_shape = None, n_hidden = 2, n_neurons=50, learning_rate=3e-3,
                activation = "selu", batch_norm = 0, opt = keras.optimizers.Nadam, dropout = 0, l2 = 0):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    if activation == "selu":
        kernel = "lecun_normal"
    elif (activation == "relu" or activation == "elu"):
        kernel = "he_normal"
    elif (activation == "tanh"):
        kernel = "glorot_normal"
    else:
        kernel = None
    for layer in range(n_hidden):
        if ((layer == 0) or (activation != "selu")) and (batch_norm == 1):
            model.add(keras.layers.BatchNormalization())
        if l2 > 0:
            model.add(keras.layers.Dense(n_neurons, activation=activation, kernel_initializer=kernel, kernel_regularizer=keras.regularizers.l2(l2)))
        else:
            model.add(keras.layers.Dense(n_neurons, activation=activation, kernel_initializer=kernel))
        if ((dropout > 0) and (activation != "selu")):
            model.add(tf.keras.layers.Dropout(dropout))
        elif (dropout > 0):
            model.add(tf.keras.layers.AlphaDropout(dropout))
    if (activation != "selu") and (batch_norm == 1):
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(output_shape))
    model.compile(loss="mean_absolute_error", optimizer=opt(learning_rate = learning_rate))
    return model


#a function that performs randomized search (without cross-validation for now) based on a model constructor (e.g., a build_model function) 
class RandomizedSearch():
    def __init__(self, X, Y, model_constructor, params, test_split = 0.2, val_split = 0.1, n_iter = 10, keras = False, eval_metric = mean_absolute_error,
                 epochs = 100, batch_size = 2056, patience = 10, lr = 0.001, output_iter = False, torch = True, optimizer = torch.optim.NAdam,
                 loss_func = torch.nn.L1Loss()):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_split, random_state=42)
        param_list = list(ParameterSampler(params, n_iter=n_iter, random_state=42))
        best_error = np.inf
        params_n_errors = []
        if keras:
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_split, random_state=42)
            for i in range (n_iter):
                current_model = model_constructor(**param_list[i])
                early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
                current_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), callbacks = [early_stopping_cb], epochs = epochs,
                                  batch_size = batch_size, verbose = 0)
                Y_pred = current_model.predict(X_test)
                current_error = eval_metric(Y_test, Y_pred)
                if ((i == 0) or (current_error < best_error)):
                    best_error = current_error
                    best_model = current_model
                    best_params = param_list[i]
                params_n_errors.append({"error_value": current_error, "params": param_list[i]})
                if output_iter:
                    print("Iteration no. " + str(i + 1) + " is completed")
        elif torch:
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_split, random_state=42)
            for i in range(n_iter):
                current_model = model_constructor(**param_list[i])
                current_lr = random.choice(lr)
                current_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), early_stopping=True,
                                  n_epochs=epochs, lr = current_lr, optimizer = optimizer,
                                  batch_size=batch_size, patience = patience, loss_func = loss_func, verbose=0)
                Y_pred = current_model.predict(X_test)
                current_error = eval_metric(Y_test, Y_pred)
                if ((i == 0) or (current_error < best_error)):
                    best_error = current_error
                    best_model = current_model
                    best_params = param_list[i] + current_lr
                params_n_errors.append({"error_value": current_error, "params": param_list[i]})
                if output_iter:
                    print("Iteration no. " + str(i + 1) + " is completed")
                    print ("Error: " + current_error)
        else:
            for i in range(n_iter):
                current_model = model_constructor(**param_list[i])
                current_model.fit(X_train, Y_train)
                Y_pred = current_model.predict(X_test)
                current_error = eval_metric(Y_test, Y_pred)
                if ((i == 0) or (current_error < best_error)):
                    best_error = current_error
                    best_model = current_model
                    best_params = param_list[i]
                params_n_errors.append({"error_value": current_error, "params": param_list[i]})
        self.params_n_errors = params_n_errors
        self.optimal_model = best_model
        self.optimal_params = best_params


#examples of parameter values for a randomized search based on a neural network
params = {
    "input_shape": [X.shape[1:]],
    "output_shape": [Y.shape[1]],
    "n_hidden": [4],
    "n_neurons": [2000],
    "learning_rate": [0.001],
    "l2": [0.00001, 0.0001, 0.001],
    "activation": ["elu"],
    "batch_norm": [0],
    "dropout": [0],
    "opt": [keras.optimizers.Adamax]
}


#an example of how to run RandomizedSearch
rs = RandomizedSearch(X, Y, build_model, params = params,test_split = 0.15, val_split = 0.1, n_iter = 3, keras = True,
                 epochs = 2000, batch_size = 2056, patience = 30, output_iter=True)