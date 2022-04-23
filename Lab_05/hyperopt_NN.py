# ------------------------------------------------------------------------
# packages

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
import matplotlib.pyplot as plt
import numpy as np
import argparse

#-------------------------------------------------------------------------
# removing cuda-tf logs

import logging, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

#-------------------------------------------------------------------------
# functions

def train(x, y, hyperpars, epochs):
    #sequential model
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    #hyperoptimizing the number of layers and their size
    for i in range(hyperpars['n_layers']):
        model.add(Dense(units=hyperpars['n_nodes'], activation='relu'))
    model.add(Dense(10, activation='softmax'))
    #hyperoptimizing the learning rate
    adam = Adam(lr=hyperpars['learning_rate'])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(x, y, epochs=epochs)
    return model


def data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test


def test(model, features, labels):
    acc = model.evaluate(features, labels)
    return acc


#------------------------------------------------------------------------
# main

def main(epochs):
    # the dataset
    x_train, y_train, x_test, y_test = data()

    def hyperfun(hypersample):
        model = train(x_train, y_train, hypersample, epochs)
        test_acc = test(model, x_test, y_test)
        # follows the loss fz, which is -accuracy (it's a minimization process)
        # and we also want the biggest absolute value for accuracy
        return {'loss': -test_acc[1], 'status': STATUS_OK}

    hypersample = {
        'learning_rate' : hp.loguniform('learning_rate', -10, 0),
        'n_layers'      : hp.uniformint('n_layers', 1, 5),
        'n_nodes'       : hp.uniformint('n_nodes', 14, 100)
    }

    trials = Trials()
    best = fmin(hyperfun, hypersample, algo=tpe.suggest, max_evals=100, trials=trials)
    print(space_eval(hypersample, best))

    iterations =  [t['tid'] for t in trials.trials]

    plt.figure(figsize=(15,6))

    plt.subplot(1,3,1)
    xs = [t['misc']['vals']['n_nodes'] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]
    plt.scatter(xs, ys, s=30, color='purple', alpha=0.5)
    #plt.xlim(xs[0]-1, xs[-1]+1)
    plt.xlabel('Nodes')
    plt.ylabel('Accuracy')

    xs = [t['misc']['vals']['n_layers'] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]

    plt.subplot(1,3,2)
    plt.scatter(xs, ys, s=30, color='purple', alpha=0.5)
    plt.xlabel('Layers')
    plt.ylabel('Accuracy')

    xs = [t['misc']['vals']['learning_rate'] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]

    plt.subplot(1,3,3)
    plt.scatter(xs, ys, s=30, color='purple', alpha=0.5)
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig('hyperpars.png')
    plt.show()

    # iterations VS learning rate plot
    plt.figure(figsize=(8,8))
    plt.title('Iterations VS learning rate')
    plt.scatter(iterations, xs, s=50, color='black', marker='s', alpha=0.4)
    plt.xlabel('Iterations')
    plt.ylabel('lr')
    plt.savefig('iter_vs_lr.png')
    plt.show()

#--------------------------------------------------------------------------
# main launch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command line options.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    args = parser.parse_args()
    main(args.epochs)
