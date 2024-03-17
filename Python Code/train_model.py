import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import keras_tuner as kt
import matplotlib.pyplot as plt

from keras.applications import VGG16, VGG19
from preprocessing_datasets import preprocess
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Input, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

class VGG_Model:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.model_type = args.model_type
        self.tuner = args.tuner
        self.train_data, self.validation_data, self.test_data, self.image_shape = preprocess(args.batch_size)

    def build_vgg(self, hp):
        if args.model_type == 'vgg16':
            vgg = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=self.image_shape.shape))
        if args.model_type == 'vgg19':
            vgg = VGG19(weights='imagenet', include_top=False, input_tensor=Input(shape=self.image_shape.shape))

        vgg.trainable = False

        model = Sequential()

        for vgg_layer in vgg.layers:
            model.add(vgg_layer)

        model.add(Flatten())

        num_hidden_layers = hp.Int('num_hidden_layers', min_value=1, max_value=5, step=1)
        for i in range(num_hidden_layers):
            num_units = hp.Choice(f'num_units_{i}', values=[8, 16, 32, 64])
            dropout_rate = hp.Float(f'dropout_rate_{i}', min_value=0.1, max_value=0.9, step=0.1)

            if i == 0:
                model.add(Dense(num_units, activation='relu'))
            else:
                model.add(Dense(num_units, activation='relu'))
                model.add(Dropout(dropout_rate))

        model.add(Dense(1, activation='sigmoid'))

        optimizer = hp.Choice('optimizer', values=['Adam', 'RMSprop', 'SGD'])
        loss_function = hp.Choice('loss_function', values=['binary_crossentropy', 'hinge', 'squared_hinge'])
        learning_rate = hp.Choice('learning_rate', values=[1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3])
        if optimizer == 'Adam':
            opt = Adam(learning_rate=learning_rate)
            model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'])

        if optimizer == 'RMSprop':
            opt = RMSprop(learning_rate=learning_rate)
            model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'])

        if optimizer == 'SGD':
            opt = SGD(learning_rate=learning_rate)
            model.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'])

        return model

    def train(self, args):
        if not os.path.exists('models'):
            os.makedirs('models')

        model_name = 'models/' + str(self.model_type + '-with-' + self.tuner + '.h5')
        model_name = rename_file_if_exists(model_name)

        cp = ModelCheckpoint(f'{model_name}', monitor='val_loss', save_best_only=True,
                             save_weights_only=False, mode='auto', save_freq='epoch', verbose=1)

        early_stopping = EarlyStopping(monitor='val_loss', patience=25)

        if args.tuner == 'hyperband':
            hyperband_tuner = kt.Hyperband(
                self.build_vgg,
                objective='val_accuracy',
                max_epochs=self.epochs,
                factor=3,
                seed=42,
                directory='tuner',
                project_name=str(self.model_type + '-with-' + self.tuner)
            )

            best_hyperband_param = {}
            hyperband_tuner.search(self.train_data, validation_data=(self.validation_data), epochs=self.epochs,
                                   batch_size=self.batch_size, shuffle=True)

            hyperband_results = hyperband_tuner.results_summary()

            best_hyperband_param = hyperband_tuner.get_best_hyperparameters(num_trials=1)[0]

            best_hyperband_model = hyperband_tuner.hypermodel.build(best_hyperband_param)
            best_hyperband_model = best_hyperband_model.fit(self.train_data, validation_data=(self.validation_data),
                                                            epochs=self.epochs, batch_size=self.batch_size,
                                                            callbacks=[cp, early_stopping], shuffle=True)

            return best_hyperband_param, hyperband_results, best_hyperband_model

        if args.tuner == 'randomsearch':
            randomsearch_tuner = kt.RandomSearch(
                self.build_vgg,
                objective='val_accuracy',
                max_trials=10,
                seed=42,
                executions_per_trial=1,
                directory='tuner',
                project_name=str(self.model_type + '-with-' + self.tuner)
            )

            best_randomsearch_param = {}
            randomsearch_tuner.search(self.train_data, validation_data=(self.validation_data),
                                      epochs=self.epochs, batch_size=self.batch_size, shuffle=True)

            randomsearch_results = randomsearch_tuner.results_summary()

            best_randomsearch_param = randomsearch_tuner.get_best_hyperparameters(num_trials=1)[0]

            best_randomsearch_model = randomsearch_tuner.hypermodel.build(best_randomsearch_param)
            best_randomsearch_model = best_randomsearch_model.fit(self.train_data,
                                                                  validation_data=(self.validation_data),
                                                                  epochs=self.epochs, batch_size=self,
                                                                  callbacks=[cp, early_stopping], shuffle=True)

            return best_randomsearch_param, randomsearch_results, best_randomsearch_model

        if args.tuner == 'bayesian':
            bayesian_tuner = kt.BayesianOptimization(
                self.build_vgg,
                objective='val_accuracy',
                max_trials=10,
                seed=42,
                directory='tuner',
                project_name=str(self.model_type + '-with-' + self.tuner)
            )

            best_bayesian_param = {}
            bayesian_tuner.search(self.train_data, validation_data=(self.validation_data),
                                  epochs=self.epochs, batch_size=self.batch_size, shuffle=True)

            bayesian_results = bayesian_tuner.results_summary()

            best_bayesian_param = bayesian_tuner.get_best_hyperparameters(num_trials=1)[0]

            best_bayesian_model = bayesian_tuner.hypermodel.build(best_bayesian_param)
            best_bayesian_model = best_bayesian_model.fit(self.train_data, validation_data=(self.validation_data),
                                                          epochs=self.epochs, batch_size=self.batch_size,
                                                          callbacks=[cp, early_stopping], shuffle=True)

            return best_bayesian_param, bayesian_results, best_bayesian_model

    def evaluate(self):
        model_name = './models/' + str(self.model_type + '-with-' + self.tuner + '.h5')
        # model_name = rename_file_if_exists(model_name)

        test_data_batches = [(data.numpy(), labels.numpy()) for data, labels in self.test_data]
        test_data, test_labels = zip(*test_data_batches)
        test_data = np.vstack(test_data)
        test_labels = np.hstack(test_labels)

        vgg_model = load_model(model_name)
        loss, accuracy = vgg_model.evaluate(self.test_data)
        print(f'{self.model_type} model with {self.tuner} tuning has loss: {loss} and accuracy: {accuracy}')

        model_prediction = vgg_model.predict(test_data)
        cnf_matrix = confusion_matrix(test_labels, np.round(model_prediction))

        print(classification_report(test_labels, np.round(model_prediction)))
        plot_confusion_matrix(cnf_matrix, self.model_type, self.tuner)

def rename_file_if_exists(file_path):
    if not os.path.exists(file_path):
        return file_path

    index = 1
    base_name, ext = os.path.splitext(file_path)
    new_file_path = f'{base_name}_{index}{ext}'

    while os.path.exists(new_file_path):
        index += 1
        new_file_path = f'{base_name}_{index}{ext}'

    return new_file_path

def plot_confusion_matrix(cnf_matrix, model_type, tuner):
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    plt.title(f'Confusion matrix for {model_type} with {tuner}', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(f'confusion_matrix_for_{model_type}_with_{tuner}.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, default='vgg16', choices=['vgg16', 'vgg19'])
    parser.add_argument('--tuner', type=str, default='hyperband',
                        choices=['hyperband', 'randomsearch', 'bayesian'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)

    args, _ = parser.parse_known_args()
    vgg_model = VGG_Model(args)
    vgg_model.train(args)
    vgg_model.evaluate()