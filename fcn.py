import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import time
class Classifier_FCN:
    def __init__(self, input_shape, verbose=True,build=True):
        #input shape is 1 row of the csv
        self.output_directory = './namelist/'
        if build == True:
            self.model = self.build_model(input_shape)
            if(verbose==True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory+'fcn_init.hdf5')
        return

    def build_model(self, input_shape):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(1)(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer = keras.optimizers.Adam(), 
            metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=25, 
            min_lr=0.0001)

        file_path = self.output_directory+'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
            save_best_only=True)

        self.callbacks = [reduce_lr,model_checkpoint]

        return model 

    def fit(self, x_train, y_train, x_test):
        #if not tf.test.is_gpu_available:
            #print('error')
            #exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training  
        batch_size = 16
        nb_epochs = 500

        mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

        start_time = time.time() 

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
            verbose=self.verbose, callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory+'last_model.hdf5')

        model = keras.models.load_model(self.output_directory+'best_model.hdf5')
        y_pred = model.predict(x_test)
        keras.backend.clear_session()
        return y_pred
        # convert the predicted from binary to integer