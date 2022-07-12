from tensorflow import keras
import tensorflow as tf
import numpy as np
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt
#lets do this where you feed in just the first time period, B02..B12
def build_model(input_shape,x_test,x_train,y_train):
    output_directory = './namelist/'
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

    file_path = output_directory+'best_model.hdf5'

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
            save_best_only=True)

    #self.callbacks = [reduce_lr,model_checkpoint]
    batch_size = 16
    nb_epochs = 500

    mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

    start_time = time.time() 

    hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
        verbose = True)

    duration = time.time() - start_time

    model.save(output_directory+'last_model.hdf5')

    model = keras.models.load_model(output_directory+'best_model.hdf5')
    plt.plot(hist.history['accuracy'], label='accuracy')
    plt.plot(hist.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig('./namelist/plot.jpg')
    y_pred = model.predict(x_test)
    keras.backend.clear_session()

    return y_pred
def prep_data(band):
    arr = np.load('./namelist/b%s.npy'%band)
    arr = np.swapaxes(arr,0,2)
    arr = np.reshape(arr,(-1,1))
    return arr
if __name__ == "__main__":
    #start_date = pd.to_datetime('2019-01-01')
    #end_date = pd.to_datetime('2020-01-01')
    x_train = np.zeros([9, 653*50*80])
    with Pool() as pool:
        x_train = pool.map(prep_data,['02','03','04','05','06','07','08','11','12'])
    pool.close()
    pool.join()
    x_train = np.asarray(x_train)
    x_train = np.swapaxes(x_train,0,1)
    y_train = x_train[1:]
    y_train = y_train[:,6,:]#takes out band 8
    indices = [i-1 for i in range(653,x_train.shape[0],653)]#change for time step
    x_train = np.delete(x_train,indices,axis = 0)
    y_train = np.delete(y_train,indices,axis = 0)
    indices_test = [i-1 for i in range(652,x_train.shape[0],652)]
    y_test = y_train[indices_test]
    x_test = x_train[indices_test]
    x_train = np.delete(x_train,indices_test,axis = 0)
    y_train = np.delete(y_train,[i for i in indices_test if i != max(indices_test)],axis = 0)
    c_train = [i for i in y_train==0]
    c_train = np.asarray(c_train)
    c_train = c_train.reshape([-1])
    y_train = np.delete(y_train,c_train,axis = 0)
    x_train = np.delete(x_train,c_train,axis = 0)
    c_test = [i for i in y_test==0]
    c_test = np.asarray(c_test)
    c_test = c_test.reshape([-1])
    y_test = np.delete(y_test,c_test,axis = 0)
    x_test = np.delete(x_test,c_test,axis = 0) 
    y_pred = build_model(x_train.shape[1:],x_test,x_train,y_train)
    np.save(y_pred,'./namelist/fcn_out.npy')
    np.save(y_test,'./namelist/benchmark.npy')
