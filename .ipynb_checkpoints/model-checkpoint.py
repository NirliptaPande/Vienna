import fcn
#import tensorflow.keras as keras
#import tensorflow_addons as tfa
import numpy as np
#return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, True)
#lets do this where you feed in just the first time period, B02..B12
def prep_data(band):
    arr = np.load('./namelist/b%s'%band)
    arr = np.swapaxes(arr,0,2)
    arr = np.reshape(arr,(-1,1))
    #Z-normalizing
    x_bar = np.mean(arr)
    x_del = np.std(arr)
    arr = (arr-mean)/std
    return arr
if __name__ == "__main__":
    #start_date = pd.to_datetime('2019-01-01')
    #end_date = pd.to_datetime('2020-01-01')
    x_test = np.zeros([9, 653*50*80])
    with Pool() as pool:
        x_test = pool.map(prep_data,['02','03','04''05','06','07','08','11','12'])
    pool.close()
    pool.join()
    y_test = x_test[50*80:]
    y_train = y_test.splice(0,50*80*651)
    y_true = y_test
    x_train = x_test.splice(0,50*80*651)
    x_train = x_train[:50*80]
