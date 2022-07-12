import fcn
import numpy as np
from multiprocessing import Pool
#lets do this where you feed in just the first time period, B02..B12
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
    x_train = np.asarray(x_test)
    x_train = np.swapaxes(x_test,0,1)
    y_train = x_train[1:]
    y_train = y_train[:][6]#takes out band 8
    indices = [i-1 for i in range(653,x_train.shape[0],653)]#change for time step
    x_train = np.delete(x_train,indices,axis = 0)
    y_train = np.delete(y_train,indices,axis = 0)
    indices_test = [i-1 for i in range(652,x_train.shape[0],652)]
    y_test = y_train[indices_test]
    x_test = x_train[indices_test]
    x_train = np.delete(x_train,indices)
    y_train = np.delete(y_train,indices)
    c_train = [i for i in y_train!=0]
    y_train = y_train[c_train]
    x_train = x_train[c_train]
    c_test = [i for i in y_test!=0]
    y_test = y_train[c_test]
    x_test = x_train[c_test]    
    classifier = fcn.Classifier_FCN(x_train.shape[1:])
    y_pred = classifier.fit(x_train, y_train, x_test)
    np.save(y_pred,'./namelist/fcn_out.npy')