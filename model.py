import fcn
#return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, True)
path = '/namelist/'

def daterange(start_date, end_date):
    for n in range(0,int((end_date - start_date).days),15):
        yield start_date + datetime.timedelta(n)
def preprocess(df):
    
def prep_data(band):
    arr = np.load('./namelist/b%s'%band)
    start_date = pd.to_datetime('2019-01-01')
    end_date = pd.to_datetime('2020-01-01')
    for start_datetime_np in date:
if __name__ == "__main__":
    start_date = pd.to_datetime('2019-01-01')
    end_date = pd.to_datetime('2020-01-01')
    if __name__ == "__main__":
    with Pool() as pool:
        pool.map(prep_data,['02','03','04''05','06','07','08','11','12'])