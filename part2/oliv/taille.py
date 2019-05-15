from tensorflow.python import keras
#import pandas as pd

#data = pd.read_csv('../train.csv')
#data1 = pd.read_csv('../test.csv')
#print(data['category_id'].value_counts())
#print(data1['category_id'].value_counts())


#L = [[1,2,3,4,5],[6,7,8,9,10]]
#L = [list( map(str,i) ) for i in L]
#print(L)

print(keras.backend.floatx())
keras.backend.set_floatx('float16')
print(keras.backend.floatx())
