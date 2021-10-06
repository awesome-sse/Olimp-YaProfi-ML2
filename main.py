import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv('train1.csv', sep=',')
df.dropna()

train_labels = df['critical_temperature'].to_numpy()
print(list(train_labels))
test_labels = train_labels
test_labels=test_labels[len(test_labels)-50:len(test_labels)]
train_labels = train_labels[0:len(train_labels)-50]
print(list(train_labels))



del df['critical_temperature']



train_set = df.to_numpy()
test_set = train_set
test_set = test_set[len(test_set)-50:len(test_set)]
train_set = train_set[:len(train_set)-50]

from sklearn.preprocessing import StandardScaler 
scale_features_std = StandardScaler() 
train_set = scale_features_std.fit_transform(train_set)

print(list(train_set))
print(list(train_labels))

model = keras.Sequential([
    keras.layers.Dense(90, activation='sigmoid'),
    keras.layers.Dense(512, activation='sigmoid'),
    keras.layers.Dense(100, activation='sigmoid'),
    keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam',
              loss='mse', metrics=['accuracy'])




model.fit(train_set, train_labels, epochs=10)


test_loss, test_acc = model.evaluate(test_set,  test_labels, verbose=2)

predictions = model.predict(test_set)

print(list(predictions))

print('\nТочность на проверочных данных:', test_acc)

'''test_set1 = df1.to_numpy()
scale_features_std = StandardScaler() 
test_set1 = scale_features_std.fit_transform(test_set1)
predictions = model.predict(test_set1)

print(list(predictions))'''

'''rezult=[]
for i in list(predictions):
    if i[0]<0.4: rezult.append('OFFSHORE')
    elif i[0]<=0.6: rezult.append('ONSHORE-OFFSHORE')
    else: rezult.append('ONSHORE')
print(rezult)

df2 = pd.DataFrame(rezult)
df2.columns = ['' for i in range(df2.shape[1])]
print(df2.describe())
df2.to_csv('prediction.csv')'''
