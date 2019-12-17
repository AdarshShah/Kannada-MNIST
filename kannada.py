import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Reshape
import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train_data = pd.read_csv("train.csv")

train_x = train_data.iloc[:,1:]
train_x = StandardScaler().fit_transform(train_x)

train_y = train_data.iloc[:,0:1]
train_y = np.reshape(np.array(train_y),(-1,1))
ohe = OneHotEncoder(categories='auto')
train_y = ohe.fit_transform(train_y)

model = Sequential()
model.add(Reshape((1,28,28),input_shape=(784,)))
model.add(Conv2D(32,kernel_size=(3,3),padding='valid',data_format='channels_first'))
model.add(Activation(keras.activations.relu))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy)

model.fit(x=np.array(train_x),y=train_y,batch_size=250,epochs=5)

test_data = pd.read_csv("test.csv")

test_x = test_data.iloc[:,1:]
test_x = StandardScaler().fit_transform(test_x)

labels = model.predict(test_x)
labels = np.dot(labels,np.reshape(ohe.categories_,(10)))

submissions = pd.read_csv("sample_submission.csv")
submissions.iloc[:,1] = labels
submissions.to_csv("submission.csv")
