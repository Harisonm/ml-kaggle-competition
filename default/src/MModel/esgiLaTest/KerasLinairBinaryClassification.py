from keras.layers import Dense
import numpy as np
from keras.activations import sigmoid
from keras.optimizers import sgd
from keras.losses import mse
from keras.models import Sequential

X = np.array([
    [0, 0],
    [1, 0],
    [0, 1]
])

Y = np.array([
    1,
    1,
    1
])

model = Sequential()
model.add(Dense(1, activation=sigmoid, input_dim=2))
model.compile(optimizer=sgd(lr=0.1), loss=mse)
model.fit(X, Y, epochs=1000)

print(model.predict(X))


