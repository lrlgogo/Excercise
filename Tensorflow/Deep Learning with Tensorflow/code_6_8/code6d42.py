from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential

max_features = 10000
maxlen = 500
batchsize = 128

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features)

x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model=Sequential()
model.add(layers.Embedding(max_features, batchsize))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=batchsize,
                    verbose=2,
                    validation_split=0.2)

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(loss) + 1)

plt.plot(acc, epochs, 'bo', label='Training acc')
plt.plot(val_acc, epochs, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.legend()
plt.figure()

plt.plot(loss, epochs, 'bo', label='Training loss')
plt.plot(val_loss, epochs, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
