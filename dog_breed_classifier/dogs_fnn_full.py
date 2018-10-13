from dogs import Dogs
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
import plotter as plt

# Load the Stanford dogs dataset
(x_train, y_train), (x_test, y_test) = Dogs().load_data()

# Set constants
train_set_len = x_train.shape[0]
test_set_len = x_test.shape[0]
input_len = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
nb_breeds = y_train.shape[1]
batch_size = 128
nb_epochs = 10

# Reshape the data to make it suitable for a flattened input layer
x_train = x_train.reshape(train_set_len, input_len)
x_test = x_test.reshape(test_set_len, input_len)

# Define the NN architecture
nn = Sequential()
nn.add(Dense(1024, activation='relu', input_dim=input_len))
# nn.add(Dropout(0.5))
nn.add(Dense(256, activation='relu'))
# nn.add(Dropout(0.5))
nn.add(Dense(256, activation='relu'))
# nn.add(Dropout(0.5))
nn.add(Dense(nb_breeds, activation='softmax'))

# Compile the NN
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
nn.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# Train the NN
history = nn.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_split=0.2)

# Evaluate the model with test set
score = nn.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plots
plt.plot_accuracy(history)
plt.plot_loss(history)

# Confusion matrix and classification report
plt.get_statistics(nn, x_test, y_test)

