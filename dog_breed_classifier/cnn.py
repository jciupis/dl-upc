from dogs import Dogs
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import plotter as plt

# Load the Stanford dogs dataset
(x_train, y_train), (x_test, y_test) = Dogs(128, 128).load_data()

# Set constants
train_set_len = x_train.shape[0]
test_set_len = x_test.shape[0]
input_len = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
nb_breeds = y_train.shape[1]
batch_size = 128
nb_epochs = 500

# Calculate the dimensionality of the input
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

# Prepare data augmentation generator
image_gen = ImageDataGenerator(
          rotation_range=60,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest')

image_gen.fit(x_train, augment=True)

# Define the NN architecture
nn = Sequential()

nn.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu', input_shape=input_shape))
nn.add(BatchNormalization())
nn.add(Dropout(0.5))

nn.add(Conv2D(128, (3, 3), strides=(2, 2), activation='relu'))
nn.add(BatchNormalization())
nn.add(Dropout(0.5))

nn.add(Conv2D(256, (3, 3), strides=(2, 2), activation='relu'))
nn.add(BatchNormalization())
nn.add(Dropout(0.5))

nn.add(Conv2D(512, (3, 3), strides=(2, 2), activation='relu'))
nn.add(BatchNormalization())
nn.add(Dropout(0.5))

nn.add(Flatten())
nn.add(Dense(1024, activation='relu'))
nn.add(BatchNormalization())
nn.add(Dropout(0.5))

nn.add(Dense(nb_breeds, activation='softmax'))

# Compile the NN
nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the NN
history = nn.fit_generator(generator=image_gen.flow(x_train, y_train, batch_size=batch_size),
                           steps_per_epoch=train_set_len // batch_size,
                           epochs=nb_epochs)

# Evaluate the model with test set
score = nn.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plots
plt.plot_accuracy(history)
plt.plot_loss(history)

# Confusion matrix and classification report
plt.get_statistics(nn, x_test, y_test)

