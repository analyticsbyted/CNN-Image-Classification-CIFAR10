import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return (x_train, y_train), (x_test, y_test)

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), name='conv_1'))
    model.add(MaxPool2D((2, 2), name='maxpool_1'))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv_2'))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv_3'))
    model.add(MaxPool2D((2, 2), name='maxpool_2'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name='dense_1'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax', name='output'))
    return model

def train_model():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    model = create_model()

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    cp = ModelCheckpoint('model/', monitor='val_accuracy', save_best_only=True)

    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(x_test, y_test),
                        callbacks=[cp])

    return history

if __name__ == '__main__':
    train_model()
