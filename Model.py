import tensorflow as tf

from emnist import extract_training_samples, extract_test_samples



# Load the EMNIST dataset
def create_model():
    #(x_train, y_train), (x_test, y_test) = emnist.load_data(type='balanced')
    x_train, y_train = extract_training_samples('balanced')
    x_test, y_test = extract_test_samples('balanced')
    # Preprocess the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    y_train = tf.keras.utils.to_categorical(y_train, 47)
    y_test = tf.keras.utils.to_categorical(y_test, 47)

    # Define the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(47, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)

    return model
