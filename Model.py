import tensorflow as tf
import numpy as np
import emnist as emn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def prox(x):
    return tf.keras.activations.relu(x - 0.1) - tf.keras.activations.relu(-0.1 - x)


# Load the EMNIST dataset
def create_model():
    x_train, y_train = emn.extract_training_samples('balanced')
    x_test, y_test = emn.extract_test_samples('balanced')
    # Preprocess the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_bar = np.mean(x_train.reshape(x_train.shape[0] * 28 * 28))
    x_train = (x_train - x_bar)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = (x_test - x_bar)
    y_train = tf.keras.utils.to_categorical(y_train, 47)
    y_test = tf.keras.utils.to_categorical(y_test, 47)


    activ=tf.keras.layers.LeakyReLU(alpha=0.3)

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir="./logs")]

    # Define the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
        #tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),  # kernel_regularizer=tf.keras.regularizers.l2(0.001)
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(47, activation='softmax')  # kernel_regularizer=tf.keras.regularizers.l2(0.001)

        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # tf.keras.layers.MaxPooling2D((2, 2)),
        # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D((2, 2)),
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dense(47, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adadelta',
                  loss='categorical_crossentropy',  # 'sparse_categorical_focal_loss',
                  metrics=['accuracy'])
    print(model.summary())
    # Train the model
    history = model.fit(x_train, y_train, epochs=150, validation_data=(x_test, y_test), validation_split=0.2)
    accuracy_Curve(history)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
    Confusion_Plot(model, x_test, y_test)
    return model


def accuracy_Curve(history):
    print(history.history.keys())
    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def Confusion_Plot(model, x_test, y_test):
    y_pred = np.argmax(model.predict(x_test), axis=1)
    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    print('Confusion matrix:\n', conf_matrix)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the confusion matrix
    im = ax.imshow(conf_matrix, cmap='Blues')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # file = open("mapping.txt", "r")
    # label_list = file.readlines()
    # Set tick labels
    # ax.set_xticks(np.arange(len(label_list)))
    # ax.set_yticks(np.arange(len(label_list)))
    # ax.set_xticklabels(label_list, rotation=45)
    # ax.set_yticklabels(label_list)

    # Set axis labels and title
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')

    # Loop over data dimensions and create text annotations
    # for i in range(len(label_list)):
    #    for j in range(len(label_list)):
    #        ax.text(j, i, conf_matrix[i, j], ha='center', va='center', color='white')

    # Display the figure
    plt.show()
