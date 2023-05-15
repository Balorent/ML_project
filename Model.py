import tensorflow as tf
from numpy import argmax
import emnist as emn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load the EMNIST dataset
def create_model():
    #(x_train, y_train), (x_test, y_test) = emnist.load_data(type='balanced')
    x_train, y_train = emn.extract_training_samples('balanced')
    x_test, y_test = emn.extract_test_samples('balanced')
    # Preprocess the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    y_train = tf.keras.utils.to_categorical(y_train, 47)
    y_test = tf.keras.utils.to_categorical(y_test, 47)

    # Define the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        #tf.keras.layers.Dropout(0.25),
        #tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu' ),#kernel_regularizer=tf.keras.regularizers.l2(0.001)
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(47, activation='softmax' )#kernel_regularizer=tf.keras.regularizers.l2(0.001)

        #tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        #tf.keras.layers.MaxPooling2D((2, 2)),
        #tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        #tf.keras.layers.MaxPooling2D((2, 2)),
        #tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(128, activation='relu'),
        #tf.keras.layers.Dense(47, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test),validation_split=0.2)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
    Confusion_Plot(model, x_test, y_test)
    return model


def Confusion_Plot(model, x_test, y_test):
    y_pred = argmax(model.predict(x_test), axis=1)
    conf_matrix = confusion_matrix(argmax(y_test, axis=1), y_pred)
    print('Confusion matrix:\n', conf_matrix)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the confusion matrix
    im = ax.imshow(conf_matrix, cmap='Blues')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)


    #file = open("mapping.txt", "r")
    #label_list = file.readlines()
    # Set tick labels
    #ax.set_xticks(np.arange(len(label_list)))
    #ax.set_yticks(np.arange(len(label_list)))
    #ax.set_xticklabels(label_list, rotation=45)
    #ax.set_yticklabels(label_list)

    # Set axis labels and title
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')

    # Loop over data dimensions and create text annotations
    #for i in range(len(label_list)):
    #    for j in range(len(label_list)):
    #        ax.text(j, i, conf_matrix[i, j], ha='center', va='center', color='white')

    # Display the figure
    plt.show()

