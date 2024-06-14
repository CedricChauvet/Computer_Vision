import cv2
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from PIL import Image
import keras as K


def resize_image(img, new_width, new_height):
    #  reconvertit en image 3 channels
    if img.mode != 'RGB':
           img = img.convert("RGB")
    # met dans de bonnes dimensions       
    img = img.resize((new_width, new_height))
    return np.array(img)


imagenet = load_dataset(
    'Maysee/tiny-imagenet',
    split='valid',
    ignore_verifications=True  # set to True if seeing splits Error
)

# create a liste of each resized image 
image_arrays = [resize_image(path, 224, 224) for path in imagenet[:]["image"]]
# stack into a numpy array
stacked_images = np.stack(image_arrays,dtype=np.float32)
# print("stacked labels", stacked_images[0:5])


stacked_labels =  np.stack(imagenet[:]["label"], dtype=np.int16)

X_train, X_test, y_train, y_test = train_test_split(np.float32(stacked_images), stacked_labels, test_size=0.2, random_state=42, shuffle=True)
#print("train", X_train.shape, len(y_train), "test", X_test.shape, len(y_test))

np.random.seed(1000)

#Instantiation
AlexNet = K.Sequential()

#1st Convolutional Layer
AlexNet.add(K.layers.Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='same'))
AlexNet.add(K.layers.BatchNormalization())
AlexNet.add(K.layers.Activation('relu'))
AlexNet.add(K.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#2nd Convolutional Layer
AlexNet.add(K.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
AlexNet.add(K.layers.BatchNormalization())
AlexNet.add(K.layers.Activation('relu'))
AlexNet.add(K.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#3rd Convolutional Layer
AlexNet.add(K.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(K.layers.BatchNormalization())
AlexNet.add(K.layers.Activation('relu'))

#4th Convolutional Layer
AlexNet.add(K.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(K.layers.BatchNormalization())
AlexNet.add(K.layers.Activation('relu'))

#5th Convolutional Layer
AlexNet.add(K.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(K.layers.BatchNormalization())
AlexNet.add(K.layers.Activation('relu'))
AlexNet.add(K.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#Passing it to a Fully Connected layer
AlexNet.add(K.layers.Flatten())
# 1st Fully Connected Layer
AlexNet.add(K.layers.Dense(4096, input_shape=(32,32,3,)))
AlexNet.add(K.layers.BatchNormalization())
AlexNet.add(K.layers.Activation('relu'))
# Add Dropout to prevent overfitting
AlexNet.add(K.layers.Dropout(0.4))

#2nd Fully Connected Layer
AlexNet.add(K.layers.Dense(4096))
AlexNet.add(K.layers.BatchNormalization())
AlexNet.add(K.layers.Activation('relu'))
#Add Dropout
AlexNet.add(K.layers.Dropout(0.4))

#3rd Fully Connected Layer
AlexNet.add(K.layers.Dense(1000))
AlexNet.add(K.layers.BatchNormalization())
AlexNet.add(K.layers.Activation('relu'))
#Add Dropout
AlexNet.add(K.layers.Dropout(0.4))

#Output Layer
AlexNet.add(K.layers.Dense(10))
AlexNet.add(K.layers.BatchNormalization())
AlexNet.add(K.layers.Activation('softmax'))

#Model Summary
# AlexNet.summary()
# AlexNet.compile(
#    optimizer=K.optimizers.Adam(learning_rate=1e-3),
#    loss=K.losses.BinaryCrossentropy(),
#    metrics=[
#        K.metrics.BinaryAccuracy(),
#        K.metrics.FalseNegatives(),
#    ],
#)
# just beware of the loss, idk what is sparse_categorical_crossentropy
AlexNet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
history = AlexNet.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Évaluation du modèle
loss, accuracy = AlexNet.evaluate(X_test, y_test)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
