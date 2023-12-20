import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
import PIL


num_classes = 4

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    
    layers.Conv2D(96, 11, strides=4),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(3, strides=2),
    
    layers.Conv2D(256, 5, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(3, strides=2),
    
    layers.Conv2D(384, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    layers.Conv2D(384, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    layers.Conv2D(256, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(3, strides=2),
    
    layers.Flatten(),
    
    layers.Dense(9216),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.4),
    
    layers.Dense(4096),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.4),
    
    layers.Dense(4096),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.4),
    
    layers.Dense(num_classes),
    layers.BatchNormalization(),
    layers.Activation('softmax')
])
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=SGD(lr=0.0001, momentum=0.9), metrics=['accuracy'])
train_ds = keras.preprocessing.image_dataset_from_directory( r'dataset/train/',
                                                               image_size=(227, 227),
                                                               validation_split=0.2,
                                                               subset="training",
                                                               seed=123,
                                                               batch_size=32,
                                                               label_mode='categorical')

val_ds = keras.preprocessing.image_dataset_from_directory( r'dataset/train/',
                                                             image_size=(227, 227),
                                                             validation_split=0.2,
                                                             subset="validation",
                                                             seed=123,
                                                             batch_size=32,
                                                             label_mode='categorical')

test_ds = keras.preprocessing.image_dataset_from_directory( r'dataset/test/',
                                                             image_size=(227, 227),
                                                             batch_size=32,
                                                             label_mode='categorical')
                                                             
                                                             
results = model.fit(
    train_ds,
    batch_size = 32,
    epochs = 50,
    validation_data = val_ds,
    verbose=1,
)

model.save("./AlexNet_ct_mix")

results.history.keys()
import matplotlib.pyplot as plt

plt.plot(results.history['loss'][1:])
plt.plot(results.history['val_loss'][1:])
plt.legend(['Training', 'Validation'])
plt.title('Training and Validation losses')
plt.xlabel('epoch')
plt.savefig("dataset/AlexNet_ct_mix.png")