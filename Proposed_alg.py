# coding: utf-8

from keras.layers import Input, Lambda, Dense, Flatten, concatenate
from keras.models import Model
from keras.applications.vgg19 import VGG19, preprocess_input as vgg_preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import GlobalAveragePooling2D
# Set the image size
IMAGE_SIZE = [224, 224]
# VGG19
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False

# InceptionV3
inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in inception.layers:
    layer.trainable = False

# Add GlobalAveragePooling2D layer to each model output
vgg_output = GlobalAveragePooling2D()(vgg.output)
inception_output = GlobalAveragePooling2D()(inception.output)

# Concatenate outputs from VGG19 and InceptionV3
concatenated = concatenate([vgg_output, inception_output])

# Add more layers as needed
x = Dense(1000, activation='relu')(concatenated)
prediction = Dense(4, activation='softmax')(x)


# Create a model object
model = Model(inputs=[vgg.input, inception.input], outputs=prediction)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=vgg_preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(preprocessing_function=vgg_preprocess_input)

training_set = train_datagen.flow_from_directory('Dataset1/train',
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('Dataset1/test',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')
# Fit the model
r = model.fit(
    x=[training_set.next()[0], training_set.next()[0]],  # Provide both VGG and Inception data
    y=training_set.next()[1],
    validation_data=([test_set.next()[0], test_set.next()[0]], test_set.next()[1]),  # Provide both VGG and Inception validation data
    validation_steps=len(test_set),
    epochs=50,
    steps_per_epoch=len(training_set)
)



# Plot loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# Plot accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

# Save the model
model.save('Combined_Model.h5')
