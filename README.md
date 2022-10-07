# ComputerVision_Face2Age
The objective of the model is to Predict the age of a customer from an image of thier face. <br>
I used ResNet50 neural network architechture and images from the supermarket chain Good Seed. <br>

## script to run on the GPU platform 

In[ ]: 

import pandas as pd <br>
import tensorflow as tf <br>
from tensorflow.keras.preprocessing.image import ImageDataGenerator <br>
from tensorflow.keras.applications.resnet import ResNet50 <br>
from tensorflow.keras.models import Sequential <br>
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Flatten <br>
from tensorflow.keras.optimizers import Adam <br>

    def load_train(path):
    
      labels = pd.read_csv(path+'labels.csv')
      datagen = ImageDataGenerator(rescale = 1/255, validation_split=0.25)
    
      train_gen_flow = datagen.flow_from_dataframe(
      dataframe=labels,
      directory=path+'final_files/',
      x_col='file_name',
      y_col='real_age',
      target_size=(150, 150),
      batch_size=16,
      class_mode='raw',
      subset='training',
      horizontal_flip=True,
      seed=12349)

      return train_gen_flow

    def load_test(path):
    
      labels = pd.read_csv(path+'labels.csv')
      datagen = ImageDataGenerator(rescale = 1/255, validation_split=0.25)

      test_gen_flow = datagen.flow_from_dataframe(
      dataframe=labels,
      directory=path+'final_files/',
      x_col='file_name',
      y_col='real_age',
      target_size=(150, 150),
      batch_size=16,
      class_mode='raw',
      subset='validation',
      seed=12349)

      return test_gen_flow

    def create_model(input_shape):

      model = Sequential()
      model.add(ResNet50(input_shape=input_shape, weights='imagenet', include_top=False))
      model.add(GlobalAveragePooling2D())
      model.add(Flatten())
      model.add(Dense(units=120, activation='relu'))
      model.add(Dense(units=84, activation='relu'))
      model.add(Dense(units=12, activation='linear'))

      model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error', metrics=['mae'])

      return model

    def train_model(model, train_data, test_data, batch_size=None, epochs=10,
        steps_per_epoch=None, validation_steps=None):

      if steps_per_epoch is None:
          steps_per_epoch = len(train_data)
      if validation_steps is None:
          validation_steps = len(test_data)

      model.fit(train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2)

      return model
