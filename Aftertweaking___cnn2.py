from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential 
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
import os.path


#convolution layer 2 variables
filters2=32
#fully connected layer 1 variables
units1=256
#Activation function
activationconvo='relu'
actiavtiondense='relu'
# fit variables
steps_per_epoch1=1284
validation_steps1=4297
epochs1=3

model = Sequential() 
#Convolution Layer-1
model.add(Convolution2D(filters=16, kernel_size=(3,3),activation='relu',input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=(2, 2),))
model.summary()

#Convolution layer-2
model.add(Convolution2D(filters=filters2,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.summary()

model.add(Convolution2D(filters=64,kernel_size=(3,3),activation=activationconvo))





#Flattening layer
model.add(Flatten())

#Fully connected layer 1

model.add(Dense(units=units1,activation='relu'))
model.add(Dense(units=128,activation=actiavtiondense))







#Output layer
model.add(Dense(units=26,activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        '/cnn/furit2/Traning',
        target_size=(100,100),
        batch_size=32,
        class_mode='categorical')
test_set = test_datagen.flow_from_directory(
        '/cnn/furit2/Test',
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical')
history = model.fit(
        training_set,
        steps_per_epoch=steps_per_epoch1,
        epochs=epochs1,
        validation_data=test_set,
        validation_steps=validation_steps1)

a = history.history['accuracy'][2]

save_to_path = '/cnn'
name_of_file = "AccuracyCNN"
complete_name = os.path.join(save_to_path, name_of_file+".txt")
print(a, file = open(complete_name,"a"))


