from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Conv2D
from keras.layers import Activation
from keras.optimizers import SGD

def nvidiaModel():
    activation = 'relu'
    model = Sequential()
    # Removing the unnecessary part from image like sky and mountains nearby
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    
    # Normalization
    model.add(Lambda(lambda x: (x)/127. -1))
    
    # Layer1: 24 filters of 5x5
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation=activation))
    
    #Layer2: 36 filters of 5x5
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation=activation))
    
    # Layer3: 48 filters of 5x5
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation=activation))
    
    #Layer4: 64 filters of 3x3
    model.add(Conv2D(filters=64, kernel_size=3, activation=activation))
    
    #Layer5: 64 filters of 3x3
    model.add(Conv2D(filters=64, kernel_size=3, activation=activation))
    
    #Layer8: Flattening
    model.add(Flatten())
    
    #Layer7: 100 Fully connected
    model.add(Dense(100))
    
    #Layer8: 50 Fully connected
    model.add(Dense(50))
    
    #Layer9: 10 Fully connected
    model.add(Dense(10))
    
    #Layer10: 1 Fully connected
    model.add(Dense(1))
    
    return model
    
