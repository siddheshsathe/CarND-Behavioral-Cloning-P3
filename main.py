import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import nvidiaModel


def loadData(dataDir):
    """
    This function is to laod the data.
    The data was kept under ../behavioral_cloning_data/Track1/<multiple folder containing IMG and driving_log.csv>/
    """
    images = []
    measurements = []
    for dataDirectory in dataDir:
        imagesPath = os.path.join(os.getcwd(), '..', 'behavioral_cloning_data', 'Track1', dataDirectory, 'IMG')
        lines = []
        correctionAngle = 0.2
        with open('../behavioral_cloning_data/Track1/'+dataDirectory+'/driving_log.csv') as csvFile:
            reader = csv.reader(csvFile)
            for line in reader:
                lines.append(line)
        for line in lines:
            
            # Center image
            src = line[0]
            fil = src.split('\\')[-1]
            currentPath = os.path.join(imagesPath, fil)
            img = cv2.imread(currentPath)
            images.append(img)
            measurements.append(float(line[3]))
            
            # Center image flipping to augument data
            src = line[0]
            fil = src.split('\\')[-1]
            currentPath = os.path.join(imagesPath, fil)
            img = cv2.imread(currentPath)
            img = cv2.flip(img, 1)
            images.append(img)
            measurements.append(-1 * float(line[3]))
            
            # Left image
            src = line[1]
            fil = src.split('\\')[-1]
            currentPath = os.path.join(imagesPath, fil)
            img = cv2.imread(currentPath)
            images.append(img)
            measurements.append(float(line[3]) + correctionAngle)
            
            # Left image flipping to augument data
            src = line[1]
            fil = src.split('\\')[-1]
            currentPath = os.path.join(imagesPath, fil)
            img = cv2.imread(currentPath)
            img = cv2.flip(img, 1)
            images.append(img)
            measurements.append(-1 * (float(line[3]) + correctionAngle))
            
            # Right image
            src = line[2]
            fil = src.split('\\')[-1]
            currentPath = os.path.join(imagesPath, fil)
            img = cv2.imread(currentPath)
            images.append(img)
            measurements.append(float(line[3]) - correctionAngle)
            
            # Right image flipping to augument data
            src = line[2]
            fil = src.split('\\')[-1]
            currentPath = os.path.join(imagesPath, fil)
            img = cv2.imread(currentPath)
            img = cv2.flip(img, 1)
            images.append(img)
            measurements.append(-1 * (float(line[3]) - correctionAngle))
            
    X_train = np.array(images)
    y_train = np.array(measurements)
    del images, measurements
    return X_train, y_train

def main():
    dataDir = os.listdir(os.path.join(os.getcwd(), '..', 'behavioral_cloning_data', 'Track1'))
    X_train , y_train = loadData(dataDir)
    ourmodel = nvidiaModel()
    ourmodel.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    ourmodel.summary()

    history = ourmodel.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=30)

    ourmodel.save('model.h5')
    # list all data in history
    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

main()
