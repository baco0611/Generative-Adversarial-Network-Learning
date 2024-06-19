import os
import joblib
import cv2
import numpy as np
from config import *

folder_path = "./dataset"

data = []
gray_data = []

for _ , _ , files in os.walk(folder_path):
    for file in files:
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path)
        
        image = cv2.resize(image, (data_size, data_size))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        B, G, R = cv2.split(image)

        image = np.concatenate((B.flatten() , G.flatten() , R.flatten()))
        gray_image = gray_image.flatten()

        data.append(image)
        gray_data.append(gray_image)

data = np.array(data)
gray_data = np.array(gray_data)

print(data.shape)
print(gray_data.shape)

data_path = f"./data/dataset/{date}_{data_size}_data.joblib"
gray_data_path = f"./data/dataset/{date}_{data_size}_gray_data.joblib"

joblib.dump(data, data_path)
joblib.dump(gray_data, gray_data_path)