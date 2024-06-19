import pandas as pd
import joblib
import numpy as np
import cv2

csv_path = "./chineseMNIST.csv"

dataset = pd.read_csv(csv_path)

data = []

for index, row in dataset.iterrows():
    row_array = row.values.tolist()
    # Thêm mảng nhỏ vào mảng lớn
    data.append(row_array[:-2])

    image = np.array(row_array[:-2]).reshape(64, 64)

    cv2.imwrite(f"./dataset/{index}.jpg", image)

data = np.array(data)
joblib.dump(data, "./data/dataset/data.joblib")