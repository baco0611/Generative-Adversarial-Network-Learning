from keras.datasets import mnist
import cv2

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

i = 1
for x in X_train:
    x.reshape(28, 28)
    image = cv2.resize(x, (64, 64))
    cv2.imwrite(f"./dataset/{i}.jpg", image)
    i+=1
