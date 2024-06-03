from config import *
import joblib
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, LeakyReLU
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time

start_time = time.time()

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

data = joblib.load(f"./data/dataset/{date}_{data_size}_gray_data.joblib")
data = data.astype('float32')/255

z_dim = 100

adam = Adam(learning_rate=0.0002, beta_1=0.5)
optimizer_g = Adam(learning_rate=0.0002, beta_1=0.5)
optimizer_d = Adam(learning_rate=0.0002, beta_1=0.5)

g = Sequential()
g.add(Dense(256, input_dim=z_dim, activation=LeakyReLU(alpha=0.2)))
g.add(Dense(512, activation=LeakyReLU(alpha=0.2)))
g.add(Dense(1024, activation=LeakyReLU(alpha=0.2)))
g.add(Dense(2048, activation=LeakyReLU(alpha=0.2)))
g.add(Dense(data_size * data_size, activation="sigmoid"))
g.compile(loss='binary_crossentropy', optimizer=optimizer_g)
g.summary()

d = Sequential()
d.add(Dense(2048, input_dim=data_size * data_size, activation=LeakyReLU(alpha=0.2)))
d.add(Dropout(0.3))
d.add(Dense(1024, activation=LeakyReLU(alpha=0.2)))
d.add(Dropout(0.3))
d.add(Dense(512, activation=LeakyReLU(alpha=0.2)))
d.add(Dropout(0.3))
d.add(Dense(256, activation=LeakyReLU(alpha=0.2)))
d.add(Dropout(0.3))
d.add(Dense(1, activation='sigmoid'))
d.compile(loss='binary_crossentropy', optimizer=optimizer_d, metrics=['accuracy'])
d.summary()

d.trainable = False
inputs = Input(shape=(z_dim, ))
hidden = g(inputs)
output = d(hidden)
gan = Model(inputs, output)
gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# Hàm vẽ loss function
def plot_loss(losses):
    d_loss = [v[0] for v in losses["D"]]
    g_loss = [v[0] for v in losses["G"]]
    
    plt.figure(figsize=(10,8))
    plt.plot(d_loss, label="Discriminator loss")
    plt.plot(g_loss, label="Generator loss")
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"./{model_name}/image/loss.png")    
    # plt.show()

# Hàm vẽ sample từ Generator
def plot_generated(epoch, n_ex=10, dim=(1, 10), figsize=(12, 2), model_name="gan"):
    noise = np.random.normal(0, 1, size=(n_ex, z_dim))
    generated_images = g.predict(noise)
    generated_images = generated_images.reshape(n_ex, data_size, data_size)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
        plt.tight_layout()
    plt.savefig(f"./{model_name}/image/{epoch}.png")    
    plt.tight_layout()
    # plt.show()


# Lưu giá trị loss và accuracy của Discriminator và Generator
losses = {"D":[], "G":[]}

def train(epochs=1, plt_frq=1, BATCH_SIZE=128, model_name="GAN"):
    # Tạo thư mục để lưu ảnh và mô hình
    image_dir = f"./{model_name}"
    make_dir(image_dir + "/image")
    make_dir(image_dir + "/model")

    # Tính số lần chạy trong mỗi epoch
    batchCount = int(data.shape[0] / BATCH_SIZE)
    print('Epochs:', epochs)
    print('Batch size:', BATCH_SIZE)
    print('Batches per epoch:', batchCount)
    
    for e in tqdm(range(1, epochs+1), desc='Epochs', total=epochs):
        if e == 1 or e%plt_frq == 0:
            print("\n\n\n", '-'*15, 'Epoch %d' % e, '-'*15)
        for _ in range(batchCount):
            # Lấy ngẫu nhiên các ảnh từ MNIST dataset (ảnh thật)
            image_batch = data[np.random.randint(0, data.shape[0], size=BATCH_SIZE)]
            # Sinh ra noise ngẫu nhiên
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
            
            # Dùng Generator sinh ra ảnh từ noise
            generated_images = g.predict(noise)
            X = np.concatenate((image_batch, generated_images))
            # Tạo label
            y = np.zeros(2*BATCH_SIZE)
            y[:BATCH_SIZE] = 0.9  # gán label bằng 1 cho những ảnh từ MNIST dataset và 0 cho ảnh sinh ra bởi Generator

            # Train discriminator
            d.trainable = True
            d_loss = d.train_on_batch(X, y)

            # Train generator
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
            # Khi train Generator gán label bằng 1 cho những ảnh sinh ra bởi Generator -> cố gắng lừa Discriminator. 
            y2 = np.ones(BATCH_SIZE)
            # Khi train Generator thì không cập nhật hệ số của Discriminator.
            d.trainable = False
            g_loss = gan.train_on_batch(noise, y2)

        # Lưu loss function
        losses["D"].append(d_loss)
        losses["G"].append(g_loss)

        # Vẽ các số được sinh ra để kiểm tra kết quả
        if e == 1 or e%plt_frq == 0:
            plot_generated(epoch = e, model_name=model_name)
    plot_loss(losses)
    gan.save(f"./{model_name}/model/gan_model.h5")
    g.save(f"./{model_name}/model/g_model.h5")
    d.save(f"./{model_name}/model/d_model.h5")


model_name = "GAN_1"
train(epochs=100, plt_frq=1, BATCH_SIZE=128, model_name=model_name)

# End and calculate time
end_time = time.time()
execution_time = end_time - start_time
print("\n\n", "-"*40)
print("Thời gian thực thi: {:.5f} giây".format(execution_time))