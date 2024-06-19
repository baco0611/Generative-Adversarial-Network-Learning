from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Softmax, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

# Định nghĩa mô hình
input_layer = Input(shape=(256, 256, 3))

x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = GlobalAveragePooling2D()(x)

dense_1 = Dense(256, activation='relu')(x)
dense_2 = Dense(256, activation='relu')(x)
dense_3 = Dense(256, activation='relu')(x)

dense_1 = Dense(192, activation='relu')(dense_1)
dense_2 = Dense(192, activation='relu')(dense_2)
dense_3 = Dense(192, activation='relu')(dense_3)

softmax_1 = Softmax()(dense_1)
softmax_2 = Softmax()(dense_2)
softmax_3 = Softmax()(dense_3)

concatenated = Concatenate()([softmax_1, softmax_2, softmax_3])

model = Model(inputs=input_layer, outputs=concatenated)

# Vẽ sơ đồ mô hình và lưu thành file PNG
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
