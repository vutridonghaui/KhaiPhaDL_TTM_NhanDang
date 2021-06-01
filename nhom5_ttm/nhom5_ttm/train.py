from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
from data import*

train_x, test_x, vali_x, train_y, test_y, vali_y = get_data()

#Chuyển label về dạng chuẩn
# train_y = to_categorical(train_y, 10)
# test_y = to_categorical(test_y, 10)

""" Xây dựng models """
model = models.Sequential()

#Tạo cấu trúc mạng

#Layer Hidden 1: 128 neuron - sigmoid - input(28x28)
model.add(layers.Dense(128, activation='sigmoid', input_shape=(28*28,)))

#Layer Hidden 2: 56 neuron - linear
model.add(layers.Dense(56, activation='linear'))

#Layer Output: 10 neuron - softmax
model.add(layers.Dense(10, activation='softmax'))

#learning rate (lr)
#mse: sai số bình phương trung bình
#loss, optimizer: 
sgd = optimizers.SGD(lr=0.1)
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

""" Huấn luyện """
#epoch: số lần lặp sau khi hoàn 1 batch 
model.fit(train_x, train_y, epochs=40, batch_size=150,validation_data= (vali_x, vali_y))

""" Testing model"""

print("\n=============================\n")
print("Testing...\n")

score = model.evaluate(test_x, test_y, verbose=0)
print("Test loss: %.4f"% score[0])
print("Test accuracy: %.4f"% score[1])
print("Model chính xác {}%".format(round(score[1]*100, 4)))

""" Lưu model """

model.save("model.h5")
print("Lưu thành công")