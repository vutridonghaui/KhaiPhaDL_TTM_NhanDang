from keras.models import load_model
from data import*


#Từ file data.py
x1, test_x, x2, x3, test_y, x4 = get_data()

#Load model
model = load_model("model.h5")
print("Load thành công")

#Đánh giá với test_x là dữ liệu và test_y là label
score = model.evaluate(test_x, test_y, verbose=0)
print("Test loss: %.4f"% score[0])
print("Test accuracy: %.4f"% score[1])
print("Model chính xác {}%".format(round(score[1]*100, 4)))