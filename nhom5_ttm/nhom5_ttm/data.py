import cv2
import os
import numpy as np
from keras.utils import to_categorical
from function import *

""" Đọc folder chứa dataset"""
def get_data():
	folder = os.getcwd()
	f_dataset = r"{}\dataset".format(folder);

	file = os.listdir(f_dataset)

	name_file=[int(i[:-4]) for i in file]
	name_file.sort();

	""" Lấy dữ liệu """
	data = [cv2.imread(r"{}\{}.png".format(f_dataset, i), 0) for i in name_file]

	""" Chuyển đổi và phân loại dữ liệu """
	d = 1
	train_x = []
	vali_x = []
	test_x = []
	for i in data:
		# # if(d==1):
		# 	x = i.reshape(-1, 784).astype(np.float32)
		# 	print(x)
		# 	cv2.imshow("0", x)
		if(d<=15000):
			x = i.reshape(784).astype(np.float32)/255
			train_x.append(x)
			d+=1
		elif d<=20000:
			x = i.reshape(784).astype(np.float32)/255
			vali_x.append(x)
			d+=1
		else:
			x = i.reshape(784).astype(np.float32)/255
			test_x.append(x)
			d+=1

	train_x = np.array(train_x)
	vali_x = np.array(vali_x)
	test_x = np.array(test_x)

	# print(name_file[15000])

	""" Lấy label"""
	train_y = []
	vali_y = []
	test_y = []
	d=1
	#Từ file function.py
	label_data = get_label_data('dataset.csv')

	train_y = label_data[:15000]
	vali_y = label_data[15000:20000]
	test_y = label_data[20000:]
	#Chuẩn hóa label
	train_y = to_categorical(train_y, 10)
	vali_y = to_categorical(vali_y, 10)
	test_y = to_categorical(test_y, 10)


	# cv2.imshow("1",train_x[0])
	#cv2.waitKey(0)
	return train_x, test_x, vali_x, train_y, test_y, vali_y
