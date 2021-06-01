import os
import cv2
import numpy as np
from keras.models import load_model
from xuLyAnh import*

def dinh_danh(label, preds, img):
	#Tạo khung

	img = cv2.copyMakeBorder(img, 2, 2, 2, 2, 
		cv2.BORDER_CONSTANT, 
		value = [255])

	#Vùng tiêu đề
	black = np.zeros((50, img.shape[1], 1), np.uint8)
	black[:] = [255]

	""" - vconcat: hàm nối ảnh theo chiều dọc"""
	img = cv2.vconcat((img, black))

	""" Tham số trong cv2.putText:
		- (1) : ảnh
		- (2) : text
		- (3) : x, y
		- (4) : font
		- (5) : kích thước chữ
		- (6) : màu sắc
		- (7) : độ dày của chữ
	"""
	#Thêm chữ vào
	font = cv2.FONT_HERSHEY_SIMPLEX
	h, w = img.shape

	cv2.putText(img, "So {}".format(int(pred_label)), (20,h-30), font, 0.6, (0), 2)

	cv2.putText(img, "({}%)".format(round(np.max(preds)*100,2)), 
													  (10,h-10), font, 0.5, (0), 1, 1)
	cv2.imshow("0", img)
	pass

"""
	Hàm:
	- os.getcwd: lấy đường link hiện tại của file đang chạy
	- os.listdir(f): lấy danh sách các thư mục và file con trong đường dẫn f
	- cv2.IMREAD_GRAYSCALE: tham số đọc ảnh thành ảnh xám
"""


#Lấy đường dẫn hiện tại
folder = os.getcwd()
ds_con = os.listdir(folder)
for i in ds_con:
	if(i == "data_nhandang"):
		folder_2 = r"{}\{}".format(folder, i)

ds_con = os.listdir(folder_2)

img = cv2.imread(r'{}\3.png'.format(folder_2), cv2.IMREAD_GRAYSCALE)

#chuẩn hóa từ file xulyAnh.py
img_norm, img_khoanh = xu_ly_anh(img)

""" Load model"""
model = load_model("model.h5")
x = img_norm.reshape(-1,784).astype(np.float32)/255
#cv2.imshow("0", img_khoanh)
"""
 - preds: Độ chính xác của số nhận dạng
 - pred_label: Label của số được nhận dạng
 - predict(x): hàm dự đoán từ model
"""
preds = model.predict(x)

#Lấy giá trị cao nhất làm lớp dự đoán
pred_label=np.argmax(preds,axis=1)

print("Ảnh này là số {} ({}%)".format(pred_label, round(np.max(preds)*100,2)))

dinh_danh(pred_label, preds, img_khoanh)

cv2.destroyAllWindows()



#Đọc vào hiển thị ảnh
#img = cv2.imread(r'{}\0.png'.format(folder), cv2.IMREAD_GRAYSCALE)

#chuan_hoa từ xulyAnh.py
#img = chuan_hoa(img)