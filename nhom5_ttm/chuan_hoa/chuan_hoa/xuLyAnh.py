import cv2
import numpy as np

def xu_ly_anh(img):
	"""Loại bỏ phần còn thiếu của một số(nếu có)"""

	img_goc = img;
	#Mặt nạ
	kernel = np.array([[0,0,1,0,0],
					   [0,1,1,1,0],
					   [1,1,1,1,1],
					   [0,1,1,1,0],
					   [0,0,1,0,0]], np.uint8)
	#Giãn 
	img = cv2.dilate(img, kernel, iterations=1) 
	#Co
	img = cv2.erode(img, kernel, iterations=1) 

	#Tăng độ mịn ảnh để khi dùng canny không bị nhiễu ảnh
	gaussian_img = cv2.GaussianBlur(img, (5, 5), 0)

	#Chuyển sang nhị phân
	nguong = 10
	img_bw = cv2.threshold(img, nguong, 255, cv2.THRESH_BINARY)[1]

	#Tìm biên
	#canny_img = cv2.Canny(img_bw, 10, 255)

	"""
	- Contour được hiểu đơn giản là một đường cong liên kết toàn bộ các điểm 
	liên tục (dọc theo đường biên) mà có cùng màu sắc hoặc giá trị cường độ.

	- hierarchy:  thông tin về phân cấp các đường biên.
	"""
	#Tìm contour
	contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, 
		cv2.CHAIN_APPROX_SIMPLE)

	# Tìm ra diện tích từ cont
	s_cnt = [cv2.contourArea(i) for i in contours]

	#Sắp xếp contour có diện tích giảm dần
	s_sort = np.argsort(s_cnt)[::-1]
	# Vẽ bounding box cho contours có diện tích lớn nhât
	cnt = contours[0]
	img_tg = img
	x,y,w,h = cv2.boundingRect(cnt)

	#Khoanh vùng
	img_khoanh = cv2.rectangle(img_goc,(x,y),(x+w,y+h),(255,0,0),1)
	#print(s_cnt)
	#print(s_sort)
	#cv2.imshow("2",img)

	#Cắt ảnh
	img_new = img_tg[y:y+h, x:x+w]

	#Chuẩn hóa dữ liệu về 28x28
	img_new = cv2.resize(img_new, (28, 28))
	#cv2.imshow("3",img_new)
	return img_new, img_khoanh


