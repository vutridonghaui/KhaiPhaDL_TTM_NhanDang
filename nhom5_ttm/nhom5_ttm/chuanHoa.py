import os
from xuLyAnh import*

#Lấy đường dẫn hiện tại của folder chứa file chuanhoa.py
#Hàm getcwd trả về thư mục làm việc hiện tại
folder = os.getcwd() 
#Lấy danh sách các file trong folder 
ds_con = os.listdir(folder)
for i in ds_con:
	if(i == "dataset_origin"):
		folder_2 = r"{}\{}".format(folder, i)
	if(i == "dataset"):
		folder_save = r"{}\{}".format(folder, i)

#Lấy danh sách các tệp hoặc thư mục trong folder dataset_origin
ds_con = os.listdir(folder_2)
dem = 0
len_ds = len(ds_con)
for i in ds_con:
	img = cv2.imread(r'{}\{}'.format(folder_2, i), cv2.IMREAD_GRAYSCALE)
	#chuan_hoa từ xulyAnh.py
	img_norm, x = xu_ly_anh(img)
	cv2.imwrite(r'{}\{}'.format(folder_save, i), img_norm)
	dem+=1
	print(i)
#img = cv2.imread(r'{}\{}'.format(folder_2, ds_con[0]), cv2.IMREAD_GRAYSCALE)
#x = xu_ly_anh(img)
print("Success")
#cv2.imshow("1", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

