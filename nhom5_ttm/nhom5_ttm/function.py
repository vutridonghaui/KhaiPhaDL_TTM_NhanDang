import csv

#Lấy dữ liệu từ dataset.csv
def get_label_data(file_name):
	csv_reader = csv.reader(open(file_name, "r"))
	label = []
	for i in csv_reader:
		label.append(int(i[1]))

	return label
	pass
