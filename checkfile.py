import os
import csv

csv_file1 = '/home/mhduan/projectsummer/data/TCGA-NSCLC PLIP/label.csv'

csv_file2 = '/home/mhduan/projectsummer/MIL/MG-Mamba-Cross-Concat-Wikg-Ins/dataset_csv/NSCLC_label.csv'

file1 = []

file2 = []

with open(csv_file2, mode='r') as file:
    reader = csv.reader(file)
    i = 0 
    for row in reader:
        if (i == 0):
            i = i + 1
            continue
        if (i != 0):
            file_name_item = row[1].split('-')  
            file2.append(file_name_item[0]+'-'+file_name_item[1]+'-'+file_name_item[2])
            i = i + 1
        #file_name = os.path.join(folder_path, file_name+'.pt')
file2.append('sss')

with open(csv_file1, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        file_name_item = row[0]
        file1.append(file_name_item)

for file_item in file2:
    if file_item not in file1:
        print(file_item)