import os

folder_path = '/home/mhduan/projectsummer/data/TCGA-NSCLC PLIP/pt_files'

file_list = os.listdir(folder_path)

for filename in file_list:
    file_path = os.path.join(folder_path, filename)
    new_file_path = file_path.split('.')[0]+'.'+file_path.split('.')[-1]
    os.rename(file_path, new_file_path)