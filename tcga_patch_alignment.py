import os

import openslide
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import h5py

# Define the following paths by yourself.
TG_10x_folder = '/home/mhduan/projectsummer/data/TCGA-BRCA-Feat/x10/h5_files'
TG_20x_folder = '/home/mhduan/projectsummer/data/TCGA-BRCA-Feat/x20/h5_files'
save_folder_20x = '/home/mhduan/projectsummer/data/TCGA-BRCA-Feat/x20/attn_mask'
slide_folder = '/media/mhduan/HIKSEMI/TCGA-BRCA'

if(not os.path.exists(save_folder_20x)):
    os.makedirs(save_folder_20x)

"""
all_data = np.array(pd.read_excel('data_BRIGHT_three.xlsx', engine='openpyxl',  header=None))
svs2uuid = {}
for i in all_data:
    svs2uuid[i[1]] = i[0]
"""
def find_alignment(file_name):
    TG_10x_file_path = os.path.join(TG_10x_folder, file_name)
    TG_20x_file_path = os.path.join(TG_20x_folder, file_name)

    TG_10x = np.array(h5py.File(TG_10x_file_path, 'r')['coords']).astype(np.int_)
    TG_20x = np.array(h5py.File(TG_20x_file_path, 'r')['coords']).astype(np.int_)

    slide_path = os.path.join(slide_folder, file_name.replace('h5', 'svs'))
    slide = openslide.open_slide(slide_path)
    shape = slide.level_dimensions[0]
    TG_10x_matrix = np.zeros((int(shape[0]/4), int(shape[1]/4)))
    TG_20x_matrix = np.zeros((int(shape[0]/4), int(shape[1]/4)))

    for i, coord in enumerate(TG_10x):
        TG_10x_matrix[int(coord[0]/4):int((coord[0]+2048)/4), int(coord[1]/4):int((coord[1]+2048)/4)] = i + 1
    for i, coord in enumerate(TG_20x):
        TG_20x_matrix[int(coord[0]/4):int((coord[0]+1024)/4), int(coord[1]/4):int((coord[1]+1024)/4)] = i + 1

    align_matrix_20x = torch.ones(TG_10x.shape[0], np.array(h5py.File(TG_20x_file_path, 'r')['coords']).shape[0])

    for i in tqdm(range(1, TG_10x.shape[0]+1)):
        cover_patches_20x = np.unique(TG_20x_matrix[TG_10x_matrix == i])
        cover_ids_20x = np.delete(cover_patches_20x, np.where(cover_patches_20x==0)) - 1
        align_matrix_20x[i-1][cover_ids_20x] = 0

    align_matrix_20x = align_matrix_20x.type(torch.bool)
    torch.save(~align_matrix_20x, os.path.join(save_folder_20x, file_name.replace('h5', 'pt')))


#pool = ThreadPoolExecutor(max_workers=48)
for file_name in os.listdir(TG_10x_folder):
    if(file_name.replace('h5', 'pt') in os.listdir(save_folder_20x)):
        print(file_name, ': have been processed!')
    else:
        find_alignment(file_name)
#        pool.submit(find_alignment, file_name)
#pool.shutdown(wait=True)
