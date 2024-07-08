import _init_path
import os
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path

def check_string_in_list(string_list, target_string):
    for string in string_list:
        if str(string) in target_string:
            return True
    return False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

folder_path = r"D:\Sharing\Programs\LaserAttack\src\color_strip\physical-test\finaltest0707\AA-yolov5-5kmh\exp13\labels"
folder_path = Path(folder_path)

frames = 330
window_size = 150
target_index = 9 # None for No Detected
result = []


txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]
first_column_data = []


for i in range(1, frames, 1):
    list_tmp = []
    file_name = '_'.join(txt_files[0].split('_')[0:-1])
    file_name_ = file_name + f"_{i}.txt"
    if file_name_ in txt_files:
        with open(os.path.join(folder_path, file_name_), 'r') as file:
            lines = file.readlines()
            for line in lines:
                list_tmp.append(int(line.split()[0]))
            first_column_data.append(list_tmp)
    else:
        first_column_data.append([None])

print(first_column_data)

for i in range(len(first_column_data) - window_size + 1):
    window = first_column_data[i:i+window_size]
    target_count = sum(1 for sublist in window if isinstance(sublist, list) and target_index in sublist)
    result.append(target_count)

print("Results:", result)
print("Max: ", str(max(result)) + ";\nASR: " + str(max(result)*100 / window_size)+"%")
